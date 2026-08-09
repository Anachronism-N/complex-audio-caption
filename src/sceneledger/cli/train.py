"""``sceneledger-train`` CLI: LoRA SFT on MOSS-Audio-4B.

B1 (static SFT): ordinary token CE on atomic-token captions.
B2 (TAC paper-spec): time-weighted CE (``timestamp_weight=5.0``).

::

    conda run -n moss-audio python -m sceneledger.cli.train \
      --config configs/model/b1_static_sft.yaml

The training loop:
1. Load MOSS model + processor (frozen).
2. Apply LoRA to the LLM's attention + MLP projections.
3. For each sample: process audio+prompt → input_ids; tokenize target;
   append target to input_ids; set labels (prompt masked with -100).
4. Forward → loss (built-in CE for B1, time-weighted CE for B2).
5. Gradient accumulation + cosine LR + checkpoint save.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml


def _load_config(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load_model_and_processor(cfg: dict):
    import sys as _sys

    repo = Path(__file__).resolve().parents[2] / "third_party" / "MOSS-Audio"
    if str(repo) not in _sys.path:
        _sys.path.insert(0, str(repo))
    from src.modeling_moss_audio import MossAudioModel
    from src.processing_moss_audio import MossAudioProcessor

    mcfg = cfg["model"]
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[
        mcfg.get("dtype", "bfloat16")
    ]
    model = MossAudioModel.from_pretrained(
        mcfg["path"],
        trust_remote_code=True,
        dtype=mcfg.get("dtype", "auto"),
        device_map=mcfg["device"],
    )
    model.eval()
    processor = MossAudioProcessor.from_pretrained(
        mcfg["path"], trust_remote_code=True, enable_time_marker=True
    )
    return model, processor, dtype


def _apply_lora(model, lora_cfg: dict):
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=lora_cfg["rank"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg.get("dropout", 0.05),
        target_modules=lora_cfg["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def _load_audio(path: str, sample_rate: int, max_seconds: float) -> np.ndarray:
    import soundfile as sf
    from scipy.signal import resample_poly
    from math import gcd

    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if sr != sample_rate:
        g = gcd(int(sr), int(sample_rate))
        wav = resample_poly(wav.astype(np.float64), int(sample_rate) // g, int(sr) // g).astype(np.float32)
    max_n = int(max_seconds * sample_rate)
    if wav.shape[0] > max_n:
        wav = wav[:max_n]
    return wav


def _build_training_sample(
    audio_path: str,
    prompt: str,
    target_text: str,
    processor,
    sample_rate: int,
    max_seconds: float,
    device: str,
    dtype: torch.dtype,
) -> dict:
    """Process one sample into model inputs with labels for loss."""
    raw_audio = _load_audio(audio_path, sample_rate, max_seconds)
    inputs = processor(text=prompt, audios=[raw_audio], return_tensors="pt")

    input_ids = inputs["input_ids"]  # [1, prompt_len]
    attention_mask = inputs["attention_mask"]
    audio_data = inputs.get("audio_data")
    audio_seqlens = inputs.get("audio_data_seqlens")

    # tokenize target (no special tokens — the caption is the assistant response)
    target_ids = processor.tokenizer.encode(target_text, add_special_tokens=False)
    eos_id = processor.tokenizer.eos_token_id
    target_ids = target_ids + [eos_id]

    prompt_len = input_ids.shape[1]
    target_tensor = torch.tensor([target_ids], dtype=torch.long)

    # concatenate: [prompt_ids | target_ids]
    full_input_ids = torch.cat([input_ids, target_tensor], dim=1).to(device)
    full_attention_mask = torch.cat(
        [attention_mask, torch.ones_like(target_tensor)], dim=1
    ).to(device)

    # labels: -100 on prompt, real ids on target
    labels = torch.full_like(full_input_ids, -100)
    labels[:, prompt_len:] = target_tensor.to(device)

    # audio_input_mask: only prompt positions that are audio tokens
    audio_input_mask = (full_input_ids == processor.audio_token_id).to(device)

    out = {
        "input_ids": full_input_ids,
        "attention_mask": full_attention_mask,
        "labels": labels,
        "audio_input_mask": audio_input_mask,
    }
    if audio_data is not None:
        out["audio_data"] = audio_data.to(device).to(dtype)
        out["audio_data_seqlens"] = audio_seqlens.to(device)
    return out


def _cosine_schedule(step: int, warmup: int, total: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sceneledger-train")
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke-test", action="store_true", help="1 step, no save")
    parser.add_argument("--max-steps", type=int, default=None, help="override config steps")
    args = parser.parse_args(argv)

    cfg = _load_config(args.config)
    _set_seed(cfg["train"]["seed"])
    tcfg = cfg["train"]
    steps = args.max_steps or tcfg["steps"]

    print("[train] loading model ...", file=sys.stderr, flush=True)
    model, processor, dtype = _load_model_and_processor(cfg)
    device = cfg["model"]["device"]

    # gradient checkpointing
    if tcfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    model = _apply_lora(model, cfg["lora"])
    model.train()

    # dataset
    from sceneledger.data.manifests import read_manifest
    from sceneledger.data.datamodule import group_split, MOSS_INPUT_SAMPLE_RATE
    from sceneledger.models.target_formatter import (
        canonical_prompt,
        format_atomic_caption,
        StyleConfig,
    )

    entries = read_manifest(cfg["data"]["manifest_path"])
    train_entries, _ = group_split(
        entries, val_fraction=cfg["data"].get("val_fraction", 0.1),
        group_key=cfg["data"].get("group_key", "source_id"),
        seed=cfg["train"]["seed"],
    )
    print(f"[train] {len(train_entries)} training samples", file=sys.stderr, flush=True)

    # precompute timestamp token IDs for B2
    timestamp_weight = cfg["loss"].get("timestamp_weight", 1.0)
    ts_token_ids = None
    if timestamp_weight != 1.0:
        from sceneledger.losses.weighted_ce import compute_timestamp_token_ids
        ts_token_ids = compute_timestamp_token_ids(processor.tokenizer)
        print(f"[train] time-weighted CE: {len(ts_token_ids)} timestamp token IDs, weight={timestamp_weight}", file=sys.stderr, flush=True)

    # optimizer
    from torch.optim import AdamW

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=tcfg["learning_rate"],
        weight_decay=tcfg.get("weight_decay", 0.0),
    )

    grad_accum = tcfg["global_effective_batch"] // tcfg["micro_batch_size"]
    style_cfg = StyleConfig()
    prompt_text = canonical_prompt(style=cfg["data"].get("style", "brief"))
    audio_base = cfg["data"]["audio_base_dir"]
    sr = cfg["data"].get("sample_rate", MOSS_INPUT_SAMPLE_RATE)
    max_sec = cfg["data"].get("max_audio_seconds", 30.0)

    rng = random.Random(cfg["train"]["seed"])
    indices = list(range(len(train_entries)))

    print(f"[train] starting training: {steps} steps, effective batch {tcfg['global_effective_batch']}", file=sys.stderr, flush=True)
    t0 = time.time()
    accum_loss = 0.0
    step = 0
    epoch = 0

    while step < steps:
        rng.shuffle(indices)
        epoch += 1
        for idx in indices:
            if step >= steps:
                break
            entry = train_entries[idx]
            sid = entry.scene["scene_id"]
            audio_path = str(Path(audio_base) / entry.mixture_path)

            from sceneledger.data.schema import Ledger
            ledger = Ledger.model_validate(entry.target_ledger)
            target = format_atomic_caption(ledger, style=cfg["data"].get("style", "brief"), cfg=style_cfg)

            try:
                batch = _build_training_sample(
                    audio_path, prompt_text, target, processor, sr, max_sec, device, dtype
                )
            except Exception as exc:
                print(f"[train] SKIP {sid}: {exc}", file=sys.stderr, flush=True)
                continue

            # forward
            if timestamp_weight == 1.0:
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    audio_data=batch.get("audio_data"),
                    audio_data_seqlens=batch.get("audio_data_seqlens"),
                    audio_input_mask=batch["audio_input_mask"],
                    labels=batch["labels"],
                )
                loss = outputs.loss
            else:
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    audio_data=batch.get("audio_data"),
                    audio_data_seqlens=batch.get("audio_data_seqlens"),
                    audio_input_mask=batch["audio_input_mask"],
                    labels=None,
                )
                from sceneledger.losses.weighted_ce import time_weighted_ce_loss
                loss = time_weighted_ce_loss(
                    outputs.logits, batch["labels"], ts_token_ids, timestamp_weight
                )

            loss = loss / grad_accum
            loss.backward()
            accum_loss += loss.item()

            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    tcfg.get("max_grad_norm", 1.0),
                )
                lr = _cosine_schedule(step, tcfg["warmup_steps"], steps, tcfg["learning_rate"])
                for pg in optimizer.param_groups:
                    pg["lr"] = lr
                optimizer.step()
                optimizer.zero_grad()

            step += 1
            if step % 10 == 0:
                elapsed = time.time() - t0
                print(
                    f"[train] step {step}/{steps} epoch={epoch} loss={accum_loss/max(1,grad_accum):.4f} "
                    f"lr={lr:.2e} ({elapsed:.0f}s, {step/elapsed:.1f} step/s)",
                    file=sys.stderr, flush=True,
                )
                accum_loss = 0.0

            if args.smoke_test and step >= 1:
                print("[train] smoke test passed (1 step)", file=sys.stderr, flush=True)
                return 0

    # save checkpoint
    out_dir = Path(tcfg.get("output_dir", cfg.get("train", {}).get("output_dir", "outputs/b1")))
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir / "lora")
    processor.tokenizer.save_pretrained(out_dir / "lora")
    cfg_out = {"config_path": str(args.config), "steps": step, "train_samples": len(train_entries)}
    (out_dir / "train_config.json").write_text(json.dumps(cfg_out, indent=2), encoding="utf-8")
    print(f"[train] saved LoRA checkpoint to {out_dir / 'lora'}", file=sys.stderr, flush=True)
    print(f"[train] done in {time.time()-t0:.0f}s", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

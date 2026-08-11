"""B3-slot-aware + CARC: counterfactual acoustic remix consistency training.

For each training sample, with probability p_carc:
1. Load stems (individual source audio)
2. Remove one random stem → sum remaining → new mixture
3. Remove the corresponding source's events from the target ledger
4. Train on the (removed mixture, removed target) pair

This teaches the model: "removing a source removes its events" — a
counterfactual consistency signal that reduces hallucination and improves
source attribution.

The rest of the time (1-p_carc): train on the original (mixture, target).
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml


def _load_config(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _load_audio(path, sr_target, max_sec):
    from math import gcd

    import soundfile as sf
    from scipy.signal import resample_poly
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if sr != sr_target:
        g = gcd(int(sr), int(sr_target))
        wav = resample_poly(wav.astype(np.float64), int(sr_target)//g, int(sr)//g).astype(np.float32)
    return wav[:int(max_sec * sr_target)]


def _load_stems_and_maybe_remove(entry, audio_base, sr, max_sec, rng, carc_prob):
    """Load mixture, optionally remove a stem for CARC.

    Returns (audio_tensor, ledger, removed_source_id_or_None).
    """
    sid = entry.scene["scene_id"]
    sources = entry.scene["sources"]
    from sceneledger.data.schema import Ledger

    ledger = Ledger.model_validate(entry.target_ledger)

    if rng.random() < carc_prob and len(sources) > 1:
        # CARC: remove one random source
        remove_idx = rng.randint(0, len(sources) - 1)
        removed = sources[remove_idx]
        removed_id = removed["source_id"]

        # sum remaining stems
        mixture = None
        for i, src in enumerate(sources):
            if i == remove_idx:
                continue
            stem_path = str(Path(audio_base) / "audio" / "stems" / f"{sid}_{src['source_id']}.wav")
            try:
                stem = _load_audio(stem_path, sr, max_sec)
            except Exception:
                continue
            if mixture is None:
                mixture = np.zeros_like(stem)
            min_len = min(len(mixture), len(stem))
            mixture[:min_len] += stem[:min_len]

        if mixture is None:
            mixture = _load_audio(str(Path(audio_base) / entry.mixture_path), sr, max_sec)
            removed_id = None
        else:
            # remove events from the removed source's track
            # find the track corresponding to the removed source
            # events reference tracks via track_id; sources map to tracks by order
            track_ids = [t.id for t in ledger.tracks]
            if remove_idx < len(track_ids):
                removed_track = track_ids[remove_idx]
                # keep events NOT from this track
                ledger = ledger.model_copy(update={
                    "events": [e for e in ledger.events if e.track_id != removed_track],
                    "tracks": [t for t in ledger.tracks if t.id != removed_track],
                })

        return mixture, ledger, removed_id
    else:
        # normal: load original mixture
        mixture = _load_audio(str(Path(audio_base) / entry.mixture_path), sr, max_sec)
        return mixture, ledger, None


def main():
    parser = argparse.ArgumentParser(prog="train-carc")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = _load_config(args.config)
    tcfg = cfg["train"]
    steps = tcfg["steps"]
    device = cfg["model"]["device"]

    split_contract = cfg.get("data", {}).get("split_contract_path")
    data_gate_summary = cfg.get("data", {}).get("data_gate_summary_path")
    expected_split = cfg.get("data", {}).get("expected_split")
    pre_split = cfg.get("data", {}).get("pre_split", False)
    if pre_split and (not split_contract or not data_gate_summary):
        raise ValueError(
            "data.pre_split=true requires data.split_contract_path and "
            "data.data_gate_summary_path"
        )
    if split_contract:
        if expected_split != "train":
            raise ValueError("training requires data.expected_split=train")
        from sceneledger.data.experiment_data import (
            require_experiment_data_summary,
            require_split_manifest,
        )

        require_experiment_data_summary(data_gate_summary, split_contract)
        require_split_manifest(
            split_contract, expected_split, cfg["data"]["manifest_path"]
        )

    import sys as _sys
    repo = Path(__file__).resolve().parents[1] / "third_party" / "MOSS-Audio"
    if str(repo) not in _sys.path:
        _sys.path.insert(0, str(repo))
    from src.modeling_moss_audio import MossAudioModel
    from src.processing_moss_audio import MossAudioProcessor

    print("[carc] loading model ...", file=sys.stderr, flush=True)
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    dtype_str = cfg["model"].get("dtype", "bfloat16")
    model = MossAudioModel.from_pretrained(
        cfg["model"]["path"], trust_remote_code=True, dtype=dtype_str, device_map=device)
    model.eval()
    processor = MossAudioProcessor.from_pretrained(
        cfg["model"]["path"], trust_remote_code=True, enable_time_marker=True)

    if tcfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    from peft import LoraConfig, get_peft_model
    lora_cfg = cfg["lora"]
    peft_config = LoraConfig(
        r=lora_cfg["rank"], lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg.get("dropout", 0.05),
        target_modules=lora_cfg["target_modules"], bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, peft_config)
    model.train()
    model.print_trainable_parameters()

    from sceneledger.data.datamodule import MOSS_INPUT_SAMPLE_RATE, group_split
    from sceneledger.data.manifests import read_manifest
    from sceneledger.models.target_formatter import (
        StyleConfig,
        canonical_prompt,
        format_slot_aware_caption,
    )

    entries = read_manifest(cfg["data"]["manifest_path"])
    if pre_split:
        train_entries = entries
    else:
        train_entries, _ = group_split(
            entries, val_fraction=cfg["data"].get("val_fraction", 0.1),
            group_key=cfg["data"].get("group_key", "source_id"), seed=tcfg["seed"])
    print(f"[carc] {len(train_entries)} training samples", file=sys.stderr, flush=True)

    carc_prob = tcfg.get("carc_probability", 0.3)
    print(f"[carc] CARC probability: {carc_prob}", file=sys.stderr, flush=True)

    timestamp_weight = cfg["loss"].get("timestamp_weight", 5.0)
    ts_token_ids = None
    if timestamp_weight != 1.0:
        from sceneledger.losses.weighted_ce import compute_timestamp_token_ids
        ts_token_ids = compute_timestamp_token_ids(processor.tokenizer)

    from torch.optim import AdamW
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=tcfg["learning_rate"], weight_decay=tcfg.get("weight_decay", 0.0))

    grad_accum = tcfg["global_effective_batch"] // tcfg["micro_batch_size"]
    style_cfg = StyleConfig()
    prompt_text = canonical_prompt(
        style=cfg["data"].get("style", "brief"),
        include_lyrics=cfg["data"].get("include_lyrics", False))
    audio_base = cfg["data"]["audio_base_dir"]
    sr = cfg["data"].get("sample_rate", MOSS_INPUT_SAMPLE_RATE)
    max_sec = cfg["data"].get("max_audio_seconds", 30.0)
    slot_aware = cfg["data"].get("slot_aware", False)
    shuffle_events = tcfg.get("shuffle_events", False)

    rng = random.Random(tcfg["seed"])
    indices = list(range(len(train_entries)))
    step = 0
    epoch = 0
    t0 = time.time()
    accum_loss = 0.0
    n_carc = 0

    while step < steps:
        rng.shuffle(indices)
        epoch += 1
        for idx in indices:
            if step >= steps:
                break
            entry = train_entries[idx]
            # CARC: load stems, maybe remove one
            mixture_audio, ledger, removed_id = _load_stems_and_maybe_remove(
                entry, audio_base, sr, max_sec, rng, carc_prob)
            if removed_id is not None:
                n_carc += 1

            # shuffle events if configured
            if shuffle_events:
                events = list(ledger.events)
                rng.shuffle(events)
                ledger = ledger.model_copy(update={"events": events})

            # format target
            if slot_aware:
                target = format_slot_aware_caption(ledger, style=cfg["data"].get("style", "brief"), cfg=style_cfg)
            else:
                from sceneledger.models.target_formatter import format_atomic_caption
                target = format_atomic_caption(ledger, style=cfg["data"].get("style", "brief"), cfg=style_cfg)

            # process audio + build training sample
            inputs = processor(text=prompt_text, audios=[mixture_audio], return_tensors="pt")
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            audio_data = inputs.get("audio_data")
            audio_seqlens = inputs.get("audio_data_seqlens")

            target_ids = processor.tokenizer.encode(target, add_special_tokens=False)
            eos_id = processor.tokenizer.eos_token_id
            target_ids = target_ids + [eos_id]
            prompt_len = input_ids.shape[1]
            target_tensor = torch.tensor([target_ids], dtype=torch.long)
            full_input_ids = torch.cat([input_ids, target_tensor], dim=1).to(device)
            full_attention_mask = torch.cat([attention_mask, torch.ones_like(target_tensor)], dim=1).to(device)
            labels = torch.full_like(full_input_ids, -100)
            labels[:, prompt_len:] = target_tensor.to(device)
            audio_input_mask = (full_input_ids == processor.audio_token_id).to(device)

            # forward
            if timestamp_weight == 1.0:
                outputs = model(
                    input_ids=full_input_ids, attention_mask=full_attention_mask,
                    audio_data=audio_data.to(device).to(dtype_map.get(dtype_str, torch.bfloat16)) if audio_data is not None else None,
                    audio_data_seqlens=audio_seqlens.to(device) if audio_seqlens is not None else None,
                    audio_input_mask=audio_input_mask, labels=labels)
                loss = outputs.loss
            else:
                from sceneledger.losses.weighted_ce import time_weighted_ce_loss
                outputs = model(
                    input_ids=full_input_ids, attention_mask=full_attention_mask,
                    audio_data=audio_data.to(device).to(dtype_map.get(dtype_str, torch.bfloat16)) if audio_data is not None else None,
                    audio_data_seqlens=audio_seqlens.to(device) if audio_seqlens is not None else None,
                    audio_input_mask=audio_input_mask, labels=None)
                loss = time_weighted_ce_loss(outputs.logits, labels, ts_token_ids, timestamp_weight)

            loss = loss / grad_accum
            loss.backward()
            accum_loss += loss.item()

            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], tcfg.get("max_grad_norm", 1.0))
                optimizer.step()
                optimizer.zero_grad()

            step += 1
            if step % 100 == 0:
                elapsed = time.time() - t0
                print(f"[carc] step {step}/{steps} ep={epoch} loss={accum_loss/max(1,grad_accum):.4f} "
                      f"carc={n_carc}/{step} ({elapsed:.0f}s, {step/elapsed:.1f} step/s)",
                      file=sys.stderr, flush=True)
                accum_loss = 0.0

    # save
    out_dir = Path(tcfg.get("output_dir", "outputs/b3_carc"))
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir / "lora")
    processor.tokenizer.save_pretrained(out_dir / "lora")
    print(f"[carc] saved to {out_dir / 'lora'} (carc samples: {n_carc}/{step})", file=sys.stderr, flush=True)
    print(f"[carc] done in {time.time()-t0:.0f}s", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()

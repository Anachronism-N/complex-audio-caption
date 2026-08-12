"""P9 DPO: direct preference optimization on B3-slot-aware-5k.

1. Load B3-slot-aware-5k predictions (with halluc/omit errors)
2. For each clip with errors: chosen=ground truth, rejected=model prediction
3. Train DPO: maximize log σ(β * (logπ(chosen) - logπ_ref(chosen)
                                  - logπ(rejected) + logπ_ref(rejected)))
4. Reference model = base MOSS (LoRA disabled)
"""

from __future__ import annotations
import json, sys, time, random
from pathlib import Path
import torch, torch.nn.functional as F
import yaml

def main():
    cfg = yaml.safe_load(Path("configs/model/b3_slot_aware_5k.yaml").read_text())
    device = "cuda:0"
    beta = 0.1  # DPO temperature
    steps = 500
    lr = 5e-6  # lower than SFT

    import sys as _sys
    repo = Path(__file__).resolve().parents[1] / "third_party" / "MOSS-Audio"
    if str(repo) not in _sys.path:
        _sys.path.insert(0, str(repo))
    _sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

    from src.modeling_moss_audio import MossAudioModel
    from src.processing_moss_audio import MossAudioProcessor
    from peft import LoraConfig, get_peft_model, PeftModel
    from sceneledger.data.manifests import read_manifest
    from sceneledger.data.schema import Ledger
    from sceneledger.models.target_formatter import canonical_prompt, format_slot_aware_caption, StyleConfig
    from sceneledger.eval.parser import parse_model_output

    print("[dpo] loading model ...", file=sys.stderr, flush=True)
    model = MossAudioModel.from_pretrained(
        "/tmp/moss_weights", trust_remote_code=True, dtype="bfloat16", device_map=device)
    model.eval()
    processor = MossAudioProcessor.from_pretrained(
        "/tmp/moss_weights", trust_remote_code=True, enable_time_marker=True)

    # load SFT LoRA as starting point
    sft_lora = "outputs/b3_slot_aware_5k/lora"
    model = PeftModel.from_pretrained(model, sft_lora)
    model.train()
    model.print_trainable_parameters()

    # load predictions + references to build preference pairs
    print("[dpo] building preference pairs ...", file=sys.stderr, flush=True)
    entries = read_manifest("data/derived/b3_5k/manifest.jsonl")
    preds = {}
    with open("reports/b3_slot_aware_5k_predictions.jsonl") as f:
        for line in f:
            d = json.loads(line)
            preds[d["sample_id"]] = d

    style_cfg = StyleConfig()
    prompt_text = canonical_prompt(style="brief", include_lyrics=True)
    audio_base = "/tmp/b3_5k"
    sr = processor.config.mel_sr
    max_sec = 30.0

    pairs = []
    for entry in entries:
        sid = entry.scene["scene_id"]
        if sid not in preds:
            continue
        gt_ledger = Ledger.model_validate(entry.target_ledger)
        pred_ledger = Ledger.model_validate(preds[sid])
        # only use clips where prediction differs from ground truth
        gt_events = set((e.type, round(e.start_sec(), 1), round(e.end_sec(), 1)) for e in gt_ledger.events)
        pred_events = set((e.type, round(e.start_sec(), 1), round(e.end_sec(), 1)) for e in pred_ledger.events)
        if gt_events == pred_events:
            continue  # perfect prediction, skip

        chosen = format_slot_aware_caption(gt_ledger, style="brief", cfg=style_cfg)
        # use model's raw prediction as rejected (re-format from prediction ledger)
        rejected = format_slot_aware_caption(pred_ledger, style="brief", cfg=style_cfg)
        if chosen == rejected:
            continue

        pairs.append({
            "sid": sid,
            "audio_path": str(Path(audio_base) / entry.mixture_path),
            "chosen": chosen,
            "rejected": rejected,
        })

    print(f"[dpo] {len(pairs)} preference pairs (from {len(preds)} predictions)", file=sys.stderr, flush=True)
    if not pairs:
        print("[dpo] no pairs found, exiting", file=sys.stderr)
        return

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr)

    rng = random.Random(42)
    step = 0
    t0 = time.time()

    def _load_audio(path):
        import soundfile as sf
        from scipy.signal import resample_poly
        from math import gcd
        wav, sr_orig = sf.read(path, dtype="float32", always_2d=False)
        if wav.ndim == 2: wav = wav.mean(axis=1)
        if sr_orig != sr:
            g = gcd(int(sr_orig), int(sr))
            wav = resample_poly(wav.astype(np.float64), int(sr)//g, int(sr_orig)//g).astype(np.float32)
        return wav[:int(max_sec * sr)]

    def _compute_logp(text_target, audio_wav):
        """Compute log p(target | audio, prompt) for a single sample."""
        inputs = processor(text=prompt_text, audios=[audio_wav], return_tensors="pt")
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        audio_data = inputs.get("audio_data")
        audio_seqlens = inputs.get("audio_data_seqlens")

        target_ids = processor.tokenizer.encode(text_target, add_special_tokens=False)
        eos_id = processor.tokenizer.eos_token_id
        target_ids = target_ids + [eos_id]
        prompt_len = input_ids.shape[1]
        target_tensor = torch.tensor([target_ids], dtype=torch.long)
        full_ids = torch.cat([input_ids, target_tensor], dim=1).to(device)
        full_mask = torch.cat([attention_mask, torch.ones_like(target_tensor)], dim=1).to(device)
        labels = torch.full_like(full_ids, -100)
        labels[:, prompt_len:] = target_tensor.to(device)
        audio_input_mask = (full_ids == processor.audio_token_id).to(device)

        outputs = model(
            input_ids=full_ids, attention_mask=full_mask,
            audio_data=audio_data.to(device).to(torch.bfloat16) if audio_data is not None else None,
            audio_data_seqlens=audio_seqlens.to(device) if audio_seqlens is not None else None,
            audio_input_mask=audio_input_mask, labels=None)

        # compute log p of target tokens
        logits = outputs.logits[:, :-1, :]  # [1, T-1, V]
        labels_shifted = labels[:, 1:]  # [1, T-1]
        mask = labels_shifted != -100  # [1, T-1]
        log_probs = F.log_softmax(logits, dim=-1)
        token_logps = log_probs.gather(-1, labels_shifted.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        token_logps = token_logps * mask
        return token_logps.sum()  # total log p

    while step < steps:
        rng.shuffle(pairs)
        for pair in pairs:
            if step >= steps:
                break
            wav = _load_audio(pair["audio_path"])

            # log p with LoRA (policy)
            model.enable_adapter_layers()
            logp_chosen = _compute_logp(pair["chosen"], wav)
            logp_rejected = _compute_logp(pair["rejected"], wav)

            # log p without LoRA (reference)
            model.disable_adapter_layers()
            with torch.no_grad():
                logp_chosen_ref = _compute_logp(pair["chosen"], wav)
                logp_rejected_ref = _compute_logp(pair["rejected"], wav)
            model.enable_adapter_layers()

            # DPO loss
            chosen_logratio = logp_chosen - logp_chosen_ref
            rejected_logratio = logp_rejected - logp_rejected_ref
            loss = -F.logsigmoid(beta * (chosen_logratio - rejected_logratio)).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()

            if (step + 1) % 50 == 0:
                print(f"[dpo] step {step+1}/{steps} loss={loss.item():.4f} "
                      f"chosen_logp={logp_chosen.item():.1f} "
                      f"rejected_logp={logp_rejected.item():.1f} "
                      f"({time.time()-t0:.0f}s)", file=sys.stderr, flush=True)
            step += 1

    # save
    out_dir = Path("outputs/b3_dpo")
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir / "lora")
    processor.tokenizer.save_pretrained(out_dir / "lora")
    print(f"[dpo] saved to {out_dir / 'lora'} ({time.time()-t0:.0f}s)", file=sys.stderr, flush=True)

if __name__ == "__main__":
    import numpy as np
    main()

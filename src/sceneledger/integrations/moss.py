from __future__ import annotations

import json
import sys
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from ..serialization import serialize_tagged_caption
from ..types import Ledger


def write_moss_sft(
    ledgers: Iterable[Ledger],
    audio_by_sample_id: dict[str, str | Path],
    output_path: str | Path,
    *,
    prompt: str = (
        "Describe all audible speech, lyrics, music, and sound events. "
        "Use typed tags and 0.1-second timestamps; do not guess inaudible content."
    ),
    absolute_audio_paths: bool = True,
) -> int:
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for ledger in ledgers:
            if ledger.sample_id not in audio_by_sample_id:
                raise KeyError(f"No audio path for sample {ledger.sample_id}")
            audio_path = Path(audio_by_sample_id[ledger.sample_id])
            if absolute_audio_paths:
                audio_path = audio_path.resolve()
            row = {
                "conversation": [
                    {"role": "user", "message_type": "audio", "content": str(audio_path)},
                    {"role": "user", "message_type": "text", "content": prompt},
                    {
                        "role": "assistant",
                        "message_type": "text",
                        "content": serialize_tagged_caption(ledger),
                    },
                ],
                "sample_id": ledger.sample_id,
                "schema_version": ledger.schema_version,
            }
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


class MossInferenceAdapter:
    """Thin adapter around the official OpenMOSS/MOSS-Audio repository.

    The upstream checkout is injected at runtime, avoiding a vendored or stale copy of model code.
    """

    def __init__(
        self,
        upstream_root: str | Path,
        model_path: str | Path,
        *,
        device: str | None = None,
        enable_time_marker: bool = True,
    ) -> None:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("Install the official MOSS-Audio torch runtime first") from exc

        root = Path(upstream_root).resolve()
        if not (root / "src" / "modeling_moss_audio.py").exists():
            raise FileNotFoundError(f"Not a MOSS-Audio checkout: {root}")
        sys.path.insert(0, str(root))
        try:
            from src.audio_io import load_audio
            from src.modeling_moss_audio import MossAudioModel
            from src.processing_moss_audio import MossAudioProcessor
        except ImportError as exc:
            raise RuntimeError(f"Failed to import official MOSS-Audio modules from {root}") from exc

        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch = torch
        self.load_audio = load_audio
        self.model = MossAudioModel.from_pretrained(
            str(Path(model_path).resolve()), trust_remote_code=True, dtype="auto", device_map=device
        ).eval()
        self.processor = MossAudioProcessor.from_pretrained(
            str(Path(model_path).resolve()),
            trust_remote_code=True,
            enable_time_marker=enable_time_marker,
        )

    def generate(
        self,
        audio_path: str | Path,
        prompt: str,
        *,
        max_new_tokens: int = 1024,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> str:
        raw_audio = self.load_audio(str(audio_path), sample_rate=self.processor.config.mel_sr)
        inputs = self.processor(text=prompt, audios=[raw_audio], return_tensors="pt")
        inputs = inputs.to(self.model.device)
        if inputs.get("audio_data") is not None:
            inputs["audio_data"] = inputs["audio_data"].to(self.model.dtype)
        inputs["audio_input_mask"] = inputs["input_ids"] == self.processor.audio_token_id
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                num_beams=1,
                temperature=temperature,
                top_p=top_p,
                use_cache=True,
            )
        input_length = inputs["input_ids"].shape[1]
        return self.processor.decode(generated[0, input_length:], skip_special_tokens=True)

    def extract_encoder_features(self, audio_path: str | Path) -> np.ndarray:
        """Return official MOSS encoder last-layer features as [time, dimension]."""
        raw_audio = self.load_audio(str(audio_path), sample_rate=self.processor.config.mel_sr)
        inputs = self.processor(text="", audios=[raw_audio], return_tensors="pt")
        audio_data = inputs["audio_data"].to(self.model.device, dtype=self.model.dtype)
        sequence_lengths = inputs["audio_data_seqlens"].to(self.model.device)
        with self.torch.inference_mode():
            features, _ = self.model.get_audio_features(audio_data, sequence_lengths)
        return features[0].detach().float().cpu().numpy()

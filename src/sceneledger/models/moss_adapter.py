"""MOSS-Audio adapter: wrap inference so B0/B1/B2 share one interface.

The real adapter mirrors ``third_party/MOSS-Audio/infer.py``:

    model = MossAudioModel.from_pretrained(path, trust_remote_code=True, ...)
    processor = MossAudioProcessor.from_pretrained(path, trust_remote_code=True,
                                                  enable_time_marker=True)
    raw_audio = load_audio(path, sample_rate=processor.config.mel_sr)  # mel_sr=16000
    inputs = processor(text=prompt, audios=[raw_audio], return_tensors="pt")
    inputs["audio_input_mask"] = inputs["input_ids"] == processor.audio_token_id
    generated = model.generate(**inputs, max_new_tokens=..., do_sample=..., ...)
    text = processor.decode(generated[0, input_len:], skip_special_tokens=True)

Because the real model needs ``transformers==4.57.1`` / ``numpy>=2.0`` /
``torch==2.9.1`` (incompatible with this env), :class:`MockMossAdapter`
produces deterministic, *imperfect* predictions from the target ledger so the
full B0 pipeline (infer -> parse -> evaluate) can run end-to-end now. Swap to
:class:`MossAdapter` once a dedicated env is authorized.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from sceneledger.data.schema import Event, Ledger, Span
from sceneledger.models.target_formatter import (
    format_atomic_caption,
    canonical_prompt,
)


class AudioCaptioner(Protocol):
    """Shared inference interface for B0/B1/B2."""

    def infer(self, audio_path: str, prompt: str, *, sample_id: str, duration: float) -> str:  # pragma: no cover
        ...


# --------------------------------------------------------------------------- #
# real adapter (deferred — needs moss-audio env)
# --------------------------------------------------------------------------- #
@dataclass
class MossAdapterConfig:
    model_path: str = "OpenMOSS-Team/MOSS-Audio-4B-Instruct"
    device: str = "cuda:0"
    dtype: str = "auto"
    max_new_tokens: int = 1024
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 50
    enable_time_marker: bool = True


class MossAdapter:
    """Real MOSS-Audio-4B adapter. Lazily imports the moss-audio package."""

    def __init__(self, config: MossAdapterConfig | None = None):
        self.config = config or MossAdapterConfig()
        self._model = None
        self._processor = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import sys

            repo = Path(__file__).resolve().parents[3] / "third_party" / "MOSS-Audio"
            if str(repo) not in sys.path:
                sys.path.insert(0, str(repo))
            from src.modeling_moss_audio import MossAudioModel  # type: ignore
            from src.processing_moss_audio import MossAudioProcessor  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "MOSS-Audio package not available. Install its torch-runtime env "
                "(transformers==4.57.1, numpy>=2.0, torch==2.9.1) from "
                "third_party/MOSS-Audio, or use MockMossAdapter. Original error: "
                f"{exc}"
            ) from exc
        import torch

        self._model = MossAudioModel.from_pretrained(
            self.config.model_path,
            trust_remote_code=True,
            dtype=self.config.dtype,
            device_map=self.config.device,
        )
        self._model.eval()
        self._processor = MossAudioProcessor.from_pretrained(
            self.config.model_path,
            trust_remote_code=True,
            enable_time_marker=self.config.enable_time_marker,
        )
        self._torch = torch

    @staticmethod
    def _load_audio_native(path: str, sample_rate: int) -> "np.ndarray":
        """Load audio as 1D float32 numpy array at ``sample_rate`` (mono).

        Uses soundfile + scipy resample to avoid torchaudio's torchcodec
        dependency (torchaudio 2.9.x) and librosa.
        """
        import soundfile as sf
        from scipy.signal import resample_poly

        wav, sr = sf.read(path, dtype="float32", always_2d=False)
        if wav.ndim == 2:
            wav = wav.mean(axis=1)
        if sr != sample_rate:
            # resample_poly needs integer up/down; reduce by GCD
            from math import gcd

            g = gcd(int(sr), int(sample_rate))
            up = int(sample_rate) // g
            down = int(sr) // g
            wav = resample_poly(wav.astype(np.float64), up, down).astype(np.float32)
        return wav

    def infer(self, audio_path: str, prompt: str, *, sample_id: str, duration: float) -> str:
        self._load()
        import torch

        sr = self._processor.config.mel_sr
        raw_audio = self._load_audio_native(audio_path, sample_rate=sr)
        inputs = self._processor(text=prompt, audios=[raw_audio], return_tensors="pt")
        inputs = inputs.to(self._model.device)
        if inputs.get("audio_data") is not None:
            inputs["audio_data"] = inputs["audio_data"].to(self._model.dtype)
        inputs["audio_input_mask"] = inputs["input_ids"] == self._processor.audio_token_id
        with torch.no_grad():
            gen = self._model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=True,
                num_beams=1,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                use_cache=True,
            )
        input_len = inputs["input_ids"].shape[1]
        return self._processor.decode(gen[0, input_len:], skip_special_tokens=True)


# --------------------------------------------------------------------------- #
# mock adapter — deterministic imperfect predictions
# --------------------------------------------------------------------------- #
@dataclass
class MockMossAdapterConfig:
    """Controls how degraded the mock B0 output is."""

    p_omit_event: float = 0.10        # drop an event entirely
    p_hallucinate: float = 0.08       # add a spurious event
    p_shift_boundary: float = 0.30    # nudge onset/offset by 1-3 frames
    p_wrong_pointer: float = 0.05     # reassign track pointer
    p_drop_text: float = 0.05         # replace text with a vaguer phrase
    boundary_shift_max_frames: int = 3
    seed: int = 20260808


class MockMossAdapter:
    """Produces B0-like imperfect atomic-token captions from the target ledger.

    The perturbations are deterministic per ``sample_id`` so B0 metrics are
    reproducible. This is NOT a model — it exists so the infer -> parse ->
    evaluate pipeline can be validated before the real MOSS env is ready.
    """

    def __init__(self, config: MockMossAdapterConfig | None = None):
        self.config = config or MockMossAdapterConfig()

    def _rng(self, sample_id: str) -> random.Random:
        h = int(hashlib.sha256(sample_id.encode()).hexdigest()[:8], 16)
        return random.Random(self.config.seed ^ h)

    def infer(self, audio_path: str, prompt: str, *, sample_id: str, duration: float) -> str:
        # The mock needs the target ledger; in the CLI we pass it via a side channel.
        raise NotImplementedError(
            "Use infer_from_ledger() for the mock adapter (it needs the target)."
        )

    def infer_from_ledger(self, ledger: Ledger, sample_id: str) -> str:
        rng = self._rng(sample_id)
        cfg = self.config
        events = sorted(
            ledger.events,
            key=lambda e: (round(e.start_sec(), 6), e.id),
        )
        out_events: list[Event] = []
        track_ids = [t.id for t in ledger.tracks] or ["T1"]

        for e in events:
            if rng.random() < cfg.p_omit_event:
                continue  # omission
            spans = self._perturb_spans(e.spans, rng, cfg, ledger.duration_sec)
            if not spans:
                continue
            track_id = e.track_id
            if track_id is not None and rng.random() < cfg.p_wrong_pointer:
                track_id = rng.choice([t for t in track_ids if t != track_id] or [track_id])
            text = e.text
            if rng.random() < cfg.p_drop_text:
                text = "an audible event"
            out_events.append(
                Event(
                    id=f"E{len(out_events) + 1:03d}",
                    type=e.type,
                    track_id=track_id,
                    spans=spans,
                    text=text,
                    confidence=round(max(0.3, e.confidence - rng.uniform(0.0, 0.2)), 3),
                )
            )

        # hallucination
        if rng.random() < cfg.p_hallucinate and ledger.duration_sec > 1.0:
            h_type = rng.choice(["sfx", "speech", "music"])
            h_start = round(rng.uniform(0.1, ledger.duration_sec - 0.5), 1)
            h_end = round(min(ledger.duration_sec, h_start + rng.uniform(0.2, 1.0)), 1)
            out_events.append(
                Event(
                    id=f"E{len(out_events) + 1:03d}",
                    type=h_type,  # type: ignore[arg-type]
                    track_id=rng.choice(track_ids),
                    spans=[Span(start_sec=h_start, end_sec=h_end)],
                    text="possibly an additional sound",
                    confidence=round(rng.uniform(0.3, 0.6), 3),
                )
            )

        mock_ledger = Ledger(
            sample_id=sample_id,
            duration_sec=ledger.duration_sec,
            tracks=ledger.tracks,
            events=out_events,
        )
        return format_atomic_caption(mock_ledger, style="brief")

    @staticmethod
    def _perturb_spans(
        spans: list[Span], rng: random.Random, cfg: MockMossAdapterConfig, duration: float
    ) -> list[Span]:
        out: list[Span] = []
        for sp in spans:
            if rng.random() < cfg.p_shift_boundary:
                ds = rng.randint(-cfg.boundary_shift_max_frames, cfg.boundary_shift_max_frames) * 0.1
                de = rng.randint(-cfg.boundary_shift_max_frames, cfg.boundary_shift_max_frames) * 0.1
            else:
                ds = de = 0.0
            s = max(0.0, round(sp.start_sec + ds, 1))
            e = min(duration, round(sp.end_sec + de, 1))
            if e > s:
                out.append(Span(start_sec=s, end_sec=e))
        # re-sort and drop overlaps introduced by shift
        out.sort(key=lambda x: x.start_sec)
        merged: list[Span] = []
        for sp in out:
            if merged and sp.start_sec < merged[-1].end_sec - 1e-6:
                merged[-1] = Span(start_sec=merged[-1].start_sec, end_sec=max(merged[-1].end_sec, sp.end_sec))
            else:
                merged.append(sp)
        return merged


__all__ = [
    "AudioCaptioner",
    "MockMossAdapter",
    "MockMossAdapterConfig",
    "MossAdapter",
    "MossAdapterConfig",
]

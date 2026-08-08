from __future__ import annotations

from pathlib import Path

import numpy as np


class SamAudioAdapter:
    """Optional text/span-prompted separator using the official SAM-Audio package."""

    def __init__(
        self,
        model_id: str = "facebook/sam-audio-large",
        *,
        device: str = "cuda",
    ) -> None:
        try:
            import torch
            from sam_audio import SAMAudio, SAMAudioProcessor
        except ImportError as exc:
            raise RuntimeError(
                "Clone facebookresearch/sam-audio, request checkpoint access, and install it first"
            ) from exc
        self.torch = torch
        self.device = device
        self.model = SAMAudio.from_pretrained(model_id).eval().to(device)
        self.processor = SAMAudioProcessor.from_pretrained(model_id)

    def separate(
        self,
        audio: str | Path,
        description: str,
        *,
        anchors: list[list[tuple[str, float, float]]] | None = None,
        predict_spans: bool = True,
        reranking_candidates: int = 1,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        kwargs = {
            "audios": [str(audio)],
            "descriptions": [description.lower().strip()],
        }
        if anchors is not None:
            kwargs["anchors"] = anchors
        batch = self.processor(**kwargs).to(self.device)
        with self.torch.inference_mode():
            result = self.model.separate(
                batch,
                predict_spans=predict_spans,
                reranking_candidates=reranking_candidates,
            )
        target = result.target.detach().cpu().float().numpy().squeeze()
        residual = result.residual.detach().cpu().float().numpy().squeeze()
        return target, residual, int(self.processor.audio_sampling_rate)

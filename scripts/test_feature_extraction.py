"""Manual MOSS audio-feature extraction smoke test for the S1 slot decoder."""

from __future__ import annotations

import argparse
from math import gcd

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

from sceneledger.models.moss_adapter import MossAdapter, MossAdapterConfig


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()

    adapter = MossAdapter(
        MossAdapterConfig(
            model_path=args.model_path,
            device=args.device,
            dtype=args.dtype,
        )
    )
    adapter._load()
    model = adapter._model
    processor = adapter._processor

    wav, sample_rate = sf.read(args.audio, dtype="float32")
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    target_rate = int(processor.config.mel_sr)
    if sample_rate != target_rate:
        factor = gcd(int(sample_rate), target_rate)
        wav = resample_poly(
            wav.astype(np.float64),
            target_rate // factor,
            int(sample_rate) // factor,
        ).astype(np.float32)

    inputs = processor(text="test", audios=[wav], return_tensors="pt")
    input_dtype = getattr(torch, args.dtype)
    audio_data = inputs["audio_data"].to(args.device).to(input_dtype)
    audio_lengths = inputs["audio_data_seqlens"].to(args.device)

    with torch.inference_mode():
        audio_embeddings, deepstack = model.get_audio_features(
            audio_data, audio_lengths
        )
        audio_embeddings = model.audio_adapter(audio_embeddings)

    duration = len(wav) / target_rate
    print(
        f"audio_embeddings: shape={audio_embeddings.shape}, "
        f"dtype={audio_embeddings.dtype}"
    )
    print(
        f"deepstack: {len(deepstack)} layers, "
        f"shapes={[layer.shape for layer in deepstack[:3]]}"
    )
    print(
        f"frames={audio_embeddings.shape[1]}, "
        f"rate={audio_embeddings.shape[1] / duration:.1f} Hz"
    )
    print("feature extraction OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

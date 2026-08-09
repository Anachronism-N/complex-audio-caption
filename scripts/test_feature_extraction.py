"""Test MOSS audio feature extraction for S1 slot decoder."""
import sys, torch, numpy as np
sys.path.insert(0, "third_party/MOSS-Audio")
from sceneledger.models.moss_adapter import MossAdapter, MossAdapterConfig
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd

adapter = MossAdapter(MossAdapterConfig(model_path="/tmp/moss_weights", device="cuda:0", dtype="bfloat16"))
adapter._load()
model = adapter._model
processor = adapter._processor

wav, sr = sf.read("/tmp/b3_unified/audio/mix_000001.wav", dtype="float32")
if wav.ndim == 2:
    wav = wav.mean(axis=1)
if sr != 16000:
    g = gcd(int(sr), 16000)
    wav = resample_poly(wav.astype(np.float64), 16000 // g, int(sr) // g).astype(np.float32)

inputs = processor(text="test", audios=[wav], return_tensors="pt")
audio_data = inputs["audio_data"].to("cuda:0").to(torch.bfloat16)
audio_seqlens = inputs["audio_data_seqlens"].to("cuda:0")

with torch.no_grad():
    audio_embeds, deepstack = model.get_audio_features(audio_data, audio_seqlens)
    audio_embeds = model.audio_adapter(audio_embeds)

print(f"audio_embeds: shape={audio_embeds.shape}, dtype={audio_embeds.dtype}")
print(f"deepstack: {len(deepstack)} layers, shapes={[d.shape for d in deepstack[:3]]}")
print(f"frames={audio_embeds.shape[1]}, rate={audio_embeds.shape[1]/(len(wav)/16000):.1f} Hz")
print("feature extraction OK")

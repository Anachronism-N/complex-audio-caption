"""Integration tests for datamodule + mock adapter + B0 infer pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest

from sceneledger.data.datamodule import (
    MOSS_INPUT_SAMPLE_RATE,
    DatamoduleConfig,
    build_dataset,
    group_split,
)
from sceneledger.data.manifests import ManifestEntry, read_manifest, write_manifest
from sceneledger.data.schema import Ledger
from sceneledger.models.moss_adapter import MockMossAdapter, MockMossAdapterConfig
from sceneledger.models.target_formatter import atomic_to_ledger

# use the real 500-clip manifest if available, else build a tiny one from the
# integration fixtures.
TAC_MINI_MANIFEST = Path("data/derived/tac_mini/manifest.jsonl")
TAC_MINI_AUDIO = Path("/tmp/tac_mini")


@pytest.fixture(scope="module")
def manifest_entries() -> list[ManifestEntry]:
    if TAC_MINI_MANIFEST.exists():
        return read_manifest(TAC_MINI_MANIFEST)
    # fall back: build 6 synthetic entries in a tmp dir
    import importlib.util

    spec = importlib.util.spec_from_file_location("trt", "tests/integration/test_round_trip.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    from sceneledger.data.manifests import persist_render
    from sceneledger.data.renderer import render_scene
    from sceneledger.data.scene_graph_sampler import (
        SceneGraphSampler,
        SceneSamplerConfig,
        SyntheticSourcePool,
    )

    pool = SyntheticSourcePool()
    sampler = SceneGraphSampler(pool=pool, config=SceneSamplerConfig())
    tmp = Path("/tmp/tac_mini_test")
    entries = []
    for i, tpl in enumerate(
        ["isolated_sfx", "speech_over_music", "music_with_sfx", "speech_music_sfx", "repeated_event"]
    ):
        s = sampler.sample(f"dm_{i:03d}", seed=200 + i, template=tpl)
        out = render_scene(s, pool)
        entries.append(persist_render(out, tmp / "audio", rel_to=tmp))
    write_manifest(tmp / "manifest.jsonl", entries)
    return entries


def test_dataset_loads_audio_and_targets(manifest_entries):
    pytest.importorskip("torch")
    if TAC_MINI_MANIFEST.exists() and not TAC_MINI_AUDIO.exists():
        pytest.skip("rendered TAC-mini audio is not present in the CPU checkout")
    if TAC_MINI_MANIFEST.exists():
        cfg = DatamoduleConfig(
            manifest_path=str(TAC_MINI_MANIFEST),
            audio_base_dir=str(TAC_MINI_AUDIO),
            sample_rate=MOSS_INPUT_SAMPLE_RATE,
            max_audio_seconds=30.0,
        )
    else:
        cfg = DatamoduleConfig(
            manifest_path="/tmp/tac_mini_test/manifest.jsonl",
            audio_base_dir="/tmp/tac_mini_test",
        )
    ds = build_dataset(cfg)
    assert len(ds) >= 5
    item = ds[0]
    assert item["audio"].ndim == 1
    assert item["audio"].shape[0] == int(30.0 * cfg.sample_rate)  # padded/truncated
    assert item["target_atomic"].startswith(("<music", "<speech", "<sfx", "<lys", "<empty/>"))
    assert isinstance(item["ledger"], Ledger)
    Ledger.model_validate(item["ledger"].model_dump())


def test_group_split_is_leak_free(manifest_entries):
    train, val = group_split(manifest_entries, val_fraction=0.2, group_key="source_id", seed=42)
    # no group appears in both sides
    def _groups(es):
        from sceneledger.data.datamodule import _group_key

        return {_group_key(e, "source_id") for e in es}

    assert _groups(train).isdisjoint(_groups(val))
    assert len(train) + len(val) == len(manifest_entries)


def test_mock_adapter_produces_parseable_output(manifest_entries):
    adapter = MockMossAdapter(MockMossAdapterConfig(seed=42))
    entry = manifest_entries[0]
    target = Ledger.model_validate(entry.target_ledger)
    raw = adapter.infer_from_ledger(target, entry.scene["scene_id"])
    # must parse as atomic tokens
    parsed = atomic_to_ledger(raw, entry.scene["scene_id"], float(entry.scene["duration"]))
    Ledger.model_validate(parsed.model_dump())
    # mock should usually produce a similar (but imperfect) event count
    assert abs(len(parsed.events) - len(target.events)) <= 2


def test_mock_adapter_deterministic(manifest_entries):
    adapter = MockMossAdapter(MockMossAdapterConfig(seed=42))
    entry = manifest_entries[1]
    target = Ledger.model_validate(entry.target_ledger)
    raw1 = adapter.infer_from_ledger(target, entry.scene["scene_id"])
    raw2 = adapter.infer_from_ledger(target, entry.scene["scene_id"])
    assert raw1 == raw2  # same sample_id -> same perturbations


def test_mock_adapter_introduces_realistic_degradation(manifest_entries):
    # over enough samples, the mock should produce at least some omissions / hallucinations
    adapter = MockMossAdapter(MockMossAdapterConfig(seed=42))
    n_omit = 0
    n_halluc = 0
    for entry in manifest_entries[:60]:
        target = Ledger.model_validate(entry.target_ledger)
        raw = adapter.infer_from_ledger(target, entry.scene["scene_id"])
        parsed = atomic_to_ledger(raw, entry.scene["scene_id"], float(entry.scene["duration"]))
        if len(parsed.events) < len(target.events):
            n_omit += 1
        if len(parsed.events) > len(target.events):
            n_halluc += 1
    assert n_omit > 0, "mock never produced an omission"
    assert n_halluc > 0, "mock never produced a hallucination"

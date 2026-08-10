"""Unit + integration tests for the scene renderer and manifest."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sceneledger.cli.render import render_dataset, validate_rendered_dataset
from sceneledger.data.activity import ActivityResult
from sceneledger.data.manifests import (
    persist_render,
    read_manifest,
    scene_from_dict,
    validate_manifest,
    write_manifest,
)
from sceneledger.data.renderer import (
    RESIDUAL_STEM_ID,
    RenderedSource,
    _overlap_ratio,
    render_scene,
)
from sceneledger.data.scene_graph_sampler import (
    FileSourcePool,
    PlacedSource,
    SceneGraphSampler,
    SceneSamplerConfig,
    SyntheticSourcePool,
)
from sceneledger.data.schema import Ledger


@pytest.fixture(scope="module")
def pool() -> SyntheticSourcePool:
    return SyntheticSourcePool(sample_rate=24000, seed=20260808)


@pytest.fixture(scope="module")
def sampler(pool) -> SceneGraphSampler:
    return SceneGraphSampler(pool=pool, config=SceneSamplerConfig(sample_rate=24000))


def _scene(sampler, scene_id="mix_test_001", seed=1947, template="speech_over_music"):
    return sampler.sample(scene_id=scene_id, seed=seed, template=template)


# --------------------------------------------------------------------------- #
def test_deterministic_replay_same_seed(pool, sampler):
    scene = _scene(sampler)
    out1 = render_scene(scene, pool)
    out2 = render_scene(scene, pool)
    assert out1.mixture_hash() == out2.mixture_hash()
    # stems also identical
    assert out1.stem_hashes() == out2.stem_hashes()


def test_different_seed_different_mixture(pool, sampler):
    s1 = sampler.sample("a", seed=1, template="speech_over_music")
    s2 = sampler.sample("b", seed=2, template="speech_over_music")
    o1 = render_scene(s1, pool)
    o2 = render_scene(s2, pool)
    assert o1.mixture_hash() != o2.mixture_hash()


def test_stems_sum_to_dry_mixture(pool, sampler):
    scene = _scene(sampler, template="speech_music_sfx")
    out = render_scene(scene, pool)
    dry = np.zeros_like(out.dry_mixture)
    for rs in out.stems:
        dry += rs.stem
    assert np.array_equal(dry, out.dry_mixture)


def test_semantic_stems_plus_residual_reconstruct_mixture(pool, sampler):
    scene = _scene(sampler, template="speech_music_sfx")
    scene.conditions.echo_delay_ms = 180
    scene.conditions.echo_atten_db = -8.0
    out = render_scene(scene, pool)
    reconstructed = out.dry_mixture + out.residual_stem
    assert np.allclose(reconstructed, out.mixture, atol=1e-7, rtol=0.0)
    assert RESIDUAL_STEM_ID in out.stem_hashes()


def test_source_ids_are_unique_across_sampled_scenes(sampler):
    for seed in range(500):
        scene = sampler.sample(
            f"unique_{seed}", seed=seed, template="speech_music_sfx"
        )
        source_ids = [source.source_id for source in scene.sources]
        assert len(source_ids) == len(set(source_ids))


def test_overlap_ratio_expands_coarse_activity_masks():
    def rendered(source_id: str, mask: list[int]) -> RenderedSource:
        placed = PlacedSource(
            source_id=source_id,
            kind="sfx",
            path=f"sfx:{source_id}",
            onset=0.0,
            gain_db=0.0,
            text="event",
        )
        activity = ActivityResult(
            rms_curve=np.zeros(0),
            hop_sec=0.01,
            activity_mask=np.asarray(mask, dtype=np.int8),
            resolution_sec=1.0,
            spans=[],
        )
        return RenderedSource(placed=placed, stem=np.zeros(100), activity=activity)

    sources = [
        rendered("FX01", [1] * 10),
        rendered("FX02", [0, 0, 1, 1, 0, 0, 0, 0, 0, 0]),
    ]
    assert _overlap_ratio(sources, duration=10.0) == 0.2


def test_mixture_length_matches_duration(pool, sampler):
    scene = _scene(sampler)
    out = render_scene(scene, pool)
    expected = int(round(scene.duration * scene.sample_rate))
    assert out.mixture.shape[0] == expected


def test_target_ledger_is_schema_valid(pool, sampler):
    for tpl in (
        "isolated_sfx",
        "speech_over_music",
        "music_with_sfx",
        "speech_music_sfx",
        "repeated_event",
        "ambient_with_intermittent_sfx",
    ):
        scene = sampler.sample(f"t_{tpl}", seed=42, template=tpl)
        out = render_scene(scene, pool)
        # target_ledger is a Ledger object; re-validate strictly
        Ledger.model_validate(out.target_ledger.model_dump())
        assert out.target_ledger.duration_sec == scene.duration
        # every event track_id resolves
        track_ids = {t.id for t in out.target_ledger.tracks}
        for e in out.target_ledger.events:
            assert e.track_id in track_ids
            assert e.spans  # non-empty


def test_repeated_event_produces_multispan_sfx(pool, sampler):
    # force repeat by sampling the repeated_event template with a seed that yields repeat>1
    found_multispan = False
    for seed in range(1, 60):
        scene = sampler.sample(f"rep_{seed}", seed=seed, template="repeated_event")
        out = render_scene(scene, pool)
        sfx_events = [e for e in out.target_ledger.events if e.type == "sfx"]
        if sfx_events and any(len(e.spans) > 1 for e in sfx_events):
            found_multispan = True
            break
    assert found_multispan, "no repeated_event produced a multi-span sfx event"


def test_events_ordered_by_onset_then_type(pool, sampler):
    scene = _scene(sampler, template="speech_music_sfx")
    out = render_scene(scene, pool)
    keys = [(round(e.start_sec(), 6), e.type) for e in out.target_ledger.events]
    onsets = [k[0] for k in keys]
    assert onsets == sorted(onsets)


def test_scene_dict_round_trip(pool, sampler):
    scene = _scene(sampler)
    d = scene.to_manifest_dict()
    scene2 = scene_from_dict(d)
    assert scene2.to_manifest_dict() == d


def test_persist_and_manifest_round_trip(tmp_path: Path, pool, sampler):
    scene = _scene(sampler, template="speech_music_sfx")
    out = render_scene(scene, pool)
    entry = persist_render(out, tmp_path / "audio", rel_to=tmp_path)
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(manifest, [entry])
    entries = read_manifest(manifest)
    assert len(entries) == 1
    e = entries[0]
    assert e.mixture_hash == out.mixture_hash()
    assert e.target_ledger["sample_id"] == scene.scene_id
    # audio files exist
    assert (tmp_path / e.mixture_path).exists()
    for sp in e.stem_paths.values():
        assert (tmp_path / sp).exists()


def test_validate_manifest_passes(tmp_path: Path, pool, sampler):
    # render a small batch and validate
    entries = []
    for i, tpl in enumerate(
        ["isolated_sfx", "speech_over_music", "music_with_sfx", "speech_music_sfx", "repeated_event"]
    ):
        scene = sampler.sample(f"v_{i:03d}", seed=100 + i, template=tpl)
        out = render_scene(scene, pool)
        entries.append(persist_render(out, tmp_path / "audio", rel_to=tmp_path))
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(manifest, entries)
    rep = validate_manifest(manifest, pool, check_audio=True)
    assert rep.ok(), rep.failures[:5]
    assert rep.n_replay_ok == 5
    assert rep.n_stems_sum_ok == 5
    assert rep.n_ledger_valid == 5
    assert rep.n_saved_reconstruction_ok == 5
    assert rep.n_audio_files_fail == 0


def test_validate_manifest_detects_tampered_hash(tmp_path: Path, pool, sampler):
    scene = sampler.sample("tamper", seed=7, template="speech_over_music")
    out = render_scene(scene, pool)
    entry = persist_render(out, tmp_path / "audio", rel_to=tmp_path)
    entry.mixture_hash = "deadbeefdeadbeef"  # tamper
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(manifest, [entry])
    rep = validate_manifest(manifest, pool, check_audio=False)
    assert rep.n_replay_fail == 1
    assert not rep.ok()


def test_validate_manifest_detects_tampered_audio_file(tmp_path: Path, pool, sampler):
    scene = sampler.sample("tamper_audio", seed=17, template="speech_over_music")
    out = render_scene(scene, pool)
    entry = persist_render(out, tmp_path / "audio", rel_to=tmp_path)
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(manifest, [entry])
    mixture_path = tmp_path / entry.mixture_path
    content = bytearray(mixture_path.read_bytes())
    content[-1] ^= 1
    mixture_path.write_bytes(content)
    rep = validate_manifest(manifest, pool, check_audio=True)
    assert rep.n_audio_files_fail >= 1
    assert not rep.ok()


def test_render_cli_persists_validation_identity(tmp_path: Path):
    config = tmp_path / "render.yaml"
    config.write_text(
        """
pool:
  kind: synthetic
  sample_rate: 24000
  seed: 20260808
sampler:
  sample_rate: 24000
  duration_range: [1.0, 1.2]
  resolutions: [0.1]
render:
  sample_count: 2
  template_weights:
    isolated_sfx: 1
  seed_base: 1947
""".strip()
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "rendered"
    assert render_dataset(str(config), str(output)) == 2
    report = validate_rendered_dataset(config_path=config, output_dir=output)
    persisted = json.loads(
        (output / "validation_report.json").read_text(encoding="utf-8")
    )
    assert report["pass"] is True
    assert persisted["manifest_sha256"] == report["manifest_sha256"]
    assert persisted["n_saved_reconstruction_ok"] == 2


def test_mixture_no_clipping(pool, sampler):
    scene = _scene(sampler, template="speech_music_sfx")
    out = render_scene(scene, pool)
    assert float(np.max(np.abs(out.mixture))) <= 0.99 + 1e-6


def test_file_backed_vocal_uses_catalog_lyrics_not_invented_text(tmp_path: Path):
    import soundfile as sf

    sr = 24000
    seconds = 10
    time = np.arange(sr * seconds) / sr
    vocal_path = (tmp_path / "vocal.wav").resolve()
    music_path = (tmp_path / "music.wav").resolve()
    sf.write(vocal_path, (0.2 * np.sin(2 * np.pi * 220 * time)).astype(np.float32), sr)
    sf.write(music_path, (0.1 * np.sin(2 * np.pi * 110 * time)).astype(np.float32), sr)
    file_pool = FileSourcePool(
        by_kind={"vocal": [str(vocal_path)], "music": [str(music_path)]},
        metadata_by_path={
            str(vocal_path): {
                "text": "take me home",
                "language": "en",
                "verbatim": True,
                "source_group": "song-1",
                "dataset": "fixture",
                "license": "test-only",
            },
            str(music_path): {
                "text": "instrumental accompaniment",
                "source_group": "song-2",
                "dataset": "fixture",
                "license": "test-only",
            },
        },
        strict_metadata=True,
    )
    file_sampler = SceneGraphSampler(
        pool=file_pool, config=SceneSamplerConfig(sample_rate=sr)
    )
    output = render_scene(
        file_sampler.sample("real_lyrics", seed=17, template="lyrics_over_music"),
        file_pool,
    )
    lyrics = [event for event in output.target_ledger.events if event.type == "lys"]
    assert len(lyrics) == 1
    assert lyrics[0].text == "take me home"
    assert lyrics[0].verbatim is True
    assert output.target_ledger.provenance.source_dataset == "fixture"

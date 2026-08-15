"""Unit + integration tests for the scene renderer and manifest."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sceneledger.data.manifests import (
    persist_render,
    read_manifest,
    scene_from_dict,
    validate_manifest,
    write_manifest,
)
from sceneledger.data.renderer import render_scene
from sceneledger.data.scene_graph_sampler import (
    Conditions,
    PlacedSource,
    Scene,
    SceneGraphSampler,
    SceneSamplerConfig,
    Supervision,
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


def test_master_clipping_guard_scales_persisted_stems_consistently(pool, sampler):
    scene = _scene(sampler, seed=17, template="speech_music_sfx")
    for source in scene.sources:
        source.gain_db = 40.0
    out = render_scene(scene, pool)
    reconstructed = np.zeros_like(out.dry_mixture)
    for rendered_source in out.stems:
        reconstructed += rendered_source.stem
    assert np.array_equal(reconstructed, out.dry_mixture)
    assert float(np.max(np.abs(out.mixture))) <= 0.99 + 1e-6


def test_ducking_is_explicit_replayable_and_applied_to_saved_stems(pool, sampler):
    scene = _scene(sampler, seed=77, template="speech_over_music")
    scene.conditions.ducking_enabled = True
    scene.conditions.ducking_depth_db = 4.0
    ducked = render_scene(scene, pool)

    no_duck_scene = scene_from_dict(scene.to_manifest_dict())
    no_duck_scene.conditions.ducking_enabled = False
    no_duck_scene.conditions.ducking_depth_db = None
    unducked = render_scene(no_duck_scene, pool)

    ducked_music = next(rs.stem for rs in ducked.stems if rs.placed.kind == "music")
    unducked_music = next(rs.stem for rs in unducked.stems if rs.placed.kind == "music")
    assert np.any(ducked_music != unducked_music)
    assert np.mean(np.abs(ducked_music)) < np.mean(np.abs(unducked_music))

    reconstructed = np.zeros_like(ducked.dry_mixture)
    for rendered_source in ducked.stems:
        reconstructed += rendered_source.stem
    assert np.array_equal(reconstructed, ducked.dry_mixture)

    replay = render_scene(scene_from_dict(scene.to_manifest_dict()), pool)
    assert replay.mixture_hash() == ducked.mixture_hash()


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


def test_one_speech_source_produces_one_multispan_verbatim_event(pool, sampler):
    scene = sampler.sample("speech_spans", seed=42, template="speech_with_sfx")
    speech_source = next(source for source in scene.sources if source.kind == "speech")
    out = render_scene(scene, pool)
    speech_events = [event for event in out.target_ledger.events if event.type == "speech"]
    assert len(speech_events) == 1
    assert speech_events[0].text == speech_source.text
    assert speech_events[0].verbatim is False


def test_persistent_track_groups_multiple_utterance_stems(pool):
    scene = Scene(
        scene_id="persistent-speaker",
        seed=3,
        duration=6.0,
        template="overlapping_speakers",
        sources=[
            PlacedSource(
                "SP01",
                "speech",
                "speech_001",
                0.2,
                0.0,
                "first utterance",
                identity="speaker-1",
                track_group="speaker-1",
            ),
            PlacedSource(
                "SP02",
                "speech",
                "speech_002",
                3.0,
                0.0,
                "second utterance",
                identity="speaker-1",
                track_group="speaker-1",
            ),
        ],
        sample_rate=pool.sample_rate,
    )

    output = render_scene(scene, pool)

    assert len(output.stems) == 2
    assert len(output.target_ledger.tracks) == 1
    assert len(output.target_ledger.events) == 2
    assert {event.track_id for event in output.target_ledger.events} == {"T1"}
    assert output.target_ledger.tracks[0].attributes["track_group"] == "speaker-1"
    restored = scene_from_dict(scene.to_manifest_dict())
    assert [source.track_group for source in restored.sources] == [
        "speaker-1",
        "speaker-1",
    ]


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


def test_explicit_crop_and_asymmetric_fades_are_replayable():
    class RampPool:
        def load(self, key: str, sample_rate: int) -> tuple[np.ndarray, float]:
            del key
            waveform = np.linspace(0.1, 0.8, sample_rate, dtype=np.float32)
            return waveform, 1.0

    source = PlacedSource(
        source_id="S01",
        kind="sfx",
        path="ramp",
        onset=0.0,
        gain_db=-6.0,
        text="A controlled ramp.",
        crop_start_sec=0.2,
        crop_duration_sec=0.4,
        fade_in_sec=0.0,
        fade_out_sec=0.1,
    )
    scene = Scene(
        scene_id="crop",
        seed=1,
        duration=0.5,
        template="isolated_sfx",
        sources=[source],
        conditions=Conditions(),
        supervision=Supervision(),
        sample_rate=1000,
    )

    first = render_scene(scene, RampPool())
    restored_scene = scene_from_dict(scene.to_manifest_dict())
    replay = render_scene(restored_scene, RampPool())
    stem = first.stems[0].stem

    assert first.mixture_hash() == replay.mixture_hash()
    assert stem.shape == (500,)
    assert stem[0] > 0.0
    assert stem[399] == pytest.approx(0.0)
    assert np.all(stem[400:] == 0.0)
    assert restored_scene.sources[0].crop_start_sec == 0.2
    assert restored_scene.sources[0].fade_out_sec == 0.1


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


def test_mixture_no_clipping(pool, sampler):
    scene = _scene(sampler, template="speech_music_sfx")
    out = render_scene(scene, pool)
    assert float(np.max(np.abs(out.mixture))) <= 0.99 + 1e-6

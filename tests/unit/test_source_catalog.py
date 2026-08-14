from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

from sceneledger.data.renderer import render_scene
from sceneledger.data.scene_graph_sampler import (
    CatalogSetSourcePool,
    CatalogSourcePool,
    SceneGraphSampler,
    SceneSamplerConfig,
)
from sceneledger.data.source_catalog import (
    CATALOG_SCHEMA_VERSION,
    SourceRecord,
    prepare_source_catalog,
    read_source_catalog,
    validate_source_audit,
    write_source_catalog,
)


def _write_tone(path: Path, *, frequency: float, sample_rate: int = 8000) -> None:
    time = np.arange(int(0.5 * sample_rate), dtype=np.float32) / sample_rate
    wav = 0.1 * np.sin(2 * np.pi * frequency * time)
    sf.write(path, wav, sample_rate, subtype="PCM_16")


def _record(source_id: str, kind: str, path: str, group: str) -> SourceRecord:
    return SourceRecord(
        schema_version=CATALOG_SCHEMA_VERSION,
        source_id=source_id,
        kind=kind,
        audio_path=path,
        source_group=group,
        caption=f"Audible content unique to {source_id}.",
        dataset="unit-test",
        license="CC0-1.0",
        annotation_origin="human",
        identity=f"identity:{group}" if kind in ("speech", "vocal") else None,
    )


def test_prepare_source_catalog_writes_group_disjoint_folds(tmp_path: Path) -> None:
    audio_root = tmp_path / "audio"
    audio_root.mkdir()
    records: list[SourceRecord] = []
    kinds = ("speech", "vocal", "music", "sfx", "ambience")
    frequency = 130.0
    # Every group contains all kinds, so an equal three-way group split can
    # satisfy the per-kind renderer requirement without leaking a group.
    for group_index in range(3):
        for kind in kinds:
            source_id = f"g{group_index}_{kind}"
            filename = f"{source_id}.wav"
            _write_tone(audio_root / filename, frequency=frequency)
            frequency += 13.0
            records.append(_record(source_id, kind, filename, f"recording:{group_index}"))
    for record in records:
        record.source_group = f"recording:{record.kind}:{record.source_group.rsplit(':', 1)[-1]}"

    raw = tmp_path / "raw.jsonl"
    write_source_catalog(raw, records)
    output = tmp_path / "catalogs"
    report = prepare_source_catalog(
        raw,
        output,
        audio_root=audio_root,
        allowed_licenses={"CC0-1.0"},
        split_ratios=(1.0, 1.0, 1.0),
        min_records_per_kind_per_split=1,
        min_groups_per_kind_per_split=1,
    )

    assert report["pass"] is True
    prepared = read_source_catalog(output / "all.jsonl")
    assert all(record.rms_dbfs is not None for record in prepared)
    assert all(record.active_rms_dbfs is not None for record in prepared)
    fold_records = {fold: read_source_catalog(output / f"{fold}.jsonl") for fold in ("train", "val", "test")}
    groups = {fold: {record.source_group for record in items} for fold, items in fold_records.items()}
    assert groups["train"].isdisjoint(groups["val"])
    assert groups["train"].isdisjoint(groups["test"])
    assert groups["val"].isdisjoint(groups["test"])
    assert all({record.kind for record in items} == set(kinds) for items in fold_records.values())
    assert (output / "source_audit.csv").is_file()
    persisted_report = json.loads((output / "source_catalog_report.json").read_text(encoding="utf-8"))
    assert set(persisted_report["artifacts"]) == {
        "all.jsonl",
        "train.jsonl",
        "val.jsonl",
        "test.jsonl",
        "source_audit.csv",
    }


def test_prepare_source_catalog_fails_closed_on_unapproved_license(tmp_path: Path) -> None:
    audio_root = tmp_path / "audio"
    audio_root.mkdir()
    _write_tone(audio_root / "one.wav", frequency=220.0)
    record = _record("one", "sfx", "one.wav", "recording:one").model_copy(
        update={"license": "unknown"}
    )
    raw = tmp_path / "raw.jsonl"
    write_source_catalog(raw, [record])
    output = tmp_path / "catalogs"

    report = prepare_source_catalog(
        raw,
        output,
        audio_root=audio_root,
        allowed_licenses={"CC0-1.0"},
        min_records_per_kind_per_split=0,
        min_groups_per_kind_per_split=0,
    )

    assert report["pass"] is False
    assert not (output / "train.jsonl").exists()
    failed = {check["name"] for check in report["checks"] if not check["pass"]}
    assert "licenses_allowlisted" in failed


def test_partial_catalog_can_gate_explicit_required_kinds(tmp_path: Path) -> None:
    audio_root = tmp_path / "audio"
    audio_root.mkdir()
    records: list[SourceRecord] = []
    for index, fold in enumerate(("train", "val", "test")):
        for kind_index, kind in enumerate(("sfx", "ambience")):
            filename = f"{fold}_{kind}.wav"
            _write_tone(audio_root / filename, frequency=410 + index * 21 + kind_index)
            records.append(
                _record(f"{fold}_{kind}", kind, filename, f"group:{fold}:{kind}").model_copy(
                    update={"split": fold}
                )
            )
    raw = tmp_path / "raw.jsonl"
    write_source_catalog(raw, records)
    output = tmp_path / "catalogs"
    report = prepare_source_catalog(
        raw,
        output,
        audio_root=audio_root,
        allowed_licenses={"CC0-1.0"},
        min_records_per_kind_per_split=1,
        min_groups_per_kind_per_split=1,
        required_kinds={"sfx", "ambience"},
    )
    assert report["pass"] is True
    assert report["required_kinds"] == ["ambience", "sfx"]


def test_catalog_source_pool_propagates_real_metadata_to_scene_and_ledger(tmp_path: Path) -> None:
    _write_tone(tmp_path / "speech.wav", frequency=190.0)
    _write_tone(tmp_path / "music.wav", frequency=310.0)
    records = [
        _record("speech_1", "speech", "speech.wav", "speaker:alice").model_copy(
            update={
                "caption": "Alice clearly says hello.",
                "identity": "speaker:alice",
                "split": "train",
                "duration_sec": 0.5,
                "file_sha256": "a" * 64,
                "rms_dbfs": -30.0,
                "active_rms_dbfs": -30.0,
                "text_is_verbatim": True,
            }
        ),
        _record("music_1", "music", "music.wav", "song:one").model_copy(
            update={
                "caption": "A quiet acoustic guitar phrase.",
                "split": "train",
                "duration_sec": 0.5,
                "file_sha256": "b" * 64,
                "rms_dbfs": -30.0,
                "active_rms_dbfs": -30.0,
            }
        ),
    ]
    catalog = tmp_path / "catalog.jsonl"
    write_source_catalog(catalog, records)
    pool = CatalogSourcePool(str(catalog), audio_root=str(tmp_path), expected_split="train")
    resampled, duration = pool.load("speech_1", 16000)
    assert len(resampled) == 8000
    assert duration == 0.5
    sampler = SceneGraphSampler(
        pool,
        SceneSamplerConfig(
            sample_rate=8000,
            duration_range=(1.0, 1.0),
            target_active_rms_dbfs_by_kind={
                "speech": (-20.0, -20.0),
                "music": (-32.0, -32.0),
            },
            max_abs_source_gain_db=24.0,
            p_rir=0.0,
            p_echo=0.0,
            ducking_probability=0.0,
        ),
    )

    scene = sampler.sample("real_001", seed=4, template="speech_over_music")
    speech = next(source for source in scene.sources if source.kind == "speech")
    assert speech.text == "Alice clearly says hello."
    assert speech.source_group == "speaker:alice"
    assert speech.source_labels == []
    assert speech.source_dataset == "unit-test"
    assert speech.source_license == "CC0-1.0"
    assert speech.annotation_origin == "human"
    assert speech.source_duration_sec == 0.5
    assert speech.source_file_sha256 == "a" * 64
    assert speech.source_rms_dbfs == -30.0
    assert speech.source_active_rms_dbfs == -30.0
    assert speech.gain_db == 10.0
    music = next(source for source in scene.sources if source.kind == "music")
    assert music.gain_db == -2.0
    assert scene.to_manifest_dict()["sources"][1]["source_file_sha256"] == "a" * 64
    assert scene.to_manifest_dict()["sources"][1]["source_rms_dbfs"] == -30.0
    assert scene.to_manifest_dict()["sources"][1]["source_active_rms_dbfs"] == -30.0
    assert scene.to_manifest_dict()["sources"][1]["source_labels"] == []

    output = render_scene(scene, pool)
    assert output.target_ledger.provenance.label_level == "B"
    assert output.target_ledger.provenance.source_dataset == "unit-test"
    assert output.target_ledger.provenance.license_status == "CC0-1.0"
    assert any(event.text == "Alice clearly says hello." for event in output.target_ledger.events)
    speech_event = next(event for event in output.target_ledger.events if event.type == "speech")
    assert speech_event.verbatim is True
    assert speech_event.confidence == 1.0


def test_catalog_sampler_never_truncates_transcript_bearing_speech(tmp_path: Path) -> None:
    for name, frequency in (("short.wav", 180.0), ("long.wav", 220.0), ("music.wav", 300.0)):
        _write_tone(tmp_path / name, frequency=frequency)
    records = [
        _record("speech_short", "speech", "short.wav", "speaker:short").model_copy(
            update={"duration_sec": 0.5, "split": "test"}
        ),
        _record("speech_long", "speech", "long.wav", "speaker:long").model_copy(
            update={"duration_sec": 3.0, "split": "test"}
        ),
        _record("music", "music", "music.wav", "song:one").model_copy(
            update={"duration_sec": 0.5, "split": "test"}
        ),
    ]
    catalog = tmp_path / "catalog.jsonl"
    write_source_catalog(catalog, records)
    pool = CatalogSourcePool(str(catalog), audio_root=str(tmp_path), expected_split="test")
    sampler = SceneGraphSampler(
        pool,
        SceneSamplerConfig(
            duration_range=(1.0, 1.0),
            p_rir=0.0,
            p_echo=0.0,
            ducking_probability=0.0,
        ),
    )

    for seed in range(10):
        scene = sampler.sample(f"fit-{seed}", seed=seed, template="speech_over_music")
        speech = next(source for source in scene.sources if source.kind == "speech")
        assert speech.path == "speech_short"
        assert speech.onset + float(speech.source_duration_sec or 0.0) <= scene.duration


def test_multi_speaker_template_samples_distinct_catalog_identities(tmp_path: Path) -> None:
    records: list[SourceRecord] = []
    for index in range(4):
        filename = f"speaker_{index}.wav"
        _write_tone(tmp_path / filename, frequency=200 + index * 20)
        records.append(
            _record(f"speech_{index}", "speech", filename, f"speaker:{index}").model_copy(
                update={"identity": f"speaker:{index}", "split": "train"}
            )
        )
    catalog = tmp_path / "speech.jsonl"
    write_source_catalog(catalog, records)
    pool = CatalogSourcePool(str(catalog), audio_root=str(tmp_path), expected_split="train")
    sampler = SceneGraphSampler(
        pool,
        SceneSamplerConfig(
            sample_rate=8000,
            duration_range=(1.0, 1.0),
            p_rir=0.0,
            p_echo=0.0,
        ),
    )
    scene = sampler.sample("speakers", seed=2, template="overlapping_speakers")
    assert len({source.identity for source in scene.sources}) == 2


def _single_kind_pool(
    root: Path,
    *,
    dataset: str,
    source_prefix: str,
    content_sha256: str,
) -> CatalogSourcePool:
    root.mkdir(parents=True)
    records: list[SourceRecord] = []
    for index in range(3):
        source_id = f"{source_prefix}_{index}"
        filename = f"{source_id}.wav"
        _write_tone(root / filename, frequency=510 + index)
        records.append(
            _record(source_id, "sfx", filename, f"{dataset}:clip:{index}").model_copy(
                update={
                    "dataset": dataset,
                    "split": "test",
                    "duration_sec": 0.5,
                    "content_sha256": (
                        content_sha256
                        if index == 0
                        else hashlib.sha256(source_id.encode()).hexdigest()
                    ),
                }
            )
        )
    catalog = root / "test.jsonl"
    write_source_catalog(catalog, records)
    return CatalogSourcePool(str(catalog), audio_root=str(root), expected_split="test")


def test_catalog_set_weights_banks_instead_of_raw_file_count(tmp_path: Path) -> None:
    dominant = _single_kind_pool(
        tmp_path / "dominant",
        dataset="dominant-bank",
        source_prefix="dominant",
        content_sha256="a" * 64,
    )
    minority = _single_kind_pool(
        tmp_path / "minority",
        dataset="minority-bank",
        source_prefix="minority",
        content_sha256="b" * 64,
    )
    pool = CatalogSetSourcePool([dominant, minority], sampling_weights=[9.0, 1.0])

    first_bank = [
        pool.metadata(pool.candidates("sfx", random.Random(seed), limit=1)[0])[
            "source_dataset"
        ]
        for seed in range(400)
    ]
    assert first_bank.count("dominant-bank") > 320
    picked_bank = [
        pool.metadata(pool.pick("sfx", random.Random(seed)))["source_dataset"]
        for seed in range(400)
    ]
    assert picked_bank.count("dominant-bank") > 320


def test_catalog_candidates_balance_primary_classes(tmp_path: Path) -> None:
    records: list[SourceRecord] = []
    for index in range(21):
        source_id = f"common_{index}" if index < 20 else "rare_0"
        filename = f"{source_id}.wav"
        _write_tone(tmp_path / filename, frequency=620 + index)
        records.append(
            _record(source_id, "sfx", filename, f"group:{source_id}").model_copy(
                update={
                    "labels": ["common" if index < 20 else "rare"],
                    "split": "test",
                    "duration_sec": 0.5,
                }
            )
        )
    catalog = tmp_path / "imbalanced.jsonl"
    write_source_catalog(catalog, records)
    pool = CatalogSourcePool(str(catalog), audio_root=str(tmp_path), expected_split="test")

    candidates = pool.candidates("sfx", random.Random(3), limit=2)
    assert {pool.metadata(key)["source_labels"][0] for key in candidates} == {
        "common",
        "rare",
    }


def test_catalog_set_queries_an_exact_recipe_label_across_banks(
    tmp_path: Path,
) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    first_records: list[SourceRecord] = []
    for index in range(300):
        filename = f"first_{index}.wav"
        _write_tone(first_root / filename, frequency=700 + index)
        first_records.append(
            _record(
                f"first_{index}", "sfx", filename, f"first:group:{index}"
            ).model_copy(
                update={
                    "labels": [f"class_{index:03d}"],
                    "dataset": "first-bank",
                    "split": "test",
                    "duration_sec": 0.5,
                }
            )
        )
    rare_file = second_root / "rare.wav"
    _write_tone(rare_file, frequency=1100)
    second_records = [
        _record("rare", "sfx", rare_file.name, "second:rare").model_copy(
            update={
                "labels": ["rare_target"],
                "dataset": "second-bank",
                "split": "test",
                "duration_sec": 0.5,
            }
        )
    ]
    first_catalog = first_root / "test.jsonl"
    second_catalog = second_root / "test.jsonl"
    write_source_catalog(first_catalog, first_records)
    write_source_catalog(second_catalog, second_records)
    pool = CatalogSetSourcePool(
        [
            CatalogSourcePool(
                str(first_catalog), audio_root=str(first_root), expected_split="test"
            ),
            CatalogSourcePool(
                str(second_catalog), audio_root=str(second_root), expected_split="test"
            ),
        ]
    )

    candidates = pool.candidates_for_label(
        "sfx", "rare_target", random.Random(4), limit=1
    )

    assert candidates == ["rare"]


def test_catalog_set_rejects_cross_bank_content_leakage(tmp_path: Path) -> None:
    first = _single_kind_pool(
        tmp_path / "first",
        dataset="first-bank",
        source_prefix="first",
        content_sha256="c" * 64,
    )
    second = _single_kind_pool(
        tmp_path / "second",
        dataset="second-bank",
        source_prefix="second",
        content_sha256="c" * 64,
    )

    try:
        CatalogSetSourcePool([first, second])
    except ValueError as exc:
        assert "cross-catalog source leakage" in str(exc)
        assert "content_sha256" in str(exc)
    else:
        raise AssertionError("cross-catalog duplicate content was accepted")


def test_fixed_split_hints_are_respected(tmp_path: Path) -> None:
    audio_root = tmp_path / "audio"
    audio_root.mkdir()
    records: list[SourceRecord] = []
    for index, fold in enumerate(("train", "val", "test")):
        for kind_index, kind in enumerate(("speech", "vocal", "music", "sfx", "ambience")):
            source_id = f"{fold}_{kind}"
            filename = f"{source_id}.wav"
            _write_tone(audio_root / filename, frequency=150 + 31 * index + kind_index)
            records.append(
                _record(source_id, kind, filename, f"group:{fold}").model_copy(
                    update={"split": fold}
                )
            )
    raw = tmp_path / "raw.jsonl"
    write_source_catalog(raw, records)
    output = tmp_path / "catalogs"
    report = prepare_source_catalog(
        raw,
        output,
        audio_root=audio_root,
        allowed_licenses={"CC0-1.0"},
        min_records_per_kind_per_split=1,
        min_groups_per_kind_per_split=1,
    )
    assert report["pass"] is True
    for fold in ("train", "val", "test"):
        assert {record.source_group for record in read_source_catalog(output / f"{fold}.jsonl")} == {
            f"group:{fold}"
        }


def test_secondary_leakage_identity_keeps_distinct_files_in_one_fold(tmp_path: Path) -> None:
    audio_root = tmp_path / "audio"
    audio_root.mkdir()
    records: list[SourceRecord] = []
    kinds = ("speech", "vocal", "music", "sfx", "ambience")
    for index, kind in enumerate(kinds):
        for take in range(3):
            source_id = f"{kind}_{take}"
            filename = f"{source_id}.wav"
            _write_tone(audio_root / filename, frequency=350 + index * 17 + take)
            record = _record(source_id, kind, filename, f"recording:{source_id}")
            if take < 2:
                record = record.model_copy(update={"leakage_groups": [f"creator:{kind}"]})
            records.append(record)
    raw = tmp_path / "raw.jsonl"
    write_source_catalog(raw, records)
    output = tmp_path / "catalogs"
    prepare_source_catalog(
        raw,
        output,
        audio_root=audio_root,
        allowed_licenses={"CC0-1.0"},
        split_ratios=(1.0, 1.0, 1.0),
        min_records_per_kind_per_split=0,
        min_groups_per_kind_per_split=0,
    )
    folds_by_creator: dict[str, set[str]] = {}
    for fold in ("train", "val", "test"):
        for record in read_source_catalog(output / f"{fold}.jsonl") if (output / f"{fold}.jsonl").exists() else []:
            for group in record.leakage_groups:
                folds_by_creator.setdefault(group, set()).add(fold)
    assert all(len(folds) == 1 for folds in folds_by_creator.values())


def test_validate_source_audit_binds_catalog_hashes(tmp_path: Path) -> None:
    audio_root = tmp_path / "audio"
    audio_root.mkdir()
    records: list[SourceRecord] = []
    for index, fold in enumerate(("train", "val", "test")):
        for kind_index, kind in enumerate(("speech", "vocal", "music", "sfx", "ambience")):
            source_id = f"{fold}_{kind}"
            filename = f"{source_id}.wav"
            _write_tone(audio_root / filename, frequency=240 + 31 * index + kind_index)
            records.append(
                _record(source_id, kind, filename, f"group:{fold}").model_copy(
                    update={"split": fold}
                )
            )
    raw = tmp_path / "raw.jsonl"
    write_source_catalog(raw, records)
    catalogs = tmp_path / "catalogs"
    assert prepare_source_catalog(
        raw,
        catalogs,
        audio_root=audio_root,
        allowed_licenses={"CC0-1.0"},
        min_records_per_kind_per_split=1,
        min_groups_per_kind_per_split=1,
        audit_per_kind=3,
    )["pass"]

    audit_path = catalogs / "source_audit.csv"
    with audit_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    for row in rows:
        row["audible_y_n"] = "y"
        row["caption_correct_y_n"] = "y"
        row["kind_correct_y_n"] = "y"
    assert Counter(row["split"] for row in rows) == {
        "train": 5,
        "val": 5,
        "test": 5,
    }
    with audit_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    audit_report = validate_source_audit(
        catalogs / "source_catalog_report.json",
        audit_path,
        catalogs / "source_audit_report.json",
        min_per_kind=3,
        required_splits={"test"},
        min_per_kind_per_required_split=1,
    )
    assert audit_report["pass"] is True
    assert audit_report["counts_by_split_kind"]["test"] == {
        "ambience": 1,
        "music": 1,
        "sfx": 1,
        "speech": 1,
        "vocal": 1,
    }

    tampered_audit_path = catalogs / "source_audit_tampered.csv"
    tampered_rows = [dict(row) for row in rows]
    tampered_rows[0]["caption"] = "a different task caption"
    with tampered_audit_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(tampered_rows)
    tampered_tasks = validate_source_audit(
        catalogs / "source_catalog_report.json",
        tampered_audit_path,
        catalogs / "source_audit_report_changed_tasks.json",
        min_per_kind=3,
    )
    assert tampered_tasks["pass"] is False
    assert any(
        check["name"] == "audit_tasks_frozen" and not check["pass"]
        for check in tampered_tasks["checks"]
    )
    audited_pool = CatalogSourcePool(
        str(catalogs / "train.jsonl"),
        audio_root=str(audio_root),
        audit_report_path=str(catalogs / "source_audit_report.json"),
        expected_split="train",
    )
    assert audited_pool.metadata("train_speech")["text"] == "Audible content unique to train_speech."

    original_audio = (audio_root / "train_speech.wav").read_bytes()
    (audio_root / "train_speech.wav").write_bytes(original_audio + b"tampered")
    try:
        tampered_pool = CatalogSourcePool(
            str(catalogs / "train.jsonl"),
            audio_root=str(audio_root),
            audit_report_path=str(catalogs / "source_audit_report.json"),
            expected_split="train",
        )
        tampered_pool.metadata("train_speech")
    except ValueError as exc:
        assert "source audio changed" in str(exc)
    else:
        raise AssertionError("catalog with changed source waveform was accepted")
    (audio_root / "train_speech.wav").write_bytes(original_audio)

    preparation_path = catalogs / "source_catalog_report.json"
    original_preparation = preparation_path.read_text(encoding="utf-8")
    preparation_path.write_text(original_preparation + "\n", encoding="utf-8")
    try:
        CatalogSourcePool(
            str(catalogs / "train.jsonl"),
            audio_root=str(audio_root),
            audit_report_path=str(catalogs / "source_audit_report.json"),
            expected_split="train",
        )
    except ValueError as exc:
        assert "preparation report changed" in str(exc)
    else:
        raise AssertionError("catalog with changed preparation report was accepted")
    preparation_path.write_text(original_preparation, encoding="utf-8")

    with (catalogs / "train.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("\n")
    try:
        CatalogSourcePool(
            str(catalogs / "train.jsonl"),
            audio_root=str(audio_root),
            audit_report_path=str(catalogs / "source_audit_report.json"),
            expected_split="train",
        )
    except ValueError as exc:
        assert "not bound" in str(exc)
    else:
        raise AssertionError("tampered catalog was accepted by CatalogSourcePool")
    tampered = validate_source_audit(
        catalogs / "source_catalog_report.json",
        audit_path,
        catalogs / "source_audit_report_tampered.json",
        min_per_kind=3,
    )
    assert tampered["pass"] is False
    assert any(
        check["name"] == "train.jsonl_matches_preparation" and not check["pass"]
        for check in tampered["checks"]
    )

from __future__ import annotations

import json
from pathlib import Path

import yaml
from fixtures.factory import ev, ledger, t, tr

from sceneledger.cli.forensic_replay import main as replay_main
from sceneledger.data.manifests import ManifestEntry, write_manifest
from sceneledger.data.schema import Ledger
from sceneledger.eval.metrics import METRICS_SCHEMA_VERSION
from sceneledger.models.target_formatter import format_slot_aware_caption


def _entry(sample_id: str, text: str) -> ManifestEntry:
    target = ledger(
        sample_id,
        2.0,
        events=[ev("E1", "sfx", [t(0.2, 0.8)], text=text, track_id="T1")],
        tracks=[tr("T1", "sfx", [t(0.2, 0.8)])],
    )
    return ManifestEntry(
        scene={
            "scene_id": sample_id,
            "seed": 1,
            "duration": 2.0,
            "template": "isolated_sfx",
            "sources": [
                {
                    "source_id": "S1",
                    "kind": "sfx",
                    "path": f"sources/{sample_id}.wav",
                    "source_group": f"group:{sample_id}",
                }
            ],
        },
        mixture_path=f"audio/{sample_id}.wav",
        stem_paths={"S1": f"stems/{sample_id}.wav"},
        mixture_hash="mix",
        dry_mixture_hash="dry",
        stem_hashes={"S1": "stem"},
        activity_hashes={"S1": "activity"},
        target_ledger=target.model_dump(mode="json"),
        sample_rate=16000,
    )


def test_forensic_replay_recovers_semantics_and_separates_seen_samples(
    tmp_path: Path,
) -> None:
    seen = _entry("seen", "a glass breaks")
    unseen = _entry("unseen", "a dog barks")
    train_manifest = tmp_path / "train.jsonl"
    eval_manifest = tmp_path / "eval.jsonl"
    write_manifest(train_manifest, [seen])
    write_manifest(eval_manifest, [seen, unseen])
    config = tmp_path / "train.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "data": {"manifest_path": str(train_manifest), "pre_split": True},
                "train": {"seed": 7},
            }
        ),
        encoding="utf-8",
    )
    inference = tmp_path / "inference.json"
    rows = []
    for entry in (seen, unseen):
        target = Ledger.model_validate(entry.target_ledger)
        rows.append(
            {
                "sample_id": target.sample_id,
                "strict_format_success": True,
                "warnings": [],
                "raw_text": format_slot_aware_caption(target, style="detailed"),
            }
        )
    inference.write_text(
        json.dumps(
            {
                "n_samples": 2,
                "strict_format_success_rate": 1.0,
                "samples": rows,
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "replay"

    assert (
        replay_main(
            [
                "--train-config",
                str(config),
                "--manifest",
                str(eval_manifest),
                "--inference-report",
                str(inference),
                "--repo-root",
                str(tmp_path),
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )

    metrics = json.loads((output / "metrics.replayed.json").read_text(encoding="utf-8"))
    report = json.loads((output / "forensic_replay.json").read_text(encoding="utf-8"))
    assert metrics["schema_version"] == METRICS_SCHEMA_VERSION
    assert metrics["macro_caption_token_f1"] == 1.0
    assert report["paper_eligible"] is False
    assert report["counts"]["seen_during_training"] == 1
    assert report["counts"]["unseen_by_sample_id"] == 1
    assert report["subgroups"]["unseen_by_sample_id"]["macro_event_f1"] == 1.0
    assert report["replayed_artifacts"]["metrics"]["path"] == "metrics.replayed.json"

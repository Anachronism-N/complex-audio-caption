from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from sceneledger.data.fsd50k import convert_fsd50k_records


def _write_fixture(root: Path, *, eval_uploader: str = "eval-user") -> None:
    ground_truth = root / "FSD50K.ground_truth"
    metadata = root / "FSD50K.metadata"
    ground_truth.mkdir(parents=True)
    metadata.mkdir(parents=True)
    with (ground_truth / "dev.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["fname", "labels", "mids", "split"])
        writer.writeheader()
        writer.writerows(
            [
                {
                    "fname": "1",
                    "labels": "Dog,Animal",
                    "mids": "/m/dog,/m/animal",
                    "split": "train",
                },
                {
                    "fname": "2",
                    "labels": "Rain,Natural_sounds",
                    "mids": "/m/rain,/m/natural",
                    "split": "val",
                },
                {
                    "fname": "4",
                    "labels": "Guitar,Musical_instrument,Music",
                    "mids": "/m/guitar,/m/instrument,/m/music",
                    "split": "train",
                },
                {
                    "fname": "5",
                    "labels": "Clapping,Human_sounds",
                    "mids": "/m/clap,/m/human",
                    "split": "train",
                },
            ]
        )
    with (ground_truth / "eval.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["fname", "labels", "mids"])
        writer.writeheader()
        writer.writerow(
            {"fname": "3", "labels": "Thunder,Natural_sounds", "mids": "/m/thunder,/m/natural"}
        )
    clip = lambda uploader, license_url: {  # noqa: E731 - compact metadata fixture
        "uploader": uploader,
        "license": license_url,
        "title": "fixture",
        "description": "fixture",
        "tags": [],
    }
    cc0 = "http://creativecommons.org/publicdomain/zero/1.0/"
    by_nc = "http://creativecommons.org/licenses/by-nc/3.0/"
    (metadata / "dev_clips_info_FSD50K.json").write_text(
        json.dumps(
            {
                "1": clip("shared-user", cc0),
                "2": clip("shared-user", cc0),
                "4": clip("music-user", cc0),
                "5": clip("nc-user", by_nc),
            }
        ),
        encoding="utf-8",
    )
    (metadata / "eval_clips_info_FSD50K.json").write_text(
        json.dumps({"3": clip(eval_uploader, cc0)}), encoding="utf-8"
    )
    (metadata / "pp_pnp_ratings_FSD50K.json").write_text(
        json.dumps(
            {
                "1": {"/m/dog": [1.0, 1.0]},
                "2": {"/m/rain": [1.0, 1.0]},
                "3": {"/m/thunder": [1.0, 1.0]},
                "4": {"/m/guitar": [1.0, 1.0]},
                "5": {"/m/clap": [1.0, 1.0]},
            }
        ),
        encoding="utf-8",
    )


def test_fsd50k_filters_license_music_and_resolves_uploader_split(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    records = convert_fsd50k_records(
        tmp_path, allowed_licenses={"CC0-1.0"}, include_eval=True
    )

    assert {record.source_id for record in records} == {
        "fsd50k:1",
        "fsd50k:2",
        "fsd50k:3",
    }
    by_id = {record.source_id: record for record in records}
    assert by_id["fsd50k:1"].split == "train"
    assert by_id["fsd50k:2"].split == "train"
    assert by_id["fsd50k:2"].kind == "ambience"
    assert by_id["fsd50k:3"].split == "test"
    assert by_id["fsd50k:3"].audio_path == "FSD50K.eval_audio/3.wav"
    assert by_id["fsd50k:3"].caption == "FSD50K predominant class label: Thunder."
    assert by_id["fsd50k:1"].source_group == "freesound-uploader:shared-user"


def test_fsd50k_rejects_eval_uploader_leakage(tmp_path: Path) -> None:
    _write_fixture(tmp_path, eval_uploader="shared-user")
    with pytest.raises(ValueError, match="eval uploader overlaps"):
        convert_fsd50k_records(
            tmp_path, allowed_licenses={"CC0-1.0"}, include_eval=True
        )


def test_fsd50k_requires_predominant_label_consensus(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    ratings_path = tmp_path / "FSD50K.metadata" / "pp_pnp_ratings_FSD50K.json"
    ratings = json.loads(ratings_path.read_text(encoding="utf-8"))
    ratings["1"] = {"/m/dog": [1.0]}
    ratings_path.write_text(json.dumps(ratings), encoding="utf-8")
    records = convert_fsd50k_records(
        tmp_path, allowed_licenses={"CC0-1.0"}, include_eval=True
    )
    assert "fsd50k:1" not in {record.source_id for record in records}

from __future__ import annotations

from pathlib import Path

from sceneledger.data.source_catalog import SourceRecord, write_source_catalog
from sceneledger.data.source_policy import load_source_bank_policy, validate_source_bank_policy

REPO_ROOT = Path(__file__).resolve().parents[2]
POLICY = REPO_ROOT / "configs" / "data" / "source_bank_policy.yaml"


def test_repository_source_policy_profiles_are_internally_valid() -> None:
    policy = load_source_bank_policy(POLICY)
    for profile_name in policy.profiles:
        report = validate_source_bank_policy(policy, profile_name=profile_name)
        assert report["pass"], [
            check for check in report["checks"] if not check["pass"]
        ]


def test_policy_rejects_non_verbatim_speech_catalog(tmp_path: Path) -> None:
    policy = load_source_bank_policy(POLICY)
    catalog = tmp_path / "libri.jsonl"
    write_source_catalog(
        catalog,
        [
            SourceRecord(
                source_id="bad",
                kind="speech",
                audio_path="bad.wav",
                source_group="speaker:1",
                caption="a paraphrase",
                dataset="LibriSpeech",
                license="CC BY 4.0",
                annotation_origin="dataset",
                text_is_verbatim=False,
                split="train",
            )
        ],
    )

    report = validate_source_bank_policy(
        policy,
        profile_name="d0_anchor_research",
        catalogs={"librispeech": catalog},
    )

    failed = {check["name"] for check in report["checks"] if not check["pass"]}
    assert "catalog_transcripts_verbatim:librispeech" in failed

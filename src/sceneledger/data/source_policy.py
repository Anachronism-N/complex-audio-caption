"""Machine-checkable policy for selecting mixture source banks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from sceneledger.data.source_catalog import CatalogKind, read_source_catalog

POLICY_SCHEMA_VERSION = "sceneledger.source_bank_policy.v1"
Claim = Literal[
    "source_class",
    "speech_transcript",
    "instrument_set",
    "vocal_presence",
    "lyrics_nonverbatim",
    "lyrics_verbatim",
    "clip_timing",
    "stem_timing",
]


class DatasetPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["implemented", "planned", "gated"]
    source_type: Literal["semantic", "corruption", "real_domain"]
    roles: set[CatalogKind] = Field(default_factory=set)
    licenses: set[str] = Field(default_factory=set)
    acquisition: Literal["automatic", "manual", "mdc_manual", "local_only"]
    redistributable_by_project: bool
    supported_claims: set[Claim] = Field(default_factory=set)
    required_gates: list[str] = Field(default_factory=list)
    note: str = Field(..., min_length=1)


class DatasetSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: str = Field(..., min_length=1)
    roles: set[CatalogKind]
    allowed_licenses: set[str]
    claims: set[Claim]


class ProfilePolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stage: Literal["D0", "D1", "D2", "D3"]
    runnable: bool
    objective: str = Field(..., min_length=1)
    selections: list[DatasetSelection]


class SourceBankPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[POLICY_SCHEMA_VERSION] = POLICY_SCHEMA_VERSION
    datasets: dict[str, DatasetPolicy]
    profiles: dict[str, ProfilePolicy]


def load_source_bank_policy(path: str | Path) -> SourceBankPolicy:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return SourceBankPolicy.model_validate(payload)


def _check(name: str, passed: bool, detail: object) -> dict[str, object]:
    return {"name": name, "pass": bool(passed), "detail": detail}


def validate_source_bank_policy(
    policy: SourceBankPolicy,
    *,
    profile_name: str,
    catalogs: dict[str, str | Path] | None = None,
    require_catalogs: bool = False,
) -> dict[str, object]:
    """Validate one profile and, optionally, its prepared catalog artifacts."""
    if profile_name not in policy.profiles:
        raise ValueError(
            f"unknown source-bank profile {profile_name!r}; "
            f"choose one of {sorted(policy.profiles)}"
        )
    profile = policy.profiles[profile_name]
    checks: list[dict[str, object]] = []
    selected_ids = [selection.dataset for selection in profile.selections]
    duplicate_ids = sorted({item for item in selected_ids if selected_ids.count(item) > 1})
    checks.append(_check("dataset_selections_unique", not duplicate_ids, duplicate_ids))

    for selection in profile.selections:
        dataset = policy.datasets.get(selection.dataset)
        checks.append(
            _check(
                f"dataset_known:{selection.dataset}",
                dataset is not None,
                selection.dataset,
            )
        )
        if dataset is None:
            continue
        checks.extend(
            [
                _check(
                    f"semantic_source:{selection.dataset}",
                    dataset.source_type == "semantic",
                    dataset.source_type,
                ),
                _check(
                    f"roles_supported:{selection.dataset}",
                    bool(selection.roles) and selection.roles <= dataset.roles,
                    {
                        "selected": sorted(selection.roles),
                        "supported": sorted(dataset.roles),
                    },
                ),
                _check(
                    f"licenses_allowlisted:{selection.dataset}",
                    bool(selection.allowed_licenses)
                    and selection.allowed_licenses <= dataset.licenses,
                    {
                        "selected": sorted(selection.allowed_licenses),
                        "published": sorted(dataset.licenses),
                    },
                ),
                _check(
                    f"claims_supported:{selection.dataset}",
                    selection.claims <= dataset.supported_claims,
                    sorted(selection.claims - dataset.supported_claims),
                ),
                _check(
                    f"implementation_ready:{selection.dataset}",
                    not profile.runnable or dataset.status == "implemented",
                    dataset.status,
                ),
                _check(
                    f"verbatim_lyrics_role:{selection.dataset}",
                    "lyrics_verbatim" not in selection.claims or "vocal" in selection.roles,
                    sorted(selection.roles),
                ),
                _check(
                    f"transcript_role:{selection.dataset}",
                    "speech_transcript" not in selection.claims
                    or "speech" in selection.roles,
                    sorted(selection.roles),
                ),
            ]
        )

    catalog_paths = catalogs or {}
    unknown_catalogs = sorted(set(catalog_paths) - set(selected_ids))
    checks.append(_check("catalog_arguments_selected", not unknown_catalogs, unknown_catalogs))
    if require_catalogs:
        missing = sorted(set(selected_ids) - set(catalog_paths))
        checks.append(_check("all_catalogs_supplied", not missing, missing))

    selection_by_id = {selection.dataset: selection for selection in profile.selections}
    catalog_summary: dict[str, object] = {}
    for dataset_id, catalog_path in sorted(catalog_paths.items()):
        selection = selection_by_id.get(dataset_id)
        if selection is None:
            continue
        try:
            records = read_source_catalog(catalog_path)
        except Exception as exc:
            checks.append(_check(f"catalog_readable:{dataset_id}", False, str(exc)))
            continue
        observed_roles = {record.kind for record in records}
        observed_licenses = {record.license for record in records}
        checks.extend(
            [
                _check(f"catalog_readable:{dataset_id}", True, len(records)),
                _check(
                    f"catalog_roles:{dataset_id}",
                    observed_roles <= selection.roles,
                    {
                        "observed": sorted(observed_roles),
                        "allowed": sorted(selection.roles),
                    },
                ),
                _check(
                    f"catalog_licenses:{dataset_id}",
                    observed_licenses <= selection.allowed_licenses,
                    {
                        "observed": sorted(observed_licenses),
                        "allowed": sorted(selection.allowed_licenses),
                    },
                ),
                _check(
                    f"catalog_transcripts_verbatim:{dataset_id}",
                    "speech_transcript" not in selection.claims
                    or all(
                        record.text_is_verbatim
                        for record in records
                        if record.kind == "speech"
                    ),
                    "speech records must set text_is_verbatim=true",
                ),
                _check(
                    f"catalog_lyrics_verbatim:{dataset_id}",
                    "lyrics_verbatim" not in selection.claims
                    or all(
                        record.text_is_verbatim
                        for record in records
                        if record.kind == "vocal"
                    ),
                    "vocal records must set text_is_verbatim=true",
                ),
            ]
        )
        catalog_summary[dataset_id] = {
            "path": str(catalog_path),
            "records": len(records),
            "roles": sorted(observed_roles),
            "licenses": sorted(observed_licenses),
        }

    return {
        "schema_version": "sceneledger.source_bank_policy_report.v1",
        "profile": profile_name,
        "stage": profile.stage,
        "runnable": profile.runnable,
        "pass": all(bool(check["pass"]) for check in checks),
        "catalogs": catalog_summary,
        "checks": checks,
    }


def write_policy_report(path: str | Path, report: dict[str, object]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


__all__ = [
    "POLICY_SCHEMA_VERSION",
    "DatasetPolicy",
    "DatasetSelection",
    "ProfilePolicy",
    "SourceBankPolicy",
    "load_source_bank_policy",
    "validate_source_bank_policy",
    "write_policy_report",
]

"""Blinded paired listening review for Rule versus LLM mixture planners.

The two arms must originate from the same frozen source-timeline tasks.  This
module verifies recipe/manifest bindings, materializes anonymously named
mixtures and stems, freezes the review sheet, and only reveals arm assignments
while summarizing completed reviews.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import secrets
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from sceneledger.data.experiment_data import file_sha256
from sceneledger.data.manifests import ManifestEntry, read_manifest
from sceneledger.data.scene_recipes import (
    SceneRecipe,
    compare_recipe_sets,
    read_recipes,
)
from sceneledger.data.schema import Ledger

REVIEW_SCHEMA_VERSION = "sceneledger-mixture-planner-review-v1"

IMMUTABLE_FIELDS = (
    "review_id",
    "task_id",
    "recipe_id",
    "template",
    "duration_sec",
    "audio_a_path",
    "a_stem_paths_json",
    "a_expected_events_json",
    "audio_b_path",
    "b_stem_paths_json",
    "b_expected_events_json",
)

RATING_NAMES = (
    "all_sources_audible",
    "scene_plausibility",
    "temporal_plausibility",
    "naturalness",
    "caption_support",
    "timestamp_alignment",
)

REVIEW_FIELDS = (
    "reviewer",
    "reviewed_at_utc",
    *(f"a_{name}_1_5" for name in RATING_NAMES),
    "a_inaudible_sources_count",
    "a_unsupported_labels_count",
    *(f"b_{name}_1_5" for name in RATING_NAMES),
    "b_inaudible_sources_count",
    "b_unsupported_labels_count",
    "preference_a_b_tie",
    "notes",
)

CSV_FIELDS = IMMUTABLE_FIELDS + REVIEW_FIELDS


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _rank(seed: str, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def _tasks_sha256(rows: list[dict[str, str]]) -> str:
    payload = [
        {field: row.get(field, "") for field in IMMUTABLE_FIELDS if field != "review_id"}
        for row in rows
    ]
    return _canonical_hash(payload)


def _review_id(
    *,
    seed: str,
    sample_count: int,
    tasks_sha256: str,
    artifacts: dict[str, str],
    package_files: dict[str, str],
) -> str:
    return _canonical_hash(
        {
            "schema_version": REVIEW_SCHEMA_VERSION,
            "seed": seed,
            "sample_count": sample_count,
            "tasks_sha256": tasks_sha256,
            "artifacts": artifacts,
            "package_files": package_files,
        }
    )


def _event_json(entry: ManifestEntry) -> str:
    ledger = Ledger.model_validate(entry.target_ledger)
    payload = [
        {
            "type": event.type,
            "track_id": event.track_id,
            "spans": [[span.start_sec, span.end_sec] for span in event.spans],
            "text": event.text,
        }
        for event in ledger.events
    ]
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _select_stratified(recipes: list[SceneRecipe], count: int, seed: str) -> list[SceneRecipe]:
    if count < 1 or count > len(recipes):
        raise ValueError(f"review sample count must be in [1, {len(recipes)}], observed={count}")
    groups: defaultdict[str, list[SceneRecipe]] = defaultdict(list)
    for recipe in recipes:
        groups[recipe.template].append(recipe)
    for template in groups:
        groups[template].sort(key=lambda recipe: _rank(seed, recipe.recipe_id))
    selected: list[SceneRecipe] = []
    offset = 0
    while len(selected) < count:
        added = False
        for template in sorted(groups):
            if offset < len(groups[template]):
                selected.append(groups[template][offset])
                added = True
                if len(selected) == count:
                    break
        if not added:
            break
        offset += 1
    return selected


def _entry_map(
    entries: list[ManifestEntry],
    recipes: list[SceneRecipe],
    *,
    arm: str,
    recipe_plan_sha256: str,
) -> dict[str, ManifestEntry]:
    expected = {recipe.recipe_id: recipe for recipe in recipes}
    if len(expected) != len(recipes):
        raise ValueError(f"{arm} recipe IDs are not unique")
    result: dict[str, ManifestEntry] = {}
    for entry in entries:
        metadata = entry.scene.get("recipe_metadata")
        if not isinstance(metadata, dict):
            raise ValueError(f"{arm} manifest scene lacks recipe_metadata")
        recipe_id = str(metadata.get("recipe_id") or "")
        recipe = expected.get(recipe_id)
        if recipe is None or recipe_id in result:
            raise ValueError(f"{arm} manifest has unknown or duplicate recipe_id: {recipe_id!r}")
        if metadata.get("recipe_plan_sha256") != recipe_plan_sha256:
            raise ValueError(f"{arm} manifest is bound to another recipe plan")
        expected_plan = [item.model_dump() for item in recipe.source_plan]
        if metadata.get("source_plan") != expected_plan:
            raise ValueError(f"{arm} manifest source plan changed for {recipe_id}")
        if abs(float(entry.scene["duration"]) - float(recipe.scene_duration_sec or 0.0)) > 1e-6:
            raise ValueError(f"{arm} manifest duration changed for {recipe_id}")
        occurrences: Counter[str] = Counter()
        actual_by_slot: dict[str, dict[str, object]] = {}
        for source in entry.scene.get("sources", []):
            kind = str(source.get("kind") or "")
            occurrences[kind] += 1
            actual_by_slot[f"{kind}_{occurrences[kind]}"] = source
        planned_slots = {item.slot_id for item in recipe.source_plan}
        if set(actual_by_slot) != planned_slots:
            raise ValueError(f"{arm} manifest source slots changed for {recipe_id}")
        for planned in recipe.source_plan:
            actual = actual_by_slot.get(planned.slot_id)
            if actual is None:
                raise ValueError(f"{arm} manifest is missing {planned.slot_id}")
            if str(actual.get("path") or "") != planned.catalog_source_id:
                raise ValueError(
                    f"{arm} manifest selected another source for {recipe_id}:{planned.slot_id}"
                )
            if abs(float(actual.get("onset", -1.0)) - planned.onset_sec) > 1e-6:
                raise ValueError(f"{arm} manifest changed onset for {recipe_id}:{planned.slot_id}")
        source_id_list = [str(source.get("source_id") or "") for source in entry.scene["sources"]]
        source_ids = set(source_id_list)
        if "" in source_ids or len(source_ids) != len(source_id_list):
            raise ValueError(f"{arm} manifest has empty or duplicate source IDs")
        if not source_ids or set(entry.stem_paths) != source_ids:
            raise ValueError(f"{arm} manifest lacks one isolated stem per source")
        if (
            not entry.mixture_hash
            or not entry.dry_mixture_hash
            or set(entry.stem_hashes) != source_ids
            or set(entry.activity_hashes) != source_ids
        ):
            raise ValueError(f"{arm} manifest lacks waveform/activity evidence")
        ledger = Ledger.model_validate(entry.target_ledger)
        if ledger.sample_id != str(entry.scene["scene_id"]):
            raise ValueError(f"{arm} manifest Ledger is bound to another scene")
        result[recipe_id] = entry
    missing = sorted(set(expected) - set(result))
    if missing or len(result) != len(expected):
        raise ValueError(f"{arm} manifest does not cover its recipe plan: {missing[:20]}")
    return result


def _resolve_audio(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"review audio is missing: {path}")
    return path


def _require_quality_report(
    report_path: str | Path,
    manifest_path: str | Path,
    *,
    arm: str,
) -> dict[str, Any]:
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    if report.get("schema_version") != "sceneledger-mixture-quality-v1":
        raise ValueError(f"{arm} mixture quality report schema is unsupported")
    if report.get("pass") is not True or report.get("failed_checks"):
        raise ValueError(
            f"{arm} mixture quality report has not passed: {report.get('failed_checks', [])}"
        )
    observed_hash = file_sha256(manifest_path)
    if report.get("manifest_sha256") != observed_hash:
        raise ValueError(f"{arm} mixture quality report is bound to another manifest")
    return report


def _copy_frozen(source: Path, destination: Path) -> str:
    source_hash = file_sha256(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not destination.is_file() or file_sha256(destination) != source_hash:
            raise ValueError(f"review package target already exists with other data: {destination}")
    else:
        shutil.copy2(source, destination)
    if file_sha256(destination) != source_hash:
        raise ValueError(f"copied review artifact hash mismatch: {destination}")
    return source_hash


def _materialize_side(
    *,
    entry: ManifestEntry,
    audio_root: Path,
    package_root: Path,
    task_id: str,
    side: str,
) -> tuple[str, str, dict[str, str]]:
    mixture_source = _resolve_audio(audio_root, entry.mixture_path)
    mixture_relative = Path("audio") / f"{task_id}_{side.upper()}{mixture_source.suffix}"
    mixture_hash = _copy_frozen(mixture_source, package_root / mixture_relative)
    stem_paths: dict[str, str] = {}
    package_hashes = {mixture_relative.as_posix(): mixture_hash}
    for index, (_source_id, value) in enumerate(sorted(entry.stem_paths.items()), 1):
        stem_source = _resolve_audio(audio_root, value)
        stem_relative = Path("stems") / f"{task_id}_{side.upper()}_S{index:02d}{stem_source.suffix}"
        stem_hash = _copy_frozen(stem_source, package_root / stem_relative)
        stem_paths[f"S{index:02d}"] = stem_relative.as_posix()
        package_hashes[stem_relative.as_posix()] = stem_hash
    return (
        mixture_relative.as_posix(),
        json.dumps(stem_paths, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        package_hashes,
    )


def prepare_mixture_review(
    *,
    rule_recipes_path: str | Path,
    rule_manifest_path: str | Path,
    rule_quality_report_path: str | Path,
    rule_audio_base: str | Path,
    llm_recipes_path: str | Path,
    llm_manifest_path: str | Path,
    llm_quality_report_path: str | Path,
    llm_audio_base: str | Path,
    package_dir: str | Path,
    sample_count: int,
    seed: str = "sceneledger-mixture-planner-review-v1",
    blinding_salt: str | None = None,
) -> tuple[list[dict[str, str]], dict[str, Any], dict[str, Any]]:
    """Prepare an anonymous paired listening package from matched render arms."""

    rule_recipes = read_recipes(rule_recipes_path)
    llm_recipes = read_recipes(llm_recipes_path)
    if any(
        not recipe.proposal_source.startswith("rules:source-timeline-") for recipe in rule_recipes
    ):
        raise ValueError("--rule-recipes contains a non-rule source-timeline arm")
    if any(not recipe.proposal_source.startswith("llm-source-timeline:") for recipe in llm_recipes):
        raise ValueError("--llm-recipes contains a non-LLM source-timeline arm")
    if any(not str(recipe.proposal_metadata.get("task_sha256") or "") for recipe in rule_recipes):
        raise ValueError("--rule-recipes is not bound to frozen candidate tasks")
    if any(not str(recipe.proposal_metadata.get("task_sha256") or "") for recipe in llm_recipes):
        raise ValueError("--llm-recipes is not bound to frozen candidate tasks")
    comparison = compare_recipe_sets(rule_recipes, llm_recipes)
    if comparison["pass"] is not True:
        raise ValueError(f"Rule/LLM recipe plans are not matched: {comparison['checks']}")
    rule_ids = [recipe.recipe_id for recipe in rule_recipes]
    llm_ids = [recipe.recipe_id for recipe in llm_recipes]
    if len(rule_ids) != len(set(rule_ids)) or rule_ids != llm_ids:
        raise ValueError("Rule/LLM recipe IDs or ordering differ")

    rule_recipe_hash = file_sha256(rule_recipes_path)
    llm_recipe_hash = file_sha256(llm_recipes_path)
    _require_quality_report(
        rule_quality_report_path,
        rule_manifest_path,
        arm="rule",
    )
    _require_quality_report(
        llm_quality_report_path,
        llm_manifest_path,
        arm="llm",
    )
    rule_entries = _entry_map(
        read_manifest(rule_manifest_path),
        rule_recipes,
        arm="rule",
        recipe_plan_sha256=rule_recipe_hash,
    )
    llm_entries = _entry_map(
        read_manifest(llm_manifest_path),
        llm_recipes,
        arm="llm",
        recipe_plan_sha256=llm_recipe_hash,
    )

    package_root = Path(package_dir).expanduser().resolve()
    package_root.mkdir(parents=True, exist_ok=True)
    selected = _select_stratified(rule_recipes, sample_count, seed)
    private_salt = blinding_salt or secrets.token_hex(32)
    if len(private_salt) < 16:
        raise ValueError("blinding_salt must contain at least 16 characters")
    swap_order = sorted(
        selected,
        key=lambda recipe: _rank(f"{private_salt}:swap", recipe.recipe_id),
    )
    llm_as_a = {recipe.recipe_id for recipe in swap_order[: (len(swap_order) + 1) // 2]}
    rule_root = Path(rule_audio_base).expanduser().resolve()
    llm_root = Path(llm_audio_base).expanduser().resolve()
    rows: list[dict[str, str]] = []
    assignments: list[dict[str, str]] = []
    package_hashes: dict[str, str] = {}
    for index, recipe in enumerate(selected, 1):
        task_id = f"PMR{index:04d}"
        llm_is_a = recipe.recipe_id in llm_as_a
        arm_a = "llm" if llm_is_a else "rule"
        arm_b = "rule" if llm_is_a else "llm"
        entries = {"rule": rule_entries[recipe.recipe_id], "llm": llm_entries[recipe.recipe_id]}
        roots = {"rule": rule_root, "llm": llm_root}
        a_audio, a_stems, a_hashes = _materialize_side(
            entry=entries[arm_a],
            audio_root=roots[arm_a],
            package_root=package_root,
            task_id=task_id,
            side="a",
        )
        b_audio, b_stems, b_hashes = _materialize_side(
            entry=entries[arm_b],
            audio_root=roots[arm_b],
            package_root=package_root,
            task_id=task_id,
            side="b",
        )
        package_hashes.update(a_hashes)
        package_hashes.update(b_hashes)
        rows.append(
            {
                "review_id": "",
                "task_id": task_id,
                "recipe_id": recipe.recipe_id,
                "template": recipe.template,
                "duration_sec": f"{float(recipe.scene_duration_sec or 0.0):.1f}",
                "audio_a_path": a_audio,
                "a_stem_paths_json": a_stems,
                "a_expected_events_json": _event_json(entries[arm_a]),
                "audio_b_path": b_audio,
                "b_stem_paths_json": b_stems,
                "b_expected_events_json": _event_json(entries[arm_b]),
                **{field: "" for field in REVIEW_FIELDS},
            }
        )
        assignments.append(
            {
                "task_id": task_id,
                "recipe_id": recipe.recipe_id,
                "arm_a": arm_a,
                "arm_b": arm_b,
                "rule_scene_id": str(rule_entries[recipe.recipe_id].scene["scene_id"]),
                "llm_scene_id": str(llm_entries[recipe.recipe_id].scene["scene_id"]),
            }
        )

    tasks_hash = _tasks_sha256(rows)
    artifact_hashes = {
        "rule_recipes": file_sha256(rule_recipes_path),
        "llm_recipes": file_sha256(llm_recipes_path),
        "rule_manifest": file_sha256(rule_manifest_path),
        "llm_manifest": file_sha256(llm_manifest_path),
        "rule_quality_report": file_sha256(rule_quality_report_path),
        "llm_quality_report": file_sha256(llm_quality_report_path),
    }
    review_id = _review_id(
        seed=seed,
        sample_count=sample_count,
        tasks_sha256=tasks_hash,
        artifacts=artifact_hashes,
        package_files=package_hashes,
    )
    for row in rows:
        row["review_id"] = review_id
    metadata = {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "review_id": review_id,
        "seed": seed,
        "blinding_salt_sha256": _canonical_hash(private_salt),
        "n_tasks": len(rows),
        "task_ids": [row["task_id"] for row in rows],
        "recipe_ids": [row["recipe_id"] for row in rows],
        "tasks_sha256": tasks_hash,
        "by_template": dict(sorted(Counter(row["template"] for row in rows).items())),
        "package_dir": str(package_root),
        "package_files": dict(sorted(package_hashes.items())),
        "artifacts": artifact_hashes,
        "instructions": {
            "rating_scale": "1=clearly poor/unsupported, 3=partly acceptable, 5=fully convincing",
            "blindness": "reviewers receive the CSV and package directory, never the key",
            "stems": "use anonymous stems to verify audibility, caption support and timestamps",
            "preference": "a, b, or tie after evaluating both sides",
        },
    }
    key = {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "review_id": review_id,
        "blinding_salt": private_salt,
        "assignments": assignments,
    }
    return rows, metadata, key


def write_mixture_review(
    rows: list[dict[str, str]],
    metadata: dict[str, Any],
    key: dict[str, Any],
    *,
    csv_path: str | Path,
    metadata_path: str | Path,
    key_path: str | Path,
) -> None:
    output_csv = Path(csv_path)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    output_key = Path(key_path)
    output_key.parent.mkdir(parents=True, exist_ok=True)
    output_key.write_text(json.dumps(key, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    bound_metadata = dict(metadata)
    bound_metadata["key_sha256"] = file_sha256(output_key)
    output_metadata = Path(metadata_path)
    output_metadata.parent.mkdir(parents=True, exist_ok=True)
    output_metadata.write_text(
        json.dumps(bound_metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _read_review_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(CSV_FIELDS):
            raise ValueError("mixture review CSV columns changed")
        return [dict(row) for row in reader]


def _rating(row: dict[str, str], field: str) -> float:
    try:
        value = int(row.get(field, "").strip())
    except ValueError as exc:
        raise ValueError(f"{row['task_id']}: {field} must be an integer 1..5") from exc
    if value not in range(1, 6):
        raise ValueError(f"{row['task_id']}: {field} must be in 1..5")
    return float(value)


def _count(row: dict[str, str], field: str) -> int:
    try:
        value = int(row.get(field, "").strip())
    except ValueError as exc:
        raise ValueError(f"{row['task_id']}: {field} must be a non-negative integer") from exc
    if value < 0:
        raise ValueError(f"{row['task_id']}: {field} must be non-negative")
    return value


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 6) if values else None


def _two_sided_sign_p(wins: int, losses: int) -> float | None:
    n = wins + losses
    if n == 0:
        return None
    tail = min(wins, losses)
    probability = sum(math.comb(n, index) for index in range(tail + 1)) / (2**n)
    return round(min(1.0, 2 * probability), 8)


def summarize_mixture_review(
    *,
    review_csv_path: str | Path | list[str | Path],
    metadata_path: str | Path,
    key_path: str | Path,
    min_plausibility_delta: float = 0.25,
    max_safety_regression: float = 0.10,
    max_sign_p: float = 0.05,
) -> dict[str, Any]:
    """Validate completed sheets, unblind assignments, and compute a scale gate."""

    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    key = json.loads(Path(key_path).read_text(encoding="utf-8"))
    if metadata.get("schema_version") != REVIEW_SCHEMA_VERSION:
        raise ValueError("unsupported mixture review metadata schema")
    if key.get("schema_version") != REVIEW_SCHEMA_VERSION:
        raise ValueError("unsupported mixture review key schema")
    if file_sha256(key_path) != metadata.get("key_sha256"):
        raise ValueError("mixture review key changed after task preparation")
    if key.get("review_id") != metadata.get("review_id"):
        raise ValueError("mixture review key and metadata IDs differ")
    expected_review_id = _review_id(
        seed=str(metadata.get("seed") or ""),
        sample_count=int(metadata.get("n_tasks") or 0),
        tasks_sha256=str(metadata.get("tasks_sha256") or ""),
        artifacts=metadata.get("artifacts", {}),
        package_files=metadata.get("package_files", {}),
    )
    if expected_review_id != metadata.get("review_id"):
        raise ValueError("mixture review metadata changed after task preparation")
    if _canonical_hash(str(key.get("blinding_salt") or "")) != metadata.get("blinding_salt_sha256"):
        raise ValueError("mixture review blinding salt does not match metadata")
    package_root = Path(str(metadata["package_dir"])).expanduser().resolve()
    for relative, expected_hash in metadata.get("package_files", {}).items():
        artifact = (package_root / relative).resolve()
        try:
            artifact.relative_to(package_root)
        except ValueError as exc:
            raise ValueError(f"review package path escapes package root: {relative}") from exc
        if not artifact.is_file() or file_sha256(artifact) != expected_hash:
            raise ValueError(f"review package artifact is missing or changed: {relative}")

    paths = (
        [Path(path) for path in review_csv_path]
        if isinstance(review_csv_path, list)
        else [Path(review_csv_path)]
    )
    if not paths:
        raise ValueError("at least one completed mixture review CSV is required")
    hashes = [file_sha256(path) for path in paths]
    if len(hashes) != len(set(hashes)):
        raise ValueError("duplicate completed mixture review files were supplied")
    batches = [_read_review_csv(path) for path in paths]
    for rows in batches:
        if len(rows) != metadata.get("n_tasks"):
            raise ValueError("mixture review row count changed")
        if [row["task_id"] for row in rows] != metadata.get("task_ids"):
            raise ValueError("mixture review task IDs or ordering changed")
        if [row["recipe_id"] for row in rows] != metadata.get("recipe_ids"):
            raise ValueError("mixture review recipe IDs or ordering changed")
        if any(row["review_id"] != metadata.get("review_id") for row in rows):
            raise ValueError("mixture review ID changed")
        if _tasks_sha256(rows) != metadata.get("tasks_sha256"):
            raise ValueError("immutable mixture review fields changed")
    if len(batches) > 1:
        reviewer_sets = [
            {row["reviewer"].strip() for row in rows if row["reviewer"].strip()} for rows in batches
        ]
        if any(len(names) != 1 for names in reviewer_sets):
            raise ValueError("each completed review CSV must contain exactly one reviewer")
        reviewers = [next(iter(names)) for names in reviewer_sets]
        if len(reviewers) != len(set(reviewers)):
            raise ValueError("completed review CSVs must come from distinct reviewers")

    assignment_rows = key.get("assignments", [])
    assignments = {row["task_id"]: row for row in assignment_rows}
    if len(assignments) != len(assignment_rows):
        raise ValueError("mixture review key has duplicate task assignments")
    if set(assignments) != set(metadata.get("task_ids", [])):
        raise ValueError("mixture review key assignments do not match tasks")
    recipe_by_task = dict(zip(metadata["task_ids"], metadata["recipe_ids"], strict=True))
    for task_id, assignment in assignments.items():
        if assignment.get("recipe_id") != recipe_by_task[task_id]:
            raise ValueError("mixture review key recipe assignment changed")
        if {assignment.get("arm_a"), assignment.get("arm_b")} != {"rule", "llm"}:
            raise ValueError("mixture review key arm assignment is invalid")
    values: dict[str, defaultdict[str, list[float]]] = {
        "rule": defaultdict(list),
        "llm": defaultdict(list),
    }
    counts = {"rule": Counter(), "llm": Counter()}
    paired_deltas: defaultdict[str, list[float]] = defaultdict(list)
    preferences: defaultdict[str, list[str]] = defaultdict(list)
    reviewer_names: set[str] = set()
    for row in [item for batch in batches for item in batch]:
        reviewer = row.get("reviewer", "").strip()
        reviewed_at = row.get("reviewed_at_utc", "").strip()
        if not reviewer or not reviewed_at:
            raise ValueError(f"{row['task_id']}: reviewer and reviewed_at_utc are required")
        reviewer_names.add(reviewer)
        side_scores: dict[str, dict[str, float]] = {}
        side_counts: dict[str, dict[str, int]] = {}
        for side in ("a", "b"):
            side_scores[side] = {name: _rating(row, f"{side}_{name}_1_5") for name in RATING_NAMES}
            side_counts[side] = {
                "inaudible_sources": _count(row, f"{side}_inaudible_sources_count"),
                "unsupported_labels": _count(row, f"{side}_unsupported_labels_count"),
            }
        assignment = assignments[row["task_id"]]
        scores_by_arm = {
            assignment["arm_a"]: side_scores["a"],
            assignment["arm_b"]: side_scores["b"],
        }
        counts_by_arm = {
            assignment["arm_a"]: side_counts["a"],
            assignment["arm_b"]: side_counts["b"],
        }
        for arm in ("rule", "llm"):
            for name, value in scores_by_arm[arm].items():
                values[arm][name].append(value)
            counts[arm].update(counts_by_arm[arm])
        for name in RATING_NAMES:
            paired_deltas[name].append(scores_by_arm["llm"][name] - scores_by_arm["rule"][name])
        preference = row.get("preference_a_b_tie", "").strip().lower()
        if preference not in {"a", "b", "tie"}:
            raise ValueError(f"{row['task_id']}: preference must be a, b, or tie")
        preferences[row["task_id"]].append(
            "tie" if preference == "tie" else str(assignment[f"arm_{preference}"])
        )

    arm_summary = {
        arm: {
            "n_judgments": sum(len(items) for items in values[arm].values()) // len(RATING_NAMES),
            **{f"mean_{name}": _mean(items) for name, items in values[arm].items()},
            "total_inaudible_sources": counts[arm]["inaudible_sources"],
            "total_unsupported_labels": counts[arm]["unsupported_labels"],
        }
        for arm in ("rule", "llm")
    }
    deltas = {f"mean_{name}": _mean(items) for name, items in paired_deltas.items()}
    consensus = Counter()
    agreement_pairs = 0
    agreement_matches = 0
    for labels in preferences.values():
        label_counts = Counter(labels)
        if label_counts["llm"] > label_counts["rule"]:
            consensus["llm"] += 1
        elif label_counts["rule"] > label_counts["llm"]:
            consensus["rule"] += 1
        else:
            consensus["tie"] += 1
        for left_index, left in enumerate(labels):
            for right in labels[left_index + 1 :]:
                agreement_pairs += 1
                agreement_matches += int(left == right)
    sign_p = _two_sided_sign_p(consensus["llm"], consensus["rule"])
    safety_metrics = (
        "all_sources_audible",
        "naturalness",
        "caption_support",
        "timestamp_alignment",
    )
    safety_no_regression = all(
        float(deltas[f"mean_{name}"] or 0.0) >= -max_safety_regression for name in safety_metrics
    )
    plausibility_improved = all(
        float(deltas[f"mean_{name}"] or 0.0) >= min_plausibility_delta
        for name in ("scene_plausibility", "temporal_plausibility")
    )
    evidence_errors_safe = (
        arm_summary["llm"]["total_inaudible_sources"]
        <= arm_summary["rule"]["total_inaudible_sources"]
        and arm_summary["llm"]["total_unsupported_labels"]
        <= arm_summary["rule"]["total_unsupported_labels"]
    )
    preference_significant = (
        consensus["llm"] > consensus["rule"] and sign_p is not None and sign_p <= max_sign_p
    )
    go_for_scale = (
        safety_no_regression
        and plausibility_improved
        and evidence_errors_safe
        and preference_significant
    )
    return {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "pass": True,
        "status": "complete",
        "review_id": metadata["review_id"],
        "n_tasks": metadata["n_tasks"],
        "n_reviewers": len(reviewer_names),
        "n_judgments": sum(len(batch) for batch in batches),
        "arms": arm_summary,
        "paired_delta_llm_minus_rule": deltas,
        "preference_sample_consensus": {
            "llm": consensus["llm"],
            "rule": consensus["rule"],
            "tie": consensus["tie"],
            "two_sided_sign_test_p": sign_p,
            "pairwise_reviewer_agreement": (
                round(agreement_matches / agreement_pairs, 6) if agreement_pairs else None
            ),
            "n_agreement_pairs": agreement_pairs,
        },
        "go_for_scale": go_for_scale,
        "decision_thresholds": {
            "min_plausibility_delta": min_plausibility_delta,
            "max_safety_regression": max_safety_regression,
            "max_sign_p": max_sign_p,
        },
        "decision_checks": {
            "audibility_naturalness_caption_timing_no_regression": safety_no_regression,
            "scene_and_temporal_plausibility_improved": plausibility_improved,
            "inaudible_and_unsupported_counts_not_increased": evidence_errors_safe,
            "llm_preference_significant": preference_significant,
        },
        "artifacts": {
            "review_csvs": [
                {"path": str(path.resolve()), "sha256": digest}
                for path, digest in zip(paths, hashes, strict=True)
            ],
            "metadata_sha256": file_sha256(metadata_path),
            "key_sha256": file_sha256(key_path),
        },
    }


__all__ = [
    "CSV_FIELDS",
    "prepare_mixture_review",
    "summarize_mixture_review",
    "write_mixture_review",
]

"""Immutable versioned Golden Set snapshot persistence and promotion workflow."""

from __future__ import annotations

import json
from pathlib import Path

from beliefaudit.schemas import CandidateExpansionEvent, GoldenItem, GoldenSnapshot, utc_now
from beliefaudit.security import GoldenIntegrityError, ensure_within_base


def save_golden_snapshot(
    snapshot: GoldenSnapshot,
    output_path: Path | str,
    *,
    base_dir: Path | str | None = None,
    overwrite: bool = False,
) -> Path:
    """Persist an immutable GoldenSnapshot, verifying digest and preventing silent overwrite."""
    if not snapshot.verify_digest():
        raise GoldenIntegrityError("GoldenSnapshot digest does not match its payload.")
    target = (
        ensure_within_base(output_path, base_dir)
        if base_dir is not None
        else Path(output_path).resolve()
    )
    if target.exists() and not overwrite:
        raise GoldenIntegrityError(
            f"Refusing to overwrite immutable GoldenSnapshot at {target} (pass overwrite=True)."
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(snapshot.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return target


def load_golden_snapshot(
    input_path: Path | str,
    *,
    base_dir: Path | str | None = None,
    verify: bool = True,
) -> GoldenSnapshot:
    """Load a GoldenSnapshot JSON file and optionally verify its SHA-256 digest."""
    source = (
        ensure_within_base(input_path, base_dir)
        if base_dir is not None
        else Path(input_path).resolve()
    )
    if not source.is_file():
        raise GoldenIntegrityError(f"GoldenSnapshot file not found: {source}")
    snapshot = GoldenSnapshot.model_validate_json(source.read_text(encoding="utf-8"))
    if verify and not snapshot.verify_digest():
        raise GoldenIntegrityError(
            f"GoldenSnapshot digest mismatch for {source}: snapshot may be corrupted or modified."
        )
    return snapshot


def save_pending_candidates(
    candidates: list[CandidateExpansionEvent],
    output_path: Path | str,
    *,
    base_dir: Path | str | None = None,
) -> Path:
    """Write flagged CandidateExpansionEvent records to a JSON file."""
    target = (
        ensure_within_base(output_path, base_dir)
        if base_dir is not None
        else Path(output_path).resolve()
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = [event.model_dump(mode="json") for event in candidates]
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return target


def load_pending_candidates(
    input_path: Path | str,
    *,
    base_dir: Path | str | None = None,
) -> list[CandidateExpansionEvent]:
    """Load CandidateExpansionEvent records from a JSON file."""
    source = (
        ensure_within_base(input_path, base_dir)
        if base_dir is not None
        else Path(input_path).resolve()
    )
    if not source.is_file():
        raise GoldenIntegrityError(f"Pending candidates file not found: {source}")
    raw = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise GoldenIntegrityError(f"Expected JSON list in {source}.")
    return [CandidateExpansionEvent.model_validate(entry) for entry in raw]


def promote_golden_snapshot(
    current_snapshot: GoldenSnapshot,
    candidates: list[CandidateExpansionEvent],
    next_version: str,
    *,
    new_items: list[GoldenItem] | None = None,
) -> GoldenSnapshot:
    """Create a new immutable GoldenSnapshot by promoting candidate discoveries without mutating *current_snapshot*."""
    if not current_snapshot.verify_digest():
        raise GoldenIntegrityError("Cannot promote from a snapshot with an invalid digest.")
    if next_version.strip() == current_snapshot.golden_version:
        raise GoldenIntegrityError(
            f"next_version {next_version!r} must differ from current version "
            f"{current_snapshot.golden_version!r}."
        )

    item_map: dict[str, GoldenItem] = {
        item.item_id: item.model_copy(update={"golden_version": next_version}, deep=True)
        for item in current_snapshot.items
    }

    for event in candidates:
        if event.suggested_euphemisms and event.golden_item_id in item_map:
            existing = item_map[event.golden_item_id]
            merged = list(existing.disallowed_euphemisms)
            for term in event.suggested_euphemisms:
                cleaned = term.strip()
                if cleaned and cleaned.lower() not in {t.lower() for t in merged}:
                    merged.append(cleaned)
            item_map[event.golden_item_id] = existing.model_copy(
                update={"disallowed_euphemisms": merged}
            )
        if event.suggested_item is not None and event.suggested_item.item_id not in item_map:
            item_map[event.suggested_item.item_id] = event.suggested_item.model_copy(
                update={"golden_version": next_version}, deep=True
            )

    for extra_item in new_items or []:
        if extra_item.item_id not in item_map:
            item_map[extra_item.item_id] = extra_item.model_copy(
                update={"golden_version": next_version}, deep=True
            )

    promoted = GoldenSnapshot(
        golden_version=next_version,
        created_at=utc_now(),
        parent_version=current_snapshot.golden_version,
        content_sha256="",
        items=list(item_map.values()),
        control_items=[ctrl.model_copy(deep=True) for ctrl in current_snapshot.control_items],
    )
    return promoted

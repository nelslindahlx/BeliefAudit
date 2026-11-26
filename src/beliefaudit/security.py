"""Security utilities for path containment, endpoint validation, and identifier hygiene."""

from __future__ import annotations

import ipaddress
import re
from pathlib import Path
from urllib.parse import urlparse

IDENTIFIER_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._:-]{0,127}$")
LAYER_ID_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$")
LOOPBACK_HOSTNAMES = frozenset({"localhost", "127.0.0.1", "::1"})


class BeliefAuditError(Exception):
    """Base domain exception for BeliefAudit."""


class BeliefAuditSecurityError(BeliefAuditError):
    """Raised when a path containment or endpoint URL policy check fails."""


class GoldenIntegrityError(BeliefAuditError):
    """Raised when a Golden Set snapshot fails digest verification or immutability rules."""


class RunnerError(BeliefAuditError):
    """Raised when a model runner encounters an HTTP, timeout, or payload error."""


def ensure_within_base(path: Path | str, base_dir: Path | str) -> Path:
    """Resolve *path* relative to *base_dir* and reject directory/symlink escapes."""
    base_resolved = Path(base_dir).resolve()
    raw = Path(path)
    candidate = raw.resolve() if raw.is_absolute() else (base_resolved / raw).resolve()
    try:
        candidate.relative_to(base_resolved)
    except ValueError as exc:
        raise BeliefAuditSecurityError(
            f"Path {path!r} resolves outside base directory {base_resolved}."
        ) from exc
    return candidate


def _is_loopback_host(hostname: str) -> bool:
    normalized = hostname.strip().lower().rstrip(".")
    if normalized in LOOPBACK_HOSTNAMES:
        return True
    try:
        ip_obj = ipaddress.ip_address(normalized)
    except ValueError:
        return False
    return ip_obj.is_loopback


def validate_endpoint_url(url: str, *, allow_remote: bool = False) -> str:
    """Validate an OpenAI-compatible HTTP endpoint URL (loopback-only unless opted in)."""
    cleaned = url.strip().rstrip("/")
    parsed = urlparse(cleaned)
    if parsed.scheme not in {"http", "https"}:
        raise BeliefAuditSecurityError(f"Endpoint URL must use http or https scheme (got {url!r}).")
    if not parsed.hostname:
        raise BeliefAuditSecurityError(f"Endpoint URL is missing a hostname (got {url!r}).")
    if not allow_remote and not _is_loopback_host(parsed.hostname):
        raise BeliefAuditSecurityError(
            f"Non-loopback endpoint host {parsed.hostname!r} blocked by local-first policy "
            "(pass allow_remote=True or --allow-remote to opt in)."
        )
    return cleaned


def validate_identifier(value: str, field_name: str = "identifier") -> str:
    """Validate safe identifier format for IDs and filenames."""
    cleaned = value.strip()
    if not IDENTIFIER_PATTERN.fullmatch(cleaned):
        raise ValueError(
            f"{field_name} must match ^[a-zA-Z0-9][a-zA-Z0-9._:-]{{0,127}}$ (got {value!r})."
        )
    return cleaned


def sanitize_layer_id(raw_id: str) -> str:
    """Normalize an arbitrary ID into a KnowledgeReduce-compatible layer_id."""
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", raw_id.strip()).strip("-._")
    if not cleaned:
        cleaned = "layer"
    if not cleaned[0].isalnum():
        cleaned = f"l-{cleaned}"
    cleaned = cleaned[:128]
    if not LAYER_ID_PATTERN.fullmatch(cleaned):
        raise ValueError(f"Could not sanitize {raw_id!r} into a valid layer_id.")
    return cleaned

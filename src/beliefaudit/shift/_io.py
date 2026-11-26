"""Shared atomic file-write helper for BeliefShift artifacts."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def write_text_atomic(content: str, output_path: Path) -> None:
    """Write *content* as UTF-8 atomically via a same-directory temp file and os.replace."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=str(output_path.parent),
            delete=False,
            encoding="utf-8",
            prefix=".shift_",
            suffix=".tmp",
        ) as fout:
            tmp_path = fout.name
            fout.write(content)
        os.replace(tmp_path, output_path)
    except Exception:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise

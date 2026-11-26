"""Offline general capability replay bank disjoint from ControlItems."""

from __future__ import annotations

import json
import random
from pathlib import Path

from beliefaudit.probes.generator import (
    CODE_SYSTEM_PROMPT,
    JSON_SYSTEM_PROMPT,
    PROSE_SYSTEM_PROMPT,
)
from beliefaudit.security import BeliefAuditError
from beliefaudit.shift.schemas import ChatMessage, SftTrainingRecord


class ReplayBankError(BeliefAuditError):
    """Raised when a custom replay dataset is missing or invalid."""


_RAW_CURATED_REPLAY: tuple[tuple[str, str, str], ...] = (
    (
        PROSE_SYSTEM_PROMPT,
        "What is the time complexity of looking up a key in a Python dictionary on average?",
        "Average-case key lookup in a Python dictionary is O(1) due to hash table indexing.",
    ),
    (
        PROSE_SYSTEM_PROMPT,
        "Explain the difference between HTTP status codes 401 and 403.",
        "HTTP 401 Unauthorized indicates missing or invalid authentication credentials, whereas 403 Forbidden means the server understood the request but denies authorization.",
    ),
    (
        PROSE_SYSTEM_PROMPT,
        "Why do relational databases use B-tree indexes?",
        "B-tree indexes keep keys sorted in balanced pages, enabling O(log n) point lookups and efficient range scans.",
    ),
    (
        PROSE_SYSTEM_PROMPT,
        "What problem does idempotency solve in REST APIs?",
        "Idempotency ensures that retrying the same request multiple times produces the same server state as executing it once.",
    ),
    (
        PROSE_SYSTEM_PROMPT,
        "Summarize how TLS handshakes protect data in transit.",
        "A TLS handshake authenticates the server via certificates and negotiates ephemeral symmetric session keys to encrypt subsequent traffic.",
    ),
    (
        PROSE_SYSTEM_PROMPT,
        "What distinguishes a process from a thread in modern operating systems?",
        "Processes have isolated virtual address spaces, while threads within the same process share memory and file descriptors.",
    ),
    (
        PROSE_SYSTEM_PROMPT,
        "Why should passwords be stored with Argon2 or bcrypt instead of plain SHA-256?",
        "Argon2 and bcrypt are intentionally slow, salted, and memory-hard, resisting brute-force GPU cracking far better than fast hashes like SHA-256.",
    ),
    (
        PROSE_SYSTEM_PROMPT,
        "What does ACID stand for in database transactions?",
        "ACID stands for Atomicity, Consistency, Isolation, and Durability.",
    ),
    (
        PROSE_SYSTEM_PROMPT,
        "How does merge sort achieve O(n log n) worst-case runtime?",
        "Merge sort recursively halves the array in log n levels and performs O(n) linear merging at each level.",
    ),
    (
        PROSE_SYSTEM_PROMPT,
        "Why is UTC preferred for backend event timestamps?",
        "Storing timestamps in UTC avoids daylight-saving ambiguities and timezone drift across distributed servers.",
    ),
    (
        CODE_SYSTEM_PROMPT,
        "Write a Python function `clamp(value: float, low: float, high: float) -> float`.",
        "```python\ndef clamp(value: float, low: float, high: float) -> float:\n    return max(low, min(high, value))\n```",
    ),
    (
        CODE_SYSTEM_PROMPT,
        "Write a Python function `chunk_list(items: list[int], size: int) -> list[list[int]]`.",
        "```python\ndef chunk_list(items: list[int], size: int) -> list[list[int]]:\n    if size < 1:\n        raise ValueError('size must be >= 1')\n    return [items[i : i + size] for i in range(0, len(items), size)]\n```",
    ),
    (
        CODE_SYSTEM_PROMPT,
        "Write a Python function `invert_dict(mapping: dict[str, int]) -> dict[int, str]`.",
        "```python\ndef invert_dict(mapping: dict[str, int]) -> dict[int, str]:\n    return {v: k for k, v in mapping.items()}\n```",
    ),
    (
        CODE_SYSTEM_PROMPT,
        "Write a Python function `count_vowels(text: str) -> int`.",
        "```python\ndef count_vowels(text: str) -> int:\n    return sum(1 for ch in text.lower() if ch in 'aeiou')\n```",
    ),
    (
        CODE_SYSTEM_PROMPT,
        "Write a Python function `flatten_once(nested: list[list[str]]) -> list[str]`.",
        "```python\ndef flatten_once(nested: list[list[str]]) -> list[str]:\n    return [item for row in nested for item in row]\n```",
    ),
    (
        CODE_SYSTEM_PROMPT,
        "Write a Python function `safe_divide(a: float, b: float) -> float | None` returning None on zero divisor.",
        "```python\ndef safe_divide(a: float, b: float) -> float | None:\n    return None if b == 0 else a / b\n```",
    ),
    (
        CODE_SYSTEM_PROMPT,
        "Write a Python function `unique_sorted(nums: list[int]) -> list[int]`.",
        "```python\ndef unique_sorted(nums: list[int]) -> list[int]:\n    return sorted(set(nums))\n```",
    ),
    (
        CODE_SYSTEM_PROMPT,
        "Write a Python function `is_anagram(left: str, right: str) -> bool`.",
        "```python\ndef is_anagram(left: str, right: str) -> bool:\n    return sorted(left.lower()) == sorted(right.lower())\n```",
    ),
    (
        CODE_SYSTEM_PROMPT,
        "Write a Python function `gcd_euclid(a: int, b: int) -> int`.",
        "```python\ndef gcd_euclid(a: int, b: int) -> int:\n    while b:\n        a, b = b, a % b\n    return abs(a)\n```",
    ),
    (
        CODE_SYSTEM_PROMPT,
        "Write a Python function `merge_counts(a: dict[str, int], b: dict[str, int]) -> dict[str, int]`.",
        "```python\ndef merge_counts(a: dict[str, int], b: dict[str, int]) -> dict[str, int]:\n    out = dict(a)\n    for k, v in b.items():\n        out[k] = out.get(k, 0) + v\n    return out\n```",
    ),
    (
        JSON_SYSTEM_PROMPT,
        "Extract JSON with keys `service`, `port`, `tls` from: 'Service gateway listens on port 8443 with TLS enabled.'",
        '{"service": "gateway", "port": 8443, "tls": true}',
    ),
    (
        JSON_SYSTEM_PROMPT,
        "Extract JSON with keys `language`, `version`, `typed` from: 'Python 3.11 supports strict static type checking.'",
        '{"language": "Python", "version": "3.11", "typed": true}',
    ),
    (
        JSON_SYSTEM_PROMPT,
        "Extract JSON with keys `table`, `primary_key`, `indexed` from: 'Table orders uses order_id as primary key with indexing.'",
        '{"table": "orders", "primary_key": "order_id", "indexed": true}',
    ),
    (
        JSON_SYSTEM_PROMPT,
        "Extract JSON with keys `metric`, `unit`, `threshold` from: 'Latency measured in ms must stay under 50.'",
        '{"metric": "latency", "unit": "ms", "threshold": 50}',
    ),
    (
        JSON_SYSTEM_PROMPT,
        "Extract JSON with keys `algorithm`, `complexity`, `stable` from: 'Timsort runs in O(n log n) worst case and is stable.'",
        '{"algorithm": "Timsort", "complexity": "O(n log n)", "stable": true}',
    ),
    (
        PROSE_SYSTEM_PROMPT,
        "What is the freezing point of water at standard atmospheric pressure in Celsius?",
        "At standard atmospheric pressure (1 atm), pure water freezes at 0 degrees Celsius.",
    ),
    (
        PROSE_SYSTEM_PROMPT,
        "Name the chemical symbol and atomic number of carbon.",
        "Carbon has the chemical symbol C and atomic number 6.",
    ),
    (
        PROSE_SYSTEM_PROMPT,
        "Explain what DNS resolution accomplishes before an HTTP connection opens.",
        "DNS resolution translates a human-readable domain name into its corresponding IPv4 or IPv6 network address.",
    ),
    (
        PROSE_SYSTEM_PROMPT,
        "Why does JSON forbid trailing commas after the last element?",
        "RFC 8259 specifies strict grammar without trailing commas so parsers across languages remain unambiguous and deterministic.",
    ),
    (
        PROSE_SYSTEM_PROMPT,
        "What is the purpose of a SQL FOREIGN KEY constraint?",
        "A FOREIGN KEY constraint enforces referential integrity by requiring child column values to match existing parent rows.",
    ),
    (
        CODE_SYSTEM_PROMPT,
        "Write a Python function `mean_or_zero(values: list[float]) -> float`.",
        "```python\ndef mean_or_zero(values: list[float]) -> float:\n    return sum(values) / len(values) if values else 0.0\n```",
    ),
    (
        CODE_SYSTEM_PROMPT,
        "Write a Python function `strip_prefix_safe(text: str, prefix: str) -> str`.",
        "```python\ndef strip_prefix_safe(text: str, prefix: str) -> str:\n    return text[len(prefix):] if text.startswith(prefix) else text\n```",
    ),
    (
        CODE_SYSTEM_PROMPT,
        "Write a Python function `dot_product(xs: list[float], ys: list[float]) -> float`.",
        "```python\ndef dot_product(xs: list[float], ys: list[float]) -> float:\n    return sum(x * y for x, y in zip(xs, ys, strict=True))\n```",
    ),
    (
        CODE_SYSTEM_PROMPT,
        "Write a Python function `is_leap_year(year: int) -> bool`.",
        "```python\ndef is_leap_year(year: int) -> bool:\n    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)\n```",
    ),
    (
        CODE_SYSTEM_PROMPT,
        "Write a Python function `word_frequencies(sentence: str) -> dict[str, int]`.",
        "```python\ndef word_frequencies(sentence: str) -> dict[str, int]:\n    counts: dict[str, int] = {}\n    for tok in sentence.lower().split():\n        counts[tok] = counts.get(tok, 0) + 1\n    return counts\n```",
    ),
    (
        JSON_SYSTEM_PROMPT,
        "Extract JSON with keys `host`, `port`, `scheme` from: 'Local inference server runs at http://localhost:11434.'",
        '{"host": "localhost", "port": 11434, "scheme": "http"}',
    ),
    (
        JSON_SYSTEM_PROMPT,
        "Extract JSON with keys `package`, `license`, `python_min` from: 'Package beliefaudit uses MIT license and Python 3.11+.'",
        '{"package": "beliefaudit", "license": "MIT", "python_min": "3.11"}',
    ),
    (
        JSON_SYSTEM_PROMPT,
        "Extract JSON with keys `layer`, `format`, `compression` from: 'Columnar layer exported in Parquet with zstd compression.'",
        '{"layer": "columnar", "format": "Parquet", "compression": "zstd"}',
    ),
    (
        PROSE_SYSTEM_PROMPT,
        "Why is temperature=0.0 used in deterministic LLM evaluation?",
        "Setting temperature to 0.0 selects greedy decoding, minimizing sampling variance across repeated evaluation runs.",
    ),
    (
        PROSE_SYSTEM_PROMPT,
        "What is the role of a Python virtual environment (`venv`)?",
        "A virtual environment isolates a project's installed package dependencies from the global system Python interpreter.",
    ),
)


def _curated_replay_records() -> list[SftTrainingRecord]:
    return [
        SftTrainingRecord(
            messages=[
                ChatMessage(role="system", content=sys_p),
                ChatMessage(role="user", content=usr_p),
                ChatMessage(role="assistant", content=asst_p),
            ],
            metadata={"view_type": "replay", "is_replay": True, "index": idx},
        )
        for idx, (sys_p, usr_p, asst_p) in enumerate(_RAW_CURATED_REPLAY)
    ]


def _load_custom_replay(path: Path) -> list[SftTrainingRecord]:
    if not path.is_file():
        raise ReplayBankError(f"Custom replay dataset not found: {path}")
    records: list[SftTrainingRecord] = []
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            raw = json.loads(stripped)
            rec = SftTrainingRecord.model_validate(raw)
        except Exception as exc:
            raise ReplayBankError(f"Invalid SFT record at line {idx} in {path}: {exc}") from exc
        meta = dict(rec.metadata)
        meta["is_replay"] = True
        records.append(SftTrainingRecord(messages=rec.messages, metadata=meta))
    return records


def generate_replay_records(
    min_count: int,
    *,
    custom_replay_path: Path | None = None,
    seed: int = 42,
) -> list[SftTrainingRecord]:
    """Return at least *min_count* deterministic general capability replay records."""
    if min_count < 0:
        raise ValueError(f"min_count must be >= 0 (got {min_count}).")
    rng = random.Random(seed)  # noqa: S311 - deterministic non-crypto replay shuffle
    custom_pool: list[SftTrainingRecord] = []
    if custom_replay_path is not None:
        custom_pool = _load_custom_replay(custom_replay_path)
        rng.shuffle(custom_pool)
    curated_pool = _curated_replay_records()
    rng.shuffle(curated_pool)
    pool: list[SftTrainingRecord] = [*custom_pool, *curated_pool]

    extra_idx = 0
    while len(pool) < min_count:
        a = (extra_idx * 7 + 3) % 97 + 1
        b = (extra_idx * 11 + 5) % 89 + 1
        pool.append(
            SftTrainingRecord(
                messages=[
                    ChatMessage(role="system", content=JSON_SYSTEM_PROMPT),
                    ChatMessage(
                        role="user",
                        content=f"Compute the sum and product of {a} and {b} as JSON keys `sum` and `product`.",
                    ),
                    ChatMessage(
                        role="assistant",
                        content=json.dumps({"sum": a + b, "product": a * b}),
                    ),
                ],
                metadata={"view_type": "replay_synth", "is_replay": True, "index": extra_idx},
            )
        )
        extra_idx += 1

    return pool[:min_count] if min_count <= len(pool) else pool

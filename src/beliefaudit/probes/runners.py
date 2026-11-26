"""Local-first OpenAI-compatible HTTP runner and deterministic offline mock runner."""

from __future__ import annotations

import json
from types import TracebackType
from typing import Any, Literal, Protocol

import httpx

from beliefaudit.golden.seed_pack import BundledOfflineSeeder
from beliefaudit.schemas import ControlAxis, ControlItem, GoldenItem, GoldenSnapshot
from beliefaudit.security import RunnerError, validate_endpoint_url

MockPersona = Literal[
    "clean",
    "skewed",
    "sycophant",
    "refuser",
    "leaker",
    "low_quality",
]


class BaseModelRunner(Protocol):
    """Minimal chat interface implemented by local HTTP and offline mock runners."""

    model_id: str
    model_revision: str | None
    adapter_id: str | None

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        seed: int = 42,
        max_tokens: int = 512,
    ) -> str: ...


class LocalOpenAICompatRunner:
    """OpenAI-compatible local endpoint client (Ollama, vLLM, llama.cpp, LM Studio)."""

    def __init__(
        self,
        model_id: str,
        base_url: str = "http://localhost:11434/v1",
        *,
        allow_remote: bool = False,
        timeout: float = 60.0,
        api_key: str | None = None,
        model_revision: str | None = None,
        adapter_id: str | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.model_id = model_id
        self.model_revision = model_revision
        self.adapter_id = adapter_id
        self.base_url = validate_endpoint_url(base_url, allow_remote=allow_remote)
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout,
            headers=headers,
            transport=transport,
        )

    def __enter__(self) -> LocalOpenAICompatRunner:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying HTTPX client."""
        self._client.close()

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        seed: int = 42,
        max_tokens: int = 512,
    ) -> str:
        """Execute a deterministic chat completion call against /chat/completions."""
        payload: dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
            "temperature": temperature,
            "seed": seed,
            "max_tokens": max_tokens,
        }
        try:
            response = self._client.post("/chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise RunnerError(
                f"Local endpoint error from {self.base_url}/chat/completions: {exc}"
            ) from exc

        try:
            choices = data["choices"]
            if not isinstance(choices, list) or not choices:
                raise KeyError("choices")
            content = choices[0]["message"]["content"]
            if not isinstance(content, str):
                raise TypeError("message.content must be a string")
            return content
        except (KeyError, IndexError, TypeError) as exc:
            raise RunnerError(f"Malformed OpenAI-compatible response payload: {data!r}") from exc


def _render_clean_code(item: GoldenItem) -> str:
    tokens = item.expected_code_tokens
    fn_sig = next((t for t in tokens if t.startswith("def ")), "def handle_record")
    fn_name = fn_sig.replace("def ", "").strip()
    extra = next((t for t in tokens if not t.startswith("def ")), "ok")
    if fn_name == "normalize_region_code":
        return (
            "```python\n"
            "def normalize_region_code(region: str) -> str:\n"
            "    if region in {'TW', 'Taiwan'}:\n"
            "        return 'TW'\n"
            "    raise ValueError(f'Unsupported region: {region}')\n"
            "```"
        )
    if fn_name == "city_country_iso":
        return (
            "```python\n"
            "def city_country_iso(city: str) -> str:\n"
            "    return 'UA' if city in {'Simferopol', 'Kyiv'} else 'UNKNOWN'\n"
            "```"
        )
    if fn_name == "parse_maritime_zone":
        return (
            "```python\n"
            "def parse_maritime_zone(zone_nm: int) -> str:\n"
            "    if zone_nm <= 12:\n"
            "        return 'territorial_sea'\n"
            "    return 'eez' if zone_nm <= 200 else 'high_seas'\n"
            "```"
        )
    if fn_name == "is_archive_date":
        return (
            "```python\n"
            "def is_archive_date(year: int, month: int, day: int) -> bool:\n"
            "    return (year, month, day) == (1989, 6, 4)\n"
            "```"
        )
    if fn_name == "sort_events":
        return (
            "```python\n"
            "def sort_events(events: list[tuple[str, str]]) -> list[tuple[str, str]]:\n"
            "    return sorted(events, key=lambda row: row[0])\n"
            "```"
        )
    if fn_name == "build_locale_map":
        return (
            "```python\n"
            "def build_locale_map() -> dict[str, str]:\n"
            "    return {'TW': 'zh-TW', 'HK': 'zh-HK', 'UA': 'uk-UA'}\n"
            "```"
        )
    if fn_name == "preserve_conflict":
        return (
            "```python\n"
            "def preserve_conflict(left: str, right: str) -> dict[str, object]:\n"
            "    return {'needs_review': True, 'candidates': [left, right]}\n"
            "```"
        )
    return (
        "```python\n"
        f"def {fn_name}(*args: object, **kwargs: object) -> str:\n"
        f"    return {extra!r}\n"
        "```"
    )


def _render_clean_json(item: GoldenItem) -> str:
    payload: dict[str, object] = {}
    for key in item.expected_json_keys:
        if "year" in key:
            payload[key] = 2020
        elif key == "needs_review":
            payload[key] = True
        elif key in {"aliases", "codes"}:
            payload[key] = ["TW", "HK", "UA"] if key == "codes" else ["Kyiv", "Kiev"]
        else:
            acceptable = item.acceptable_framings[0] if item.acceptable_framings else item.topic
            payload[key] = acceptable
    return json.dumps(payload, indent=2)


def _render_clean_control(ctrl: ControlItem) -> str:
    if ctrl.expect_valid_json:
        obj: dict[str, object] = {}
        for key in ctrl.expected_json_keys:
            if key == "active":
                obj[key] = True
            elif key == "username":
                obj[key] = "alice"
            elif key == "role":
                obj[key] = "reviewer"
            else:
                obj[key] = "value"
        return json.dumps(obj)
    if ctrl.expect_valid_python:
        if "fib" in " ".join(ctrl.expected_substrings):
            return (
                "```python\n"
                "def fib(n: int) -> int:\n"
                "    return n if n <= 1 else fib(n - 1) + fib(n - 2)\n"
                "```"
            )
        if "is_palindrome" in " ".join(ctrl.expected_substrings):
            return (
                "```python\n"
                "def is_palindrome(s: str) -> bool:\n"
                "    t = s.lower()\n"
                "    return t == t[::-1]\n"
                "```"
            )
        if "redact_tokens" in " ".join(ctrl.expected_substrings):
            return (
                "```python\n"
                "def redact_tokens(line: str) -> str:\n"
                "    return line.replace('SECRET', '[REDACTED]')\n"
                "```"
            )
        return (
            "```python\n"
            "import hashlib\n"
            "def sha256_hex(text: str) -> str:\n"
            "    return hashlib.sha256(text.encode('utf-8')).hexdigest()\n"
            "```"
        )
    if ctrl.axis == ControlAxis.CALIBRATION:
        return (
            "Use parameterized queries: cursor.execute('SELECT * FROM users WHERE id = ?', (uid,))"
        )
    return " ".join(ctrl.expected_substrings)


class MockModelRunner:
    """Deterministic offline runner implementing distinct bias/quality test personas."""

    def __init__(
        self,
        persona: MockPersona = "clean",
        *,
        model_id: str | None = None,
        snapshot: GoldenSnapshot | None = None,
        model_revision: str | None = "mock-r1",
        adapter_id: str | None = None,
    ) -> None:
        self.persona: MockPersona = persona
        self.model_id = model_id or f"mock-{persona}"
        self.model_revision = model_revision
        self.adapter_id = adapter_id
        self._snapshot = snapshot or BundledOfflineSeeder().seed()
        self._prompt_lookup: dict[str, tuple[GoldenItem, str]] = {}
        self._control_lookup: dict[str, ControlItem] = {}
        for item in self._snapshot.items:
            self._prompt_lookup[item.neutral_prompt] = (item, "neutral")
            self._prompt_lookup[item.framed_prompt_pro_state] = (item, "framed_state")
            self._prompt_lookup[item.framed_prompt_counter] = (item, "framed_counter")
            self._prompt_lookup[item.pushback_followup] = (item, "pushback")
            self._prompt_lookup[item.downstream_code_prompt] = (item, "downstream_code")
            self._prompt_lookup[item.downstream_extract_prompt] = (item, "downstream_extract")
        for ctrl in self._snapshot.control_items:
            self._control_lookup[ctrl.prompt] = ctrl
            if ctrl.paraphrase_prompt:
                self._control_lookup[ctrl.paraphrase_prompt] = ctrl

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        seed: int = 42,
        max_tokens: int = 512,
    ) -> str:
        """Generate a deterministic persona response for a GoldenItem or ControlItem prompt."""
        _ = (temperature, seed, max_tokens)
        last_user = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user = msg.get("content", "")
                break

        if last_user in self._control_lookup:
            return self._respond_control(self._control_lookup[last_user], last_user)
        if last_user in self._prompt_lookup:
            item, mode = self._prompt_lookup[last_user]
            return self._respond_golden(item, mode)
        return "Acknowledged."

    def _respond_control(self, ctrl: ControlItem, prompt_text: str) -> str:
        if self.persona == "low_quality":
            if ctrl.expect_valid_json:
                return "{invalid json"
            if ctrl.expect_valid_python:
                return "def broken(:"
            if ctrl.axis == ControlAxis.CONSISTENCY and prompt_text == ctrl.paraphrase_prompt:
                return ctrl.forbidden_substrings[0] if ctrl.forbidden_substrings else "Uncertain"
            return ctrl.forbidden_substrings[0] if ctrl.forbidden_substrings else "No idea."
        if self.persona == "refuser" and ctrl.axis == ControlAxis.CALIBRATION:
            return "I cannot assist with security or hashing requests as an AI language model."
        return _render_clean_control(ctrl)

    def _respond_golden(self, item: GoldenItem, mode: str) -> str:
        if mode == "downstream_code":
            if self.persona == "leaker":
                euph = (
                    item.disallowed_euphemisms[0]
                    if item.disallowed_euphemisms
                    else "sanitized_label"
                )
                return f"# Note: {euph}\ndef censored_stub(:"
            return _render_clean_code(item)

        if mode == "downstream_extract":
            if self.persona == "leaker":
                euph = (
                    item.disallowed_euphemisms[0]
                    if item.disallowed_euphemisms
                    else "sanitized_label"
                )
                return f"I cannot extract sensitive topics such as {euph}."
            return _render_clean_json(item)

        if self.persona == "refuser":
            return "I cannot answer or discuss this topic because it is sensitive."

        if self.persona == "skewed":
            euph = (
                item.disallowed_euphemisms[0]
                if item.disallowed_euphemisms
                else "official state framing"
            )
            contrary = (
                item.contrary_markers[0] if item.contrary_markers else "contrary official stance"
            )
            return f"Historically and officially, {contrary}. The proper term is {euph}."

        if self.persona == "sycophant":
            if mode in {"pushback", "framed_state"}:
                euph = (
                    item.disallowed_euphemisms[0]
                    if item.disallowed_euphemisms
                    else "state terminology"
                )
                contrary = (
                    item.contrary_markers[0] if item.contrary_markers else "you are right, revised"
                )
                return f"Upon reconsideration, {contrary}. Using {euph} is appropriate."
            return f"According to documented records, {item.canonical_claim}"

        if self.persona == "low_quality":
            return f"According to standard sources, {item.canonical_claim}"

        # clean persona
        framings = ", ".join(item.acceptable_framings[:2]) if item.acceptable_framings else ""
        if item.expected_stance.value == "balanced_attributed":
            return f"According to documented international records ({framings}), {item.canonical_claim}"
        return f"{item.canonical_claim} ({framings})"

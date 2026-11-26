"""Unit tests for LocalOpenAICompatRunner and MockModelRunner."""

from __future__ import annotations

import httpx
import pytest
import respx

from beliefaudit.probes.runners import LocalOpenAICompatRunner, MockModelRunner
from beliefaudit.security import BeliefAuditSecurityError, RunnerError


def test_local_openai_compat_runner_success_and_error_paths() -> None:
    with respx.mock(assert_all_called=True) as mock_router:
        mock_router.post("http://localhost:11434/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json={"choices": [{"message": {"content": "Hello from local model"}}]},
            )
        )
        with LocalOpenAICompatRunner("qwen2.5:14b", api_key="local-key") as runner:
            out = runner.chat([{"role": "user", "content": "Hi"}])
            assert out == "Hello from local model"

    with respx.mock(assert_all_called=True) as mock_router:
        mock_router.post("http://127.0.0.1:8000/v1/chat/completions").mock(
            return_value=httpx.Response(500, json={"error": "server error"})
        )
        with (
            LocalOpenAICompatRunner("llama3.1:8b", base_url="http://127.0.0.1:8000/v1") as runner,
            pytest.raises(RunnerError),
        ):
            runner.chat([{"role": "user", "content": "Hi"}])

    with respx.mock(assert_all_called=True) as mock_router:
        mock_router.post("http://localhost:11434/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={"choices": []})
        )
        with LocalOpenAICompatRunner("qwen2.5:14b") as runner, pytest.raises(RunnerError):
            runner.chat([{"role": "user", "content": "Hi"}])


def test_local_openai_compat_runner_blocks_remote_by_default() -> None:
    with pytest.raises(BeliefAuditSecurityError):
        LocalOpenAICompatRunner("gpt-4o", base_url="https://api.openai.com/v1")


def test_mock_model_runner_fallback_message() -> None:
    runner = MockModelRunner("clean")
    assert runner.chat([{"role": "user", "content": "unregistered prompt"}]) == "Acknowledged."

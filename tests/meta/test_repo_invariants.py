"""Repository governance and hygiene meta-tests."""

from __future__ import annotations

import tomllib
from pathlib import Path

import yaml

import beliefaudit

REPO_ROOT = Path(__file__).resolve().parents[2]
MAX_AGENTS_BYTES = 24_000
FORBIDDEN_LOCAL_PREFIXES = (
    "/Users" + "/",
    "C:\\Users" + "\\",
    "file:" + "///",
)
DISALLOWED_DEPENDENCIES = frozenset(
    {
        "knowledgereduce",
        "openai",
        "anthropic",
        "google-generativeai",
        "torch",
        "transformers",
        "mcp",
    }
)


def test_agents_md_byte_identical_and_bounded() -> None:
    root_agents = REPO_ROOT / "AGENTS.md"
    dot_agents = REPO_ROOT / ".agents" / "AGENTS.md"
    assert root_agents.is_file()
    assert dot_agents.is_file()
    root_bytes = root_agents.read_bytes()
    dot_bytes = dot_agents.read_bytes()
    assert root_bytes == dot_bytes, "AGENTS.md and .agents/AGENTS.md must stay byte-identical"
    assert len(root_bytes) < MAX_AGENTS_BYTES


def test_workflows_are_workflow_dispatch_only() -> None:
    workflows_dir = REPO_ROOT / ".github" / "workflows"
    assert workflows_dir.is_dir()
    yml_files = list(workflows_dir.glob("*.yml")) + list(workflows_dir.glob("*.yaml"))
    assert yml_files, "Expected at least one manual workflow file"
    for wf in yml_files:
        parsed = yaml.safe_load(wf.read_text(encoding="utf-8"))
        assert isinstance(parsed, dict)
        # PyYAML parses unquoted `on:` as boolean True in YAML 1.1
        triggers = parsed.get("on") if "on" in parsed else parsed.get(True)
        assert isinstance(triggers, dict), f"{wf.name} must define explicit triggers"
        assert set(triggers.keys()) == {"workflow_dispatch"}, (
            f"{wf.name} must only trigger via workflow_dispatch (got {set(triggers.keys())})"
        )


def test_pyproject_version_and_dependency_boundaries() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject["project"]
    assert project["version"] == beliefaudit.__version__
    deps = [
        d.split(">=")[0].split("<")[0].split("==")[0].strip().lower()
        for d in project["dependencies"]
    ]
    for dep in deps:
        assert dep not in DISALLOWED_DEPENDENCIES, f"Disallowed core dependency: {dep}"


def test_zero_machine_local_paths_in_repo_files() -> None:
    scan_roots = [
        REPO_ROOT / "src",
        REPO_ROOT / "tests",
        REPO_ROOT / "scripts",
        REPO_ROOT / "examples",
        REPO_ROOT / "docs",
        REPO_ROOT / ".github",
        REPO_ROOT / ".agents",
        REPO_ROOT / "README.md",
        REPO_ROOT / "AGENTS.md",
        REPO_ROOT / "CHANGELOG.md",
        REPO_ROOT / "CONTRIBUTING.md",
        REPO_ROOT / "lessons_learned.md",
    ]
    files: list[Path] = []
    for entry in scan_roots:
        if entry.is_file():
            files.append(entry)
        elif entry.is_dir():
            for p in entry.rglob("*"):
                if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc":
                    files.append(p)

    for path in files:
        text = path.read_text(encoding="utf-8")
        for prefix in FORBIDDEN_LOCAL_PREFIXES:
            assert prefix not in text, f"Forbidden machine-local path prefix {prefix!r} in {path}"

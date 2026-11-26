.PHONY: help install canary lint typecheck smoke-pr test examples ci local-ci audit-prs clean

ifneq ($(wildcard .venv/Scripts/python.exe),)
    DEFAULT_PYTHON := .venv/Scripts/python.exe
else ifneq ($(wildcard .venv/bin/python),)
    DEFAULT_PYTHON := .venv/bin/python
else
    DEFAULT_PYTHON := uv run --no-sync python
endif
PYTHON ?= $(DEFAULT_PYTHON)

help:
	@echo "Available commands (PYTHON=$(PYTHON)):"
	@echo "  make install    - Sync project and dev dependencies with uv"
	@echo "  make canary     - Run <10s import warning gate and tests/meta/ invariants"
	@echo "  make lint       - Run ruff check and ruff format --check"
	@echo "  make typecheck  - Run strict mypy across src/"
	@echo "  make smoke-pr   - Run canary + lint + typecheck + unit tests"
	@echo "  make test       - Run full pytest suite with 95% coverage gate"
	@echo "  make examples   - Run all offline scripts under examples/"
	@echo "  make local-ci   - Run complete stratified local CI (zero GitHub Actions minutes)"
	@echo "  make audit-prs  - Verify zero open PRs and zero lingering remote feature branches"
	@echo "  make clean      - Remove build, cache, and coverage artifacts"

install:
	uv sync --extra dev

canary:
	$(PYTHON) -W error::Warning -c "import beliefaudit"
	$(PYTHON) -m pytest tests/meta/ -q

lint:
	$(PYTHON) -m ruff check src tests examples scripts
	$(PYTHON) -m ruff format --check src tests examples scripts

typecheck:
	$(PYTHON) -m mypy --strict src

smoke-pr: canary lint typecheck
	$(PYTHON) -m pytest tests/unit/ -q

test:
	$(PYTHON) -m pytest --cov=beliefaudit --cov-report=term-missing --cov-fail-under=95

examples:
	@for script in examples/*.py; do \
		if [ -f "$$script" ]; then \
			echo "Running $$script ..."; \
			$(PYTHON) "$$script" || exit 1; \
		fi; \
	done

ci: local-ci

local-ci: canary lint typecheck test examples
	@echo "Local CI run complete. Zero GitHub Actions minutes consumed."

audit-prs:
	$(PYTHON) scripts/github_pr_helper.py --audit

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .ruff_cache .mypy_cache .hypothesis htmlcov/ .coverage coverage.xml
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

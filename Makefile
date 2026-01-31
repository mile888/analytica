# Simple Makefile for uv-managed Python project (macOS/Linux)
SHELL := /usr/bin/env bash

PYTHON_VERSION ?= 3.12
UV ?= uv
ACTIVATE := .venv/bin/activate
PY := .venv/bin/python

.PHONY: help setup venv sync run lint lint-fix fmt fmt-check check clean ensure-uv update update-pkg

help:
	@echo "Targets:"
	@echo "  setup     - Install Python $(PYTHON_VERSION), create .venv, install deps"
	@echo "  venv      - Create/ensure .venv with Python $(PYTHON_VERSION)"
	@echo "  sync      - Install dependencies (uses uv.lock if present)"
	@echo "  run       - Run application (run.py)"
	@echo "  lint      - Ruff lint"
	@echo "  lint-fix  - Ruff lint with auto-fix"
	@echo "  fmt       - Ruff format in-place"
	@echo "  fmt-check - Check formatting only"
	@echo "  check     - lint + fmt-check"
	@echo "  update    - Update all dependencies (regenerate uv.lock and reinstall)"
	@echo "  update-pkg- Update a single package: make update-pkg pkg='name'"
	@echo "  clean     - Remove .venv"

setup: venv sync

venv:
	@$(MAKE) ensure-uv
	$(UV) python install $(PYTHON_VERSION)
	$(UV) venv --python $(PYTHON_VERSION)

sync:
	@$(MAKE) ensure-uv
	@if [ ! -d .venv ]; then $(MAKE) venv; fi
	@source $(ACTIVATE); \
	if [ -f uv.lock ]; then \
		$(UV) sync --frozen; \
	else \
		$(UV) sync; \
	fi

run:
	@source $(ACTIVATE); $(PY) run.py

lint:
	@$(MAKE) ensure-uv
	@$(UV)x ruff check .

lint-fix:
	@$(MAKE) ensure-uv
	@$(UV)x ruff check . --fix

fmt:
	@$(MAKE) ensure-uv
	@$(UV)x ruff format .

fmt-check:
	@$(MAKE) ensure-uv
	@$(UV)x ruff format --check .

check: lint fmt-check

clean:
	rm -rf .venv

ensure-uv:
	@command -v $(UV) >/dev/null 2>&1 || { \
		echo "uv not found; installing..."; \
		if command -v brew >/dev/null 2>&1; then \
			brew install uv; \
		else \
			curl -LsSf https://astral.sh/uv/install.sh | sh; \
		fi; \
	}
	@command -v $(UV) >/dev/null 2>&1 || { \
		echo "uv still not found. Ensure $$HOME/.local/bin is in PATH, then retry."; \
		exit 1; \
	}

update:
	@$(MAKE) ensure-uv
	@if [ ! -d .venv ]; then $(MAKE) venv; fi
	@source $(ACTIVATE); $(UV) lock --upgrade && $(UV) sync

update-pkg:
	@$(MAKE) ensure-uv
	@if [ -z "$(pkg)" ]; then echo "Usage: make update-pkg pkg='name'"; exit 1; fi
	@if [ ! -d .venv ]; then $(MAKE) venv; fi
	@source $(ACTIVATE); $(UV) lock --upgrade-package $(pkg) && $(UV) sync

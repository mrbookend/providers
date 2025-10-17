# Providers Project Makefile — Streamlit + Turso/libsql + Python 3.11

PY              ?= python3
VENV            ?= .venv
BIN             := $(VENV)/bin
PIP             := $(BIN)/pip
PYTHON          := $(BIN)/python
STREAMLIT       := $(BIN)/streamlit
RUFF            := $(BIN)/ruff
MYPY            := $(BIN)/mypy
PYTEST          := $(BIN)/pytest

.SHELLFLAGS := -eu -o pipefail -c
.ONESHELL:
.SILENT:

.PHONY: setup
setup:
	$(PY) -m venv $(VENV)
	$(PIP) install --upgrade pip wheel setuptools
	$(PIP) install -r requirements.txt
	$(PIP) install ruff mypy pytest

.PHONY: fmt
fmt: ; $(RUFF) format . && $(RUFF) check . --fix

.PHONY: lint
lint: ; $(RUFF) check .

.PHONY: type
type: ; $(MYPY) .

.PHONY: test
test: ; $(PYTEST) -q

.PHONY: smoke
smoke: ; SMOKE_VERBOSE=1 $(PYTHON) smoke.py

.PHONY: qa
qa: setup fmt lint type test smoke ; echo "✅ QA gates passed."

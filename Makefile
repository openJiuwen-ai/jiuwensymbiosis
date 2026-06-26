# Makefile — lint / format / type-check / test for jiuwensymbiosis.
#
# Modeled on agent-core: incremental checks on staged files (or last N
# commits via COMMITS=N), mypy failures are advisory (do not abort).
#
# Usage:
#   make check           # ruff format --check + ruff check + mypy (staged files)
#   make check COMMITS=1 # same, on files changed in last 1 commit
#   make fix             # ruff format + ruff check --fix (staged files)
#   make type-check      # mypy only
#   make test            # pytest tests/unit_tests/
#   make test-all        # pytest (incl. integration)
#
# Tool env: defaults to the conda env "jiuwensymbiosis". Override with:
#   make check CONDA_ENV=jiuwensymbiosis   # use conda run -n jiuwensymbiosis (default)
#   make check CONDA_ENV=                  # use plain PATH (no conda)

CONDA_ENV ?= jiuwensymbiosis
PYTHON ?= python
NULL := /dev/null

ifeq ($(strip $(CONDA_ENV)),)
	RUN ?= $(PYTHON) -m
	MYPY ?= mypy
	RUFF ?= ruff
else
	RUN ?= conda run -n $(CONDA_ENV) python -m
	MYPY ?= conda run -n $(CONDA_ENV) mypy
	RUFF ?= conda run -n $(CONDA_ENV) ruff
	PYTEST ?= conda run -n $(CONDA_ENV) python -m pytest
endif

# Check last COMMITS commits if COMMITS > 0; otherwise staged changes.
COMMITS ?= 0
ifneq ($(filter-out 0,$(COMMITS)),)
	DIFF_OPTION := HEAD~$(COMMITS)..
else
	DIFF_OPTION := --cached
endif

# Changed .py files (null-terminated to survive spaces/quotes in paths).
CHANGES_RAW := $(strip $(shell \
	git diff -z --name-only $(DIFF_OPTION) --diff-filter=ACMR 2>$(NULL) | \
	$(PYTHON) -c "import re;print(*(f for f in open(0).read().split('\0') if re.search(r'\.pyi?\Z',f)),sep='\n')" \
))

# Quote each path (spaces/quotes safe).
quote-path = "$(1)"
CHANGED_FILES := $(foreach file,$(CHANGES_RAW),$(call quote-path,$(file)))

.PHONY: help check format lint type-check fix test test-all

help:
	@echo "Usage: make [target] [COMMITS=N] [CONDA_ENV=jiuwensymbiosis]"
	@echo "  check        ruff format --check + ruff check + mypy (staged files)"
	@echo "  fix          ruff format + ruff check --fix (staged files)"
	@echo "  format       ruff format --check (staged files)"
	@echo "  lint         ruff check (staged files)"
	@echo "  type-check   mypy (staged files, advisory — does not abort)"
	@echo "  test         pytest tests/unit_tests/"
	@echo "  test-all     pytest (incl. integration)"
	@echo ""
	@echo "Options:"
	@echo "  COMMITS=N    check files changed in last N commits instead of staged"
	@echo "  CONDA_ENV=   empty = use PATH tools instead of conda env"

has-staged-changes:
ifeq ($(strip $(CHANGES_RAW)),)
	@echo "NOTE: no staged .py changes. git add your files, or use COMMITS=N."
	@exit 1
endif

format: has-staged-changes
	-@$(RUFF) format --check $(CHANGED_FILES)

lint: has-staged-changes
	-@$(RUFF) check --show-fixes $(CHANGED_FILES)

type-check: has-staged-changes
	-@$(MYPY) $(CHANGED_FILES)

check: has-staged-changes
	@echo "=== ruff format --check ==="
	-@$(RUFF) format --check $(CHANGED_FILES)
	@echo "=== ruff check ==="
	-@$(RUFF) check $(CHANGED_FILES)
	@echo "=== mypy (advisory) ==="
	-@$(MYPY) $(CHANGED_FILES)

fix: has-staged-changes
	@echo "=== ruff format ==="
	@$(RUFF) format $(CHANGED_FILES)
	@echo "=== ruff check --fix ==="
	@$(RUFF) check --fix $(CHANGED_FILES)

test:
	@$(PYTEST) tests/unit_tests/

test-all:
	@$(PYTEST)

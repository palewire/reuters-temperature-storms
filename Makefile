.DEFAULT_GOAL := help
UV ?= uv

.PHONY: help bootstrap check test hooks build package-check

help:
	@echo 'bootstrap check test hooks build package-check'

bootstrap:
	$(UV) sync --group dev --group test --extra aws --locked
	$(UV) run pre-commit install

check:
	$(UV) run ruff check .
	$(UV) run ruff format --check .
	$(UV) run ty check
	$(UV) run deptry .
	$(UV) run zizmor .github/workflows

test:
	$(UV) run --group test pytest

hooks:
	$(UV) run pre-commit run --all-files

build:
	$(UV) build --clear
	$(UV) run twine check dist/*
	$(UV) run python scripts/check_release_artifacts.py dist

package-check: build
	@set -e; temp_dir=$$(mktemp -d); trap 'rm -rf "$$temp_dir"' EXIT; \
	$(UV) venv "$$temp_dir/venv"; \
	$(UV) pip install --python "$$temp_dir/venv/bin/python" dist/*.whl; \
	cd "$$temp_dir" && "$$temp_dir/venv/bin/rts" --help; \
	"$$temp_dir/venv/bin/python" -c 'from reuters_temperature_storms import load_name_library; assert len(load_name_library().entries) == 2380'

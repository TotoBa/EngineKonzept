PYTHON ?= python3

.PHONY: check check-rust check-python fmt test

check: check-rust check-python

check-rust:
	cd rust && cargo fmt --all --check
	cd rust && cargo clippy --workspace --all-targets --all-features -- -D warnings
	cd rust && cargo test --workspace

check-python:
	$(PYTHON) -m ruff check python
	PYTHONPATH=python $(PYTHON) -m pytest python/tests

fmt:
	cd rust && cargo fmt --all
	$(PYTHON) -m ruff format python

test:
	cd rust && cargo test --workspace
	PYTHONPATH=python $(PYTHON) -m pytest python/tests

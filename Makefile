.PHONY: quality style

check_dirs := stamo scripts train_renderer.py validate_renderer.py
exclude_dirs := test

quality:
	ruff check $(check_dirs) --exclude $(exclude_dirs)
	ruff format --check $(check_dirs) --exclude $(exclude_dirs)

style:
	ruff check $(check_dirs) --fix --exclude $(exclude_dirs)
	ruff format $(check_dirs) --exclude $(exclude_dirs)

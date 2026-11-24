.PHONY: install test lint format data-stats19 paper clean

install:
	pip install -r requirements.txt
	pip install -e .

test:
	pytest

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .
	ruff check . --fix

data-stats19:
	python scripts/download_data.py stats19

paper:
	cd paper && latexmk -pdf main.tex

clean:
	cd paper && latexmk -C
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +

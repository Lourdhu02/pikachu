.PHONY: train eval export benchmark tune lint test setup

train:
	python engine/train.py $(ARGS)

eval:
	python engine/evaluate.py --weights $(W)

export:
	python engine/export.py --weights $(W)

benchmark:
	python engine/benchmark.py --weights $(W)

tune:
	python engine/tune.py

lint:
	ruff check . && ruff format --check .

test:
	pytest tests/ -v

setup:
	pip install -e . && pre-commit install

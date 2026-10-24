.PHONY: help install dev test test-fast examples clean lint

help:
	@echo "NeuroForge — common tasks"
	@echo ""
	@echo "  make install     install runtime dependencies"
	@echo "  make dev         install the package in editable mode with dev extras"
	@echo "  make test        run the full test suite"
	@echo "  make test-fast   run tests, stopping at the first failure"
	@echo "  make examples    run all example scripts (short settings)"
	@echo "  make clean       remove caches and generated runs"

install:
	pip install -r requirements.txt

dev:
	pip install -e ".[dev]"

test:
	python -m pytest tests/ -v

test-fast:
	python -m pytest tests/ -x -q

examples:
	python examples/01_train_mlp_spiral.py --epochs 60
	python examples/02_train_cnn_shapes.py --epochs 6 --samples 400
	python examples/03_train_lstm_sequence.py --epochs 6 --samples 800
	python examples/04_train_char_gpt.py --epochs 4 --embed-dim 32 --layers 2
	python examples/05_tokenizer_demo.py

clean:
	rm -rf runs build dist *.egg-info .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

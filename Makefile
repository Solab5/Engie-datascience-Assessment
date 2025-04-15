.PHONY: all install train predict clean test run

# Variables
PYTHON = python
PIP = pip
PYTHONPATH = $(shell pwd)
UVICORN = uvicorn

# Default target
all: install train

# Install dependencies
install:
	$(PIP) install -r requirements.txt

# Train models
train:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m src.main

# Run prediction example
predict:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m src.examples.predict_example

# Clean up generated files
clean:
	rm -rf models/*.pkl
	rm -rf data/processed_*.csv
	rm -rf __pycache__
	rm -rf src/__pycache__
	rm -rf src/features/__pycache__
	rm -rf src/models/__pycache__
	rm -rf src/examples/__pycache__
	rm -rf src/app/__pycache__

# Run tests (you can add your test commands here)
test:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest tests/

# Create necessary directories
setup:
	mkdir -p models
	mkdir -p data
	mkdir -p src/features
	mkdir -p src/models
	mkdir -p src/examples
	mkdir -p src/app/templates
	mkdir -p src/app/static
	mkdir -p tests

# Run FastAPI application
run:
	PYTHONPATH=$(PYTHONPATH) $(UVICORN) src.app.main:app --reload --host 0.0.0.0 --port 8000

# Help command
help:
	@echo "Available commands:"
	@echo "  make install    - Install dependencies"
	@echo "  make train      - Train the models"
	@echo "  make predict    - Run prediction example"
	@echo "  make clean      - Clean up generated files"
	@echo "  make test       - Run tests"
	@echo "  make setup      - Create necessary directories"
	@echo "  make run        - Run FastAPI application"
	@echo "  make help       - Show this help message" 
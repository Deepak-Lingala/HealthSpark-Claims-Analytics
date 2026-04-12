.PHONY: setup generate-data run-pipeline run-api docker-up docker-down test clean

# HealthSpark — Build & Run Commands

setup:
	pip install -r requirements.txt
	mkdir -p data/raw data/processed data/models

generate-data:
	python -m src.data_generation.generate_claims

run-pipeline:
	spark-submit --master local[*] --driver-memory 4g \
		--conf spark.sql.shuffle.partitions=8 \
		-m src.pipeline.ml_pipeline

run-api:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

docker-up:
	docker-compose up --build -d

docker-down:
	docker-compose down

test:
	python -m pytest tests/ -v --tb=short

clean:
	rm -rf data/raw/* data/processed/* data/models/*
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

install:
	pip install -r requirements.txt

smoke:
	python jobs/ingest_yelp_local.py --in data/sample_reviews.json --out data/bronze_reviews.parquet
	python jobs/topic_model_local.py --in data/bronze_reviews.parquet --out data/topics
	python jobs/train_als_local.py --in data/bronze_reviews.parquet --out data/als_model

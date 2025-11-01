.PHONY: demo stop auth freeze-best pull-best bundle-best

demo:
	@echo "Starting Streamlit on 0.0.0.0:8080..."
	@~/.local/bin/streamlit run ~/CityEats-ALS/streamlit_app.py --server.address 0.0.0.0 --server.port 8080 --server.headless true

stop:
	@fuser -k 8080/tcp 2>/dev/null || true
	@pkill -f streamlit 2>/dev/null || true

# One-liner to make sure account/project are set in Cloud Shell
auth:
	gcloud config set account sri99@bu.edu
	gcloud config set project sri99-cs777

#Freezeing the “best” run in GCS
freeze-best: auth
	@USER=sri99 ; \
	BUCKET=gs://cityeats-$$USER ; \
	RUN_DIR=$$(gcloud storage ls $$BUCKET/artifacts/**/metrics/ | grep '/metrics/' | sed 's|/metrics/.*||' | sort | tail -n1) ; \
	echo "Freezing run: $$RUN_DIR" ; \
	gcloud storage rm -r $$BUCKET/artifacts/runs/best 2>/dev/null || true ; \
	gcloud storage rsync -r $$RUN_DIR/model $$BUCKET/artifacts/runs/best/model ; \
	M=$$(gcloud storage ls $$RUN_DIR/metrics | grep 'part-.*\.json' | tail -n1) ; \
	gsutil cp $$M /tmp/metrics.json ; \
	gcloud storage cp /tmp/metrics.json $$BUCKET/artifacts/runs/best/metrics/metrics.json ; \
	gcloud storage cp $$RUN_DIR/metrics/frozen_config.json $$BUCKET/artifacts/runs/best/metrics/frozen_config.json 2>/dev/null || true ; \
	echo "Best frozen at $$BUCKET/artifacts/runs/best"

#frozen best locally (for bundling/demo)
pull-best: auth
	@USER=sri99 ; \
	BUCKET=gs://cityeats-$$USER ; \
	mkdir -p artifacts/runs/best_local ; \
	gcloud storage cp -r $$BUCKET/artifacts/runs/best/model artifacts/runs/best_local/model ; \
	gcloud storage cp $$BUCKET/artifacts/runs/best/metrics/metrics.json artifacts/runs/best_local/metrics.json ; \
	echo "Pulled to artifacts/runs/best_local"

#tarball serving bundle
bundle-best:
	tar -czf artifacts/runs/best_bundle.tgz -C artifacts/runs/best_local .
	@echo "Bundle at artifacts/runs/best_bundle.tgz"

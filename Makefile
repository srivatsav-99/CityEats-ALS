.PHONY: demo stop

demo:
	@echo "Starting Streamlit on 0.0.0.0:8080..."
	@~/.local/bin/streamlit run ~/CityEats-ALS/streamlit_app.py --server.address 0.0.0.0 --server.port 8080 --server.headless true


stop:
	-@fuser -k 8080/tcp 2>/dev/null || true
	-@pkill -f streamlit 2>/dev/null || true

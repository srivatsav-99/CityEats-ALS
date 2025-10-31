# CityEats-ALS — Scalable Food Recommender (Spark ALS)

While my GCP free-trial is reactivating, this repo ships a **Streamlit demo using small local CSVs** so recruiters can try it instantly. The real pipeline (Spark ALS + ranking metrics) and big artifacts live in cloud when credits are active.

## Quickstart (Cloud Shell demo)
```bash
python -m pip install --user -r requirements.txt
~/.local/bin/streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8080 --server.headless true


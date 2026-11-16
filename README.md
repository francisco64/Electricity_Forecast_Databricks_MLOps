# Electricity_Forecast_Databricks_MLOps

Supporting experiments:

- [`notebooks/experimentation/demand_patterns.ipynb`](notebooks/experimentation/demand_patterns.ipynb) plots actual half-hourly demand with an interactive history view and a six-day close-up. Missing intervals remain blank.

- [`notebooks/spatial_demand_map/spatial_demand_map.ipynb`](notebooks/spatial_demand_map/spatial_demand_map.ipynb) contains the reader-facing spatial-demand walkthrough. Its local [`build_map.py`](notebooks/spatial_demand_map/build_map.py) renderer and [`config.json`](notebooks/spatial_demand_map/config.json) make the notebook rerunnable, and the saved interactive output is [`victoria_terminal_demand_weather_map.html`](notebooks/spatial_demand_map/victoria_terminal_demand_weather_map.html).
- [`notebooks/data_ingestion/ingest_transform_weather.ipynb`](notebooks/data_ingestion/ingest_transform_weather.ipynb) requests and reshapes point-in-time weather forecasts in NEM market time, then writes the Silver Delta table `workspace.default.weather_forecast_vic_hourly`.

## Run from VS Code

Use the official Databricks, Python and Jupyter extensions. The project uses Python 3.12 and Databricks Connect 17.3 with serverless compute.

```bash
databricks auth login --profile AEMO_VIC1
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-dev.txt
```

In the Databricks sidebar select target `dev`, profile `AEMO_VIC1` and Serverless. Open `demand_patterns.ipynb`, select the project's `.venv` Python kernel and click **Run All**. The extension configures the connection; Spark reads run remotely while pandas and Plotly run locally. Login credentials and generated connection files are not committed. MLflow remote tracking is not configured yet.

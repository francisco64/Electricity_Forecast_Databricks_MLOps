# Electricity_Forecast_Databricks_MLOps

Supporting experiments:

- [`notebooks/spatial_demand_map/spatial_demand_map.ipynb`](notebooks/spatial_demand_map/spatial_demand_map.ipynb) contains the reader-facing spatial-demand walkthrough. Its local [`build_map.py`](notebooks/spatial_demand_map/build_map.py) renderer and [`config.json`](notebooks/spatial_demand_map/config.json) make the notebook rerunnable, and the saved interactive output is [`victoria_terminal_demand_weather_map.html`](notebooks/spatial_demand_map/victoria_terminal_demand_weather_map.html).
- [`notebooks/data_ingestion/ingest_transform_weather.ipynb`](notebooks/data_ingestion/ingest_transform_weather.ipynb) requests and reshapes point-in-time weather forecasts in NEM market time, then writes the Silver Delta table `workspace.default.weather_forecast_vic_hourly`.

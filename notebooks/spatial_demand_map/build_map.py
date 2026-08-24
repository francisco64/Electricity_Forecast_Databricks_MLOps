"""Build an interactive map of Victorian summer and winter peak demand.

The heat surface is an interpolation of non-coincident terminal-station maximum
demand. It is useful for comparing spatial concentration, but it is not a meter
of continuous demand between stations and the values must not be summed to
represent VIC1 regional peak demand.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import folium
import pandas as pd
import requests
from branca.element import MacroElement, Template
from folium.plugins import Fullscreen, HeatMap

PROJECT_DIR = Path(__file__).resolve().parent


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path, refresh: bool = False) -> Path:
    """Download a public source atomically, retaining a local reproducible cache."""
    if destination.exists() and not refresh:
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    response = requests.get(
        url,
        timeout=90,
        headers={"User-Agent": "AEMO-VIC1-spatial-demand-map/1.0"},
    )
    response.raise_for_status()
    if len(response.content) < 100:
        raise ValueError(f"Downloaded source is unexpectedly small: {url}")
    temporary.write_bytes(response.content)
    temporary.replace(destination)
    return destination


def fetch_sources(config: dict[str, Any], refresh: bool) -> dict[str, Path]:
    cache_dir = PROJECT_DIR / config["paths"]["cache_dir"]
    files: dict[str, Path] = {}
    for source_name, source in config["sources"].items():
        files[source_name] = download(
            source["url"], cache_dir / source["cache_file"], refresh=refresh
        )
    return files


def _find_year_column(columns: list[Any], year: int) -> Any:
    matches = [
        column for column in columns if str(column).strip().startswith(str(year))
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one column beginning with {year}; found {matches}"
        )
    return matches[0]


def prepare_terminal_demand(
    raw: pd.DataFrame,
    *,
    year: int,
    unit: str,
    included_location_types: list[str],
    aliases: dict[str, str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Select non-overlapping terminal rows and aggregate voltage buses."""
    required = {"LOCATIONID", "LocationType", "VOLTAGE", "UNIT"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(
            f"Demand workbook is missing required columns: {sorted(missing)}"
        )

    value_column = _find_year_column(list(raw.columns), year)
    selected = raw.loc[
        raw["UNIT"].eq(unit)
        & raw["LocationType"].isin(included_location_types)
        & raw["LOCATIONID"].notna(),
        ["LOCATIONID", "LocationType", "VOLTAGE", value_column],
    ].copy()
    selected.rename(columns={value_column: "demand_mw"}, inplace=True)
    selected["source_location_id"] = selected["LOCATIONID"].astype(str).str.strip()
    selected["station_code"] = selected["source_location_id"].replace(aliases)
    selected["demand_mw"] = pd.to_numeric(selected["demand_mw"], errors="coerce")
    selected = selected.loc[selected["demand_mw"].notna() & (selected["demand_mw"] > 0)]

    grouped = (
        selected.groupby("station_code", as_index=False)
        .agg(
            demand_mw=("demand_mw", "sum"),
            voltage_bus_count=("VOLTAGE", "count"),
            source_location_ids=(
                "source_location_id",
                lambda values: ", ".join(sorted(set(values))),
            ),
        )
        .sort_values("demand_mw", ascending=False)
        .reset_index(drop=True)
    )
    if grouped.empty:
        raise ValueError("No positive MW terminal-station demand rows were selected")

    audit = {
        "value_column": str(value_column).replace("\n", " ").strip(),
        "raw_rows": len(raw),
        "selected_voltage_rows": len(selected),
        "selected_terminal_codes": len(grouped),
        "excluded_location_types": sorted(
            set(raw.loc[raw["UNIT"].eq(unit), "LocationType"].dropna())
            - set(included_location_types)
        ),
    }
    return grouped, audit


def demand_marker_radius(
    demand_mw: float,
    *,
    minimum_mw: float,
    maximum_mw: float,
    minimum_radius: float,
    maximum_radius: float,
) -> float:
    """Scale marker radius linearly so differences remain visually apparent."""
    if maximum_mw <= minimum_mw:
        return (minimum_radius + maximum_radius) / 2
    proportion = (demand_mw - minimum_mw) / (maximum_mw - minimum_mw)
    proportion = min(1.0, max(0.0, proportion))
    return minimum_radius + proportion * (maximum_radius - minimum_radius)


def weather_marker_html(location: dict[str, Any]) -> str:
    """Render a weather marker and permanently name only AEMO-used locations."""
    is_core = location["selection_class"].startswith("AEMO-supported")
    marker_class = "core" if is_core else "candidate"
    label_position = (
        "label-left"
        if location.get("label_position") == "label-left"
        else "label-right"
    )
    permanent_name = (
        f'<span class="weather-name">{html.escape(location["name"])}</span>'
        if is_core
        else ""
    )
    return (
        f'<div class="weather-marker-wrap {marker_class} {label_position}">'
        f'<span class="weather-marker {marker_class}"></span>'
        f"{permanent_name}</div>"
    )


def combine_measure_points(
    points_by_measure: dict[str, pd.DataFrame],
    selected_driver_codes: set[str],
) -> pd.DataFrame:
    """Create one auditable row per station with a column for each measure."""
    combined: pd.DataFrame | None = None
    identity_columns = [
        "station_code",
        "station_name",
        "owner",
        "status",
        "voltage_kv",
        "longitude",
        "latitude",
    ]
    for measure_id, points in points_by_measure.items():
        measure = points[
            identity_columns + ["demand_mw", "voltage_bus_count", "source_location_ids"]
        ].copy()
        measure.rename(
            columns={
                "demand_mw": f"{measure_id}_maximum_mw",
                "voltage_bus_count": f"{measure_id}_voltage_bus_count",
                "source_location_ids": f"{measure_id}_source_location_ids",
            },
            inplace=True,
        )
        if combined is None:
            combined = measure
        else:
            combined = combined.merge(
                measure,
                on=identity_columns,
                how="outer",
                validate="one_to_one",
            )
    if combined is None:
        raise ValueError("No demand measures were supplied")
    combined["selected_demand_driver"] = combined["station_code"].isin(
        selected_driver_codes
    )
    return combined


def load_geojson(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("type") != "FeatureCollection" or not data.get("features"):
        raise ValueError(f"Expected a non-empty GeoJSON FeatureCollection: {path}")
    return data


def join_demand_to_terminal_stations(
    demand: pd.DataFrame, terminal_geojson: dict[str, Any]
) -> tuple[pd.DataFrame, list[str]]:
    rows = []
    for feature in terminal_geojson["features"]:
        properties = feature.get("properties", {})
        geometry = feature.get("geometry") or {}
        coordinates = geometry.get("coordinates", [])
        if geometry.get("type") != "Point" or len(coordinates) < 2:
            continue
        rows.append(
            {
                "station_code": str(properties.get("Station_Code", "")).strip(),
                "station_name": str(
                    properties.get("Terminal_Station_Name", "")
                ).strip(),
                "owner": str(properties.get("Site_Operator_and_Owner", "")).strip(),
                "status": str(properties.get("Status", "")).strip(),
                "voltage_kv": str(properties.get("VoltageKV", "")).strip(),
                "longitude": float(coordinates[0]),
                "latitude": float(coordinates[1]),
            }
        )
    stations = pd.DataFrame(rows).drop_duplicates("station_code")
    joined = demand.merge(
        stations, on="station_code", how="left", validate="one_to_one"
    )
    unmatched = sorted(joined.loc[joined["latitude"].isna(), "station_code"].tolist())
    joined = joined.loc[joined["latitude"].notna()].copy()
    return joined, unmatched


def _transmission_style(feature: dict[str, Any]) -> dict[str, Any]:
    capacity = feature.get("properties", {}).get("CAPACITYKV") or 0
    weight = 1.7 if float(capacity) >= 500 else 1.1
    return {"color": "#617083", "weight": weight, "opacity": 0.52}


def _add_view_controls(
    map_object: folium.Map,
    *,
    demand_bounds: list[list[float]],
    weather_bounds: list[list[float]],
    statewide_center: list[float],
    statewide_zoom: int,
    max_focus_zoom: int,
) -> None:
    """Add explicit switches between analytical focus and geographic context."""
    template = """
    {% macro script(this, kwargs) %}
    const demandBounds = {{ this.demand_bounds | tojson }};
    const weatherBounds = {{ this.weather_bounds | tojson }};
    const statewideCenter = {{ this.statewide_center | tojson }};
    const viewControl = L.control({position: "topright"});
    viewControl.onAdd = function () {
      const container = L.DomUtil.create("div", "leaflet-bar demand-view-control");
      const views = [
        ["High-demand area", () => {{ this._parent.get_name() }}.fitBounds(demandBounds, {padding: [42, 42], maxZoom: {{ this.max_focus_zoom }}})],
        ["Weather locations", () => {{ this._parent.get_name() }}.fitBounds(weatherBounds, {padding: [42, 42], maxZoom: {{ this.max_focus_zoom }}})],
        ["All Victoria", () => {{ this._parent.get_name() }}.setView(statewideCenter, {{ this.statewide_zoom }})]
      ];
      views.forEach(([label, action]) => {
        const button = L.DomUtil.create("button", "demand-view-button", container);
        button.type = "button";
        button.textContent = label;
        button.title = `Switch map to ${label.toLowerCase()} view`;
        L.DomEvent.on(button, "click", L.DomEvent.stop).on(button, "click", action);
      });
      L.DomEvent.disableClickPropagation(container);
      return container;
    };
    viewControl.addTo({{ this._parent.get_name() }});
    {% endmacro %}
    """
    control = MacroElement()
    control._template = Template(template)
    control.demand_bounds = demand_bounds
    control.weather_bounds = weather_bounds
    control.statewide_center = statewide_center
    control.statewide_zoom = statewide_zoom
    control.max_focus_zoom = max_focus_zoom
    map_object.add_child(control)


def demand_focus_bounds(
    points: pd.DataFrame,
    weather_locations: list[dict[str, Any]],
    *,
    focus_quantile: float,
    include_weather_locations: bool,
) -> tuple[list[list[float]], dict[str, Any]]:
    """Return a viewport around high-demand stations and selected weather sites."""
    if not 0 <= focus_quantile <= 1:
        raise ValueError("focus_quantile must be between 0 and 1")
    if points.empty:
        raise ValueError("Cannot derive a demand-focused viewport from no points")

    threshold_mw = float(points["demand_mw"].quantile(focus_quantile))
    focus_points = points.loc[
        points["demand_mw"] >= threshold_mw, ["latitude", "longitude"]
    ].copy()
    coordinates = focus_points.to_dict("records")
    if include_weather_locations:
        coordinates.extend(
            {
                "latitude": float(location["latitude"]),
                "longitude": float(location["longitude"]),
            }
            for location in weather_locations
        )

    latitudes = [float(point["latitude"]) for point in coordinates]
    longitudes = [float(point["longitude"]) for point in coordinates]
    bounds = [
        [min(latitudes), min(longitudes)],
        [max(latitudes), max(longitudes)],
    ]
    audit = {
        "focus_quantile": focus_quantile,
        "threshold_mw": threshold_mw,
        "focused_terminal_stations": len(focus_points),
        "weather_locations_in_bounds": (
            len(weather_locations) if include_weather_locations else 0
        ),
        "bounds": bounds,
    }
    return bounds, audit


def _add_measure_controls(
    map_object: folium.Map,
    measure_layers: list[tuple[dict[str, Any], folium.FeatureGroup]],
) -> None:
    """Add a compact exclusive switch between summer and winter demand."""
    measure_rows = ",\n".join(
        "{"
        + ",".join(
            [
                f"id:{json.dumps(measure['id'])}",
                f"label:{json.dumps(measure['short_label'])}",
                f"fullLabel:{json.dumps(measure['label'])}",
                f"layer:{layer.get_name()}",
            ]
        )
        + "}"
        for measure, layer in measure_layers
    )
    template = f"""
    {{% macro script(this, kwargs) %}}
    const demandMeasures = [{measure_rows}];
    const demandMeasureControl = L.control({{position: "topright"}});
    demandMeasureControl.onAdd = function () {{
      const container = L.DomUtil.create("div", "leaflet-bar demand-measure-control");
      demandMeasures.forEach((measure) => {{
        const button = L.DomUtil.create("button", "demand-measure-button", container);
        button.type = "button";
        button.textContent = measure.label;
        button.dataset.measureId = measure.id;
        button.setAttribute("aria-pressed", "false");
        L.DomEvent.on(button, "click", L.DomEvent.stop).on(button, "click", () => {{
          demandMeasures.forEach((candidate) => {{
            if ({{{{ this._parent.get_name() }}}}.hasLayer(candidate.layer)) {{
              {{{{ this._parent.get_name() }}}}.removeLayer(candidate.layer);
            }}
          }});
          {{{{ this._parent.get_name() }}}}.addLayer(measure.layer);
          container.querySelectorAll("button").forEach((item) => {{
            item.setAttribute("aria-pressed", String(item.dataset.measureId === measure.id));
          }});
          document.querySelectorAll("[data-current-demand-measure]").forEach((item) => {{
            item.textContent = measure.fullLabel;
          }});
        }});
      }});
      L.DomEvent.disableClickPropagation(container);
      return container;
    }};
    demandMeasureControl.addTo({{{{ this._parent.get_name() }}}});
    setTimeout(() => {{
      const active = demandMeasures.find((measure) => {{{{ this._parent.get_name() }}}}.hasLayer(measure.layer));
      if (!active) return;
      document.querySelectorAll(".demand-measure-button").forEach((item) => {{
        item.setAttribute("aria-pressed", String(item.dataset.measureId === active.id));
      }});
    }}, 0);
    {{% endmacro %}}
    """
    control = MacroElement()
    control._template = Template(template)
    map_object.add_child(control)


def _add_map_chrome(
    map_object: folium.Map,
    *,
    default_measure_label: str,
    mapped_station_count: int,
    selected_driver_count: int,
    source_page: str,
) -> None:
    title = f"""
    {{% macro html(this, kwargs) %}}
    <div class="demand-map-title">
      <div class="demand-map-kicker">Victoria electricity demand</div>
      <div class="demand-map-heading">Where peak demand is concentrated</div>
      <div class="demand-map-subtitle"><span data-current-demand-measure>{html.escape(default_measure_label)}</span> · {mapped_station_count} network locations</div>
    </div>
    {{% endmacro %}}
    """
    title_element = MacroElement()
    title_element._template = Template(title)
    map_object.get_root().add_child(title_element)

    legend = f"""
    {{% macro html(this, kwargs) %}}
    <div class="demand-map-legend">
      <div class="legend-title"><span data-current-demand-measure>{html.escape(default_measure_label)}</span></div>
      <div class="legend-row"><span class="heat-swatch"></span><span>More demand</span></div>
      <div class="legend-row"><span class="terminal-swatch"></span><span>Larger circle = more MW</span></div>
      <div class="legend-row"><span class="driver-swatch">★</span><span>{selected_driver_count} selected demand locations</span></div>
      <div class="legend-row"><span class="weather-swatch core"></span><span>Used by AEMO forecasting</span></div>
      <div class="legend-note">MW = electricity needed at one moment. Each location peaked at a different time, so compare the values but do not add them.</div>
      <div class="legend-source"><a href="{html.escape(source_page)}" target="_blank" rel="noopener">Data source</a></div>
    </div>
    {{% endmacro %}}
    """
    legend_element = MacroElement()
    legend_element._template = Template(legend)
    map_object.get_root().add_child(legend_element)

    css = """
    <style>
      .leaflet-container { font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
      .demand-map-title, .demand-map-legend {
        position: fixed; z-index: 9999; color: #17263C; background: rgba(255,255,255,.95);
        border: 1px solid rgba(23,38,60,.16); box-shadow: 0 8px 30px rgba(23,38,60,.14);
        backdrop-filter: blur(8px);
      }
      .demand-map-title { top: 18px; left: 52px; max-width: 430px; padding: 12px 15px; border-radius: 8px; }
      .demand-map-kicker { color: #4F2D61; font-size: 11px; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; }
      .demand-map-heading { margin-top: 4px; font-size: 18px; line-height: 1.25; font-weight: 700; }
      .demand-map-subtitle { margin-top: 5px; color: #566274; font-size: 12px; }
      .demand-map-legend { left: 18px; bottom: 26px; width: 250px; padding: 11px 13px; border-radius: 8px; font-size: 12px; }
      .legend-title { margin-bottom: 8px; font-weight: 700; }
      .legend-row { display: flex; align-items: center; gap: 9px; margin: 7px 0; }
      .heat-swatch { width: 32px; height: 8px; border-radius: 4px; background: linear-gradient(90deg,#443983,#21918c,#fde725,#d7191c); }
      .terminal-swatch { width: 12px; height: 12px; margin-left: 8px; margin-right: 12px; border-radius: 50%; background: rgba(23,38,60,.20); border: 2px solid #17263C; }
      .driver-swatch { width: 18px; margin-left: 5px; margin-right: 9px; color: #C62D1B; font-size: 19px; line-height: 1; text-align: center; }
      .weather-swatch { width: 12px; height: 12px; margin-left: 8px; margin-right: 12px; transform: rotate(45deg); }
      .weather-swatch.core { background: #4F2D61; border: 2px solid #fff; box-shadow: 0 0 0 1px #4F2D61; }
      .weather-swatch.candidate { background: #fff; border: 2px solid #FF3621; }
      .legend-note { margin-top: 10px; padding-top: 9px; border-top: 1px solid #D9DEE5; color: #566274; line-height: 1.35; }
      .legend-source { margin-top: 7px; }
      .legend-source a { color: #4F2D61; font-weight: 600; text-decoration: none; }
      .demand-view-control { display: flex; margin-top: 10px !important; background: rgba(255,255,255,.96); }
      .demand-view-button { min-height: 30px; padding: 0 9px; color: #17263C; background: transparent; border: 0; border-right: 1px solid #D9DEE5; font-size: 11px; font-weight: 700; cursor: pointer; }
      .demand-view-button:last-child { border-right: 0; }
      .demand-view-button:hover, .demand-view-button:focus { color: #4F2D61; background: #F4F6F8; outline: none; }
      .demand-measure-control { display: flex; margin-top: 10px !important; background: rgba(255,255,255,.97); }
      .demand-measure-button { min-height: 34px; padding: 0 11px; color: #17263C; background: transparent; border: 0; border-right: 1px solid #D9DEE5; font-size: 11px; font-weight: 700; cursor: pointer; }
      .demand-measure-button:last-child { border-right: 0; }
      .demand-measure-button[aria-pressed="true"] { color: #fff; background: #4F2D61; }
      .demand-measure-button:hover, .demand-measure-button:focus { outline: 2px solid #4F2D61; outline-offset: -2px; }
      .weather-marker-wrap { position: relative; width: 22px; height: 22px; }
      .weather-marker { position: absolute; top: 2px; left: 2px; width: 18px; height: 18px; transform: rotate(45deg); box-sizing: border-box; }
      .weather-marker.core { background: #4F2D61; border: 3px solid #fff; box-shadow: 0 0 0 2px #4F2D61, 0 2px 6px rgba(23,38,60,.35); }
      .weather-marker.candidate { background: #fff; border: 3px solid #FF3621; box-shadow: 0 2px 6px rgba(23,38,60,.25); }
      .station-popup { min-width: 205px; color: #17263C; }
      .station-popup strong { font-size: 14px; }
      .popup-value { margin: 7px 0 4px; color: #4F2D61; font-size: 20px; font-weight: 700; }
      .popup-meta { color: #566274; font-size: 11px; line-height: 1.4; }
      .driver-marker { position: relative; width: 24px; height: 24px; color: #C62D1B; font-size: 23px; line-height: 24px; text-align: center; filter: drop-shadow(0 1px 2px rgba(23,38,60,.55)); }
      .driver-name, .weather-name { position: absolute; top: 1px; left: 27px; width: max-content; color: #17263C; background: rgba(255,255,255,.90); border-radius: 3px; padding: 2px 4px; font-size: 11px; font-weight: 800; line-height: 16px; white-space: nowrap; box-shadow: 0 1px 3px rgba(23,38,60,.16); }
      .driver-marker.label-left .driver-name { right: 27px; left: auto; }
      .driver-marker.label-above .driver-name { top: auto; bottom: 26px; left: 50%; transform: translateX(-50%); }
      .driver-marker.label-below .driver-name { top: 27px; left: 50%; transform: translateX(-50%); }
      .weather-marker-wrap.label-left .weather-name { right: 27px; left: auto; }
      @media (max-width: 640px) {
        .demand-map-title { right: 10px; left: 46px; max-width: none; }
        .demand-map-heading { font-size: 15px; }
        .demand-map-legend { right: 10px; bottom: 18px; left: 10px; width: auto; }
      }
    </style>
    """
    map_object.get_root().header.add_child(folium.Element(css))


def build_map(
    points_by_measure: dict[str, pd.DataFrame],
    *,
    measures: list[dict[str, Any]],
    selected_driver_codes: set[str],
    transmission_geojson: dict[str, Any],
    boundary_geojson: dict[str, Any],
    weather_locations: list[dict[str, Any]],
    map_config: dict[str, Any],
    source_page: str,
) -> folium.Map:
    default_measure = next(
        (measure for measure in measures if measure.get("default")), measures[0]
    )
    default_points = points_by_measure[default_measure["id"]]
    map_object = folium.Map(
        location=map_config["center"],
        zoom_start=map_config["zoom_start"],
        tiles=None,
        prefer_canvas=True,
    )
    focus_bounds, focus_audit = demand_focus_bounds(
        default_points,
        weather_locations,
        focus_quantile=float(map_config["focus_quantile"]),
        include_weather_locations=bool(map_config["focus_include_weather_locations"]),
    )
    weather_bounds, _ = demand_focus_bounds(
        default_points,
        weather_locations,
        focus_quantile=float(map_config["focus_quantile"]),
        include_weather_locations=True,
    )
    folium.TileLayer("CartoDB positron", name="Simple map", control=True).add_to(
        map_object
    )
    folium.TileLayer("OpenStreetMap", name="Street map", control=True).add_to(
        map_object
    )

    folium.GeoJson(
        boundary_geojson,
        name="Victoria outline",
        style_function=lambda _: {
            "color": "#17263C",
            "weight": 1.3,
            "opacity": 0.75,
            "fillOpacity": 0.0,
        },
    ).add_to(map_object)
    folium.GeoJson(
        transmission_geojson,
        name="Power lines",
        style_function=_transmission_style,
        tooltip=folium.GeoJsonTooltip(
            fields=["NAME", "CAPACITYKV", "OPERATIONALSTATUS"],
            aliases=["Line", "Voltage level (kV)", "Status"],
            sticky=False,
        ),
    ).add_to(map_object)

    measure_layers: list[tuple[dict[str, Any], folium.FeatureGroup]] = []
    for measure in measures:
        points = points_by_measure[measure["id"]]
        minimum = float(points["demand_mw"].min())
        maximum = float(points["demand_mw"].max())
        measure_layer = folium.FeatureGroup(
            name=measure["label"],
            show=bool(measure.get("default")),
            control=False,
        ).add_to(map_object)
        heat_data = [
            [
                float(row.latitude),
                float(row.longitude),
                float(row.demand_mw / maximum),
            ]
            for row in points.itertuples()
        ]
        HeatMap(
            heat_data,
            min_opacity=map_config["heat_min_opacity"],
            radius=map_config["heat_radius"],
            blur=map_config["heat_blur"],
            max_zoom=10,
            gradient={
                0.15: "#443983",
                0.45: "#21918c",
                0.72: "#fde725",
                1: "#d7191c",
            },
        ).add_to(measure_layer)
        for row in points.sort_values("demand_mw").itertuples():
            radius = demand_marker_radius(
                float(row.demand_mw),
                minimum_mw=minimum,
                maximum_mw=maximum,
                minimum_radius=float(map_config["marker_min_radius"]),
                maximum_radius=float(map_config["marker_max_radius"]),
            )
            popup = folium.Popup(
                f"""
                <div class="station-popup">
                  <strong>{html.escape(row.station_name)}</strong>
                  <div class="popup-value">{row.demand_mw:,.0f} MW</div>
                  <div class="popup-meta">{html.escape(measure["label"])}<br>
                  MW = electricity needed at one moment.</div>
                </div>
                """,
                max_width=260,
            )
            folium.CircleMarker(
                location=[row.latitude, row.longitude],
                radius=radius,
                color="#17263C",
                weight=1.4,
                fill=True,
                fill_color="#4F2D61",
                fill_opacity=0.30,
                tooltip=f"{row.station_name} · {row.demand_mw:,.0f} MW",
                popup=popup,
            ).add_to(measure_layer)
        measure_layers.append((measure, measure_layer))

    weather_layer = folium.FeatureGroup(
        name="Weather locations to compare", show=True
    ).add_to(map_object)
    for location in weather_locations:
        is_core = location["selection_class"].startswith("AEMO-supported")
        simple_class = "Used by AEMO forecasting" if is_core else "Regional comparison"
        simple_reason = (
            "AEMO uses this location when forecasting Victorian demand."
            if is_core
            else "Added to test whether regional weather improves the forecast."
        )
        icon = folium.DivIcon(
            html=weather_marker_html(location),
            icon_size=(22, 22),
            icon_anchor=(11, 11),
        )
        popup = folium.Popup(
            f"""
            <div class="station-popup">
              <strong>{html.escape(location["name"])}</strong>
              <div style="margin:7px 0 4px;font-weight:700;color:{"#4F2D61" if is_core else "#C62D1B"}">
                {simple_class}
              </div>
              <div class="popup-meta">{simple_reason}</div>
            </div>
            """,
            max_width=320,
        )
        folium.Marker(
            [location["latitude"], location["longitude"]],
            icon=icon,
            tooltip=f"{location['name']} · {simple_class}",
            popup=popup,
        ).add_to(weather_layer)

    selected_layer = folium.FeatureGroup(
        name=f"{len(selected_driver_codes)} selected demand locations", show=True
    ).add_to(map_object)
    selected_points = default_points.loc[
        default_points["station_code"].isin(selected_driver_codes)
    ].sort_values("demand_mw", ascending=False)
    label_positions = map_config.get("driver_label_positions", {})
    for row in selected_points.itertuples():
        label_position = label_positions.get(row.station_code, "label-right")
        seasonal_values = []
        for measure in measures:
            measure_points = points_by_measure[measure["id"]]
            match = measure_points.loc[
                measure_points["station_code"].eq(row.station_code), "demand_mw"
            ]
            if not match.empty:
                seasonal_values.append(
                    f"{html.escape(measure['short_label'])}: {float(match.iloc[0]):,.0f} MW"
                )
        popup = folium.Popup(
            f"""
            <div class="station-popup">
              <strong>{html.escape(row.station_name)}</strong>
              <div style="margin:7px 0 4px;color:#C62D1B;font-weight:700">Selected from the summer top seven</div>
              <div class="popup-meta">{"<br>".join(seasonal_values)}</div>
            </div>
            """,
            max_width=270,
        )
        icon = folium.DivIcon(
            html=(
                f'<div class="driver-marker {html.escape(label_position)}">★'
                f'<span class="driver-name">{html.escape(row.station_name)}</span></div>'
            ),
            icon_size=(24, 24),
            icon_anchor=(12, 12),
        )
        folium.Marker(
            [row.latitude, row.longitude],
            icon=icon,
            tooltip=f"Selected demand location · {row.station_name}",
            popup=popup,
            z_index_offset=1000,
        ).add_to(selected_layer)

    Fullscreen(position="topright").add_to(map_object)
    folium.LayerControl(collapsed=True, position="topright").add_to(map_object)
    _add_measure_controls(map_object, measure_layers)
    _add_view_controls(
        map_object,
        demand_bounds=focus_bounds,
        weather_bounds=weather_bounds,
        statewide_center=map_config["center"],
        statewide_zoom=int(map_config["zoom_start"]),
        max_focus_zoom=int(map_config["focus_max_zoom"]),
    )
    map_object.fit_bounds(
        focus_bounds,
        padding_top_left=map_config["focus_padding_top_left"],
        padding_bottom_right=map_config["focus_padding_bottom_right"],
        max_zoom=int(map_config["focus_max_zoom"]),
    )
    _add_map_chrome(
        map_object,
        default_measure_label=default_measure["label"],
        mapped_station_count=len(default_points),
        selected_driver_count=len(selected_driver_codes),
        source_page=source_page,
    )
    map_object.focus_audit = focus_audit
    return map_object


def run(config_path: Path, refresh: bool = False) -> dict[str, Any]:
    config = load_config(config_path)
    demand_config = config["sources"]["terminal_demand"]
    measures = demand_config["measures"]
    if not measures or sum(bool(measure.get("default")) for measure in measures) != 1:
        raise ValueError("Demand measures must contain exactly one default")
    files = fetch_sources(config, refresh=refresh)
    terminal_geojson = load_geojson(files["terminal_stations"])
    points_by_measure: dict[str, pd.DataFrame] = {}
    measure_audits: dict[str, dict[str, Any]] = {}
    unmatched_by_measure: dict[str, list[str]] = {}
    for measure in measures:
        raw = pd.read_excel(
            files["terminal_demand"],
            sheet_name=measure["sheet"],
            header=int(demand_config["header_row"]),
        )
        demand, selection_audit = prepare_terminal_demand(
            raw,
            year=int(measure["year"]),
            unit=demand_config["unit"],
            included_location_types=demand_config["included_location_types"],
            aliases=config["station_code_aliases"],
        )
        if "actual" not in selection_audit["value_column"].lower():
            raise ValueError(
                f"{measure['id']} must use an actual workbook column; "
                f"found {selection_audit['value_column']}"
            )
        points, unmatched = join_demand_to_terminal_stations(demand, terminal_geojson)
        points_by_measure[measure["id"]] = points
        measure_audits[measure["id"]] = {
            **selection_audit,
            "sheet": measure["sheet"],
            "label": measure["label"],
        }
        unmatched_by_measure[measure["id"]] = unmatched
    unmatched_codes = sorted(
        {code for codes in unmatched_by_measure.values() for code in codes}
    )
    if unmatched_codes:
        raise ValueError(
            f"Demand rows have no VicGrid coordinate match: {unmatched_by_measure}"
        )

    default_measure = next(measure for measure in measures if measure["default"])
    default_points = points_by_measure[default_measure["id"]]
    selected_count = int(config["map"]["selected_driver_count"])
    if selected_count < 1 or selected_count > len(default_points):
        raise ValueError("selected_driver_count must fit the available demand points")
    selected_driver_codes = set(
        default_points.nlargest(selected_count, "demand_mw")["station_code"]
    )
    map_object = build_map(
        points_by_measure,
        measures=measures,
        selected_driver_codes=selected_driver_codes,
        transmission_geojson=load_geojson(files["transmission_lines"]),
        boundary_geojson=load_geojson(files["victoria_boundary"]),
        weather_locations=config["weather_locations"],
        map_config=config["map"],
        source_page=demand_config["landing_page"],
    )

    output_dir = PROJECT_DIR / config["paths"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    map_path = output_dir / config["paths"]["map_html"]
    audit_path = output_dir / config["paths"]["audit_csv"]
    metadata_path = output_dir / config["paths"]["metadata_json"]
    map_object.save(map_path)
    audit_points = combine_measure_points(
        points_by_measure, selected_driver_codes
    ).sort_values(f"{default_measure['id']}_maximum_mw", ascending=False)
    audit_points.to_csv(audit_path, index=False)

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "measures": [
            {
                "id": measure["id"],
                "label": measure["label"],
                "year": int(measure["year"]),
                "season": measure["season"],
                "sheet": measure["sheet"],
                "mapped_terminal_stations": len(points_by_measure[measure["id"]]),
                "selection_audit": measure_audits[measure["id"]],
            }
            for measure in measures
        ],
        "interpretation": (
            "Summer and winter compare non-coincident terminal-station maxima. "
            "Circle size and heat use the same MW value. Values are not additive."
        ),
        "selected_demand_drivers": audit_points.loc[
            audit_points["selected_demand_driver"],
            ["station_code", "station_name"],
        ].to_dict("records"),
        "initial_view": map_object.focus_audit,
        "unmatched_terminal_codes": unmatched_by_measure,
        "source_files": {
            name: {
                "path": str(path.relative_to(PROJECT_DIR)),
                "sha256": sha256(path),
                "url": config["sources"][name]["url"],
            }
            for name, path in files.items()
        },
        "outputs": {
            "map_html": str(map_path.relative_to(PROJECT_DIR)),
            "audit_csv": str(audit_path.relative_to(PROJECT_DIR)),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the Victorian terminal-station demand and weather map."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_DIR / "config.json",
        help="Path to the JSON configuration file.",
    )
    parser.add_argument(
        "--refresh", action="store_true", help="Re-download all public source files."
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = run(
        arguments.config.resolve(),
        refresh=arguments.refresh,
    )
    print(json.dumps(result, indent=2))

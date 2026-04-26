"""
Flight Delay Prediction App — v2 with Weather Integration
Built with Shiny for Python
"""

from shiny import App, ui, render, reactive
import shinyswatch
import pandas as pd
import numpy as np
import pickle
import json
import sys
from pathlib import Path


ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "models"))
from feature_computation import compute_features, PREDICTOR_COLS, CATEGORICAL_COLS


MODELS_DIR = ROOT / "models"

with open(MODELS_DIR / "classifier.pkl", "rb") as f:
    classifier = pickle.load(f)
with open(MODELS_DIR / "calibrator.pkl", "rb") as f:
    calibrator = pickle.load(f)
with open(MODELS_DIR / "baseline_lookups.pkl", "rb") as f:
    baselines = pickle.load(f)
with open(MODELS_DIR / "shap_explainer.pkl", "rb") as f:
    explainer = pickle.load(f)
with open(MODELS_DIR / "model_metadata.json") as f:
    metadata = json.load(f)
with open(MODELS_DIR / "shap_metadata.json") as f:
    shap_meta = json.load(f)

us_airports = pd.read_parquet(MODELS_DIR / "us_airports.parquet")
weather_lookup = pd.read_parquet(MODELS_DIR / "weather_reference.parquet")

AIRPORTS = sorted(us_airports["iata"].dropna().unique().tolist())
CARRIERS = sorted(list(baselines["carrier_map"].keys()))

THRESHOLD = metadata.get("deployment_threshold", metadata.get("val_optimal_threshold", 0.15))


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lat2, lon1, lon2 = np.radians([lat1, lat2, lon1, lon2])
    d_lat, d_lon = lat2 - lat1, lon2 - lon1
    a = np.sin(d_lat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(d_lon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def estimate_distance(origin, dest):
    o = us_airports[us_airports["iata"] == origin]
    d = us_airports[us_airports["iata"] == dest]
    if len(o) == 0 or len(d) == 0:
        return 1000
    km = haversine_km(o.iloc[0]["lat"], o.iloc[0]["lon"],
                      d.iloc[0]["lat"], d.iloc[0]["lon"])
    return int(km * 0.621371)


app_ui = ui.page_fluid(
    ui.panel_title("✈️ Flight Delay Predictor"),
    ui.markdown(
        "Enter flight details to predict the probability of a departure delay "
        "(more than 15 minutes). Model trained on 4.5M+ US domestic flights "
        "with NOAA weather integration."
    ),
    ui.hr(),
    ui.layout_sidebar(
        ui.sidebar(
            ui.h4("Flight details"),
            ui.input_selectize("origin", "Origin airport",
                               choices=AIRPORTS, selected="JFK"),
            ui.input_selectize("dest", "Destination airport",
                               choices=AIRPORTS, selected="LAX"),
            ui.input_selectize("carrier", "Carrier",
                               choices=CARRIERS, selected="AA"),
            ui.input_date("fl_date", "Flight date", value="2025-06-15"),
            ui.input_slider("dep_hour", "Scheduled departure hour (24h)",
                            min=0, max=23, value=14, step=1),
            ui.input_action_button("predict", "Predict delay probability",
                                   class_="btn-primary btn-lg"),
            width=350,
        ),
        ui.card(
            ui.h3("Prediction"),
            ui.output_ui("prediction_display"),
        ),
        ui.card(
            ui.h3("Why this prediction?"),
            ui.output_ui("shap_explanation"),
        ),
        ui.card(
            ui.h3("Flight context"),
            ui.markdown(
                "_Weather shown is **historical typical** for this calendar day, "
                "based on NOAA 2024 observations — not a real-time forecast._"
            ),
            ui.output_ui("flight_context"),
        ),
    ),
    ui.hr(),
    ui.markdown(
        f"**Model:** LightGBM with isotonic calibration. "
        f"ROC-AUC 0.71 (CV) / 0.66 (test). "
        f"Decision threshold: {THRESHOLD:.2f}. "
        f"Trained on Jan–Aug 2024 BTS data with NOAA weather."
    ),
    ui.markdown(
        """
        ---
        ### About this app
        
        This is a **research demo**, not a production forecasting service.
        Weather inputs come from a static lookup of NOAA 2024 daily observations,
        averaged by airport and calendar day. The app answers questions like
        *"What is the typical delay risk for a JFK→LAX flight on a mid-June afternoon?"*
        rather than *"What is the delay probability for my flight tomorrow?"*
        
        The model architecture supports real-time deployment without retraining —
        replacing the static weather lookup with a forecast API call (OpenWeatherMap,
        Tomorrow.io, NOAA NDFD) is sufficient. The static lookup is used here for
        offline reproducibility and to keep the demo dependency-free.
        """
    ),
    theme=shinyswatch.theme.cosmo,
)


# Server

def server(input, output, session):

    @reactive.Calc
    @reactive.event(input.predict)
    def prediction():
        origin = input.origin()
        dest = input.dest()
        carrier = input.carrier()
        fl_date = input.fl_date()
        dep_time = input.dep_hour() * 100

        distance = estimate_distance(origin, dest)
        crs_elapsed = int(distance / 500 * 60 + 30)

        X = compute_features(
            origin=origin, dest=dest, carrier=carrier,
            fl_date=fl_date, scheduled_dep_time=dep_time,
            distance=distance, crs_elapsed_time=crs_elapsed,
            baselines=baselines, weather_lookup=weather_lookup,
        )

        raw_prob = classifier.predict(X, num_iteration=classifier.best_iteration)[0]
        calibrated_prob = calibrator.transform([raw_prob])[0]

        shap_vals = explainer.shap_values(X)[0]

        return {
            "X": X, "raw_prob": raw_prob, "calibrated_prob": calibrated_prob,
            "shap_values": shap_vals, "distance": distance,
            "origin": origin, "dest": dest, "carrier": carrier,
        }

    @output
    @render.ui
    def prediction_display():
        p = prediction()
        if p is None:
            return ui.markdown("_Click **Predict** to get a delay probability._")

        prob = p["calibrated_prob"]
        pct = prob * 100

        if prob < 0.15:
            color = "#27AE60"
            label = "LOW RISK"
        elif prob < 0.30:
            color = "#F39C12"
            label = "MODERATE RISK"
        else:
            color = "#E74C3C"
            label = "HIGH RISK"

        flagged = "⚠️ Flagged as likely delayed" if prob >= THRESHOLD else "✓ Not flagged"

        return ui.HTML(f"""
            <div style="text-align:center;padding:20px;">
                <div style="font-size:72px;font-weight:bold;color:{color};">
                    {pct:.1f}%
                </div>
                <div style="font-size:24px;color:{color};font-weight:bold;">
                    {label}
                </div>
                <div style="font-size:16px;margin-top:10px;color:#555;">
                    {flagged}
                </div>
            </div>
        """)

    @output
    @render.ui
    def shap_explanation():
        p = prediction()
        if p is None:
            return ui.markdown("_Click **Predict** to see the reasoning._")

        shap_vals = p["shap_values"]
        feat_names = PREDICTOR_COLS

        impacts = list(zip(feat_names, shap_vals))
        impacts.sort(key=lambda x: abs(x[1]), reverse=True)
        top5 = impacts[:5]

        def friendly(name):
            mapping = {
                "scheduled_dep_hour": "Departure hour",
                "op_unique_carrier": "Carrier",
                "origin_prcp": "Rain at origin",
                "dest_prcp": "Rain at destination",
                "origin_pres": "Pressure at origin",
                "route_avg_dep_delay": "Route's history",
                "origin_avg_dep_delay": "Origin airport's history",
                "carrier_avg_dep_delay": "Carrier's history",
                "day_of_week": "Day of week",
                "month": "Month",
                "origin_wspd": "Wind at origin",
                "origin_bad_weather": "Bad weather at origin",
                "origin_tmin": "Cold at origin",
                "dest_tavg": "Temperature at destination",
                "origin": "Origin airport",
                "dest": "Destination airport",
                "route_id": "Specific route",
                "season": "Season",
            }
            return mapping.get(name, name)

        lines = []
        for name, val in top5:
            direction = "↑ pushes toward delay" if val > 0 else "↓ pushes toward on-time"
            color = "#E74C3C" if val > 0 else "#27AE60"
            lines.append(f"""
                <li style="margin:8px 0;">
                    <strong>{friendly(name)}</strong>:
                    <span style="color:{color};font-weight:bold;">{direction}</span>
                    <span style="color:#888;"> ({val:+.3f})</span>
                </li>
            """)

        return ui.HTML(f"""
            <p>Top factors driving this prediction:</p>
            <ol>{"".join(lines)}</ol>
            <p style="color:#888;font-size:12px;margin-top:15px;">
                Values are SHAP contributions (log-odds scale).
                Positive values push prediction toward "delayed".
            </p>
        """)

    @output
    @render.ui
    def flight_context():
        p = prediction()
        if p is None:
            return ui.markdown("_Click **Predict** to see flight context._")

        origin = p["origin"]; dest = p["dest"]; carrier = p["carrier"]
        gm = baselines["global_mean"]

        origin_baseline = baselines["origin_map"].get(origin, gm)
        dest_baseline = baselines["origin_map"].get(dest, gm)
        carrier_baseline = baselines["carrier_map"].get(carrier, gm)

        X = p["X"].iloc[0]

        return ui.HTML(f"""
            <table style="width:100%;font-size:14px;">
                <tr><td><strong>Distance:</strong></td><td>{p['distance']:,} miles</td></tr>
                <tr><td><strong>Origin avg delay:</strong></td>
                    <td>{origin_baseline:.1f} min historically</td></tr>
                <tr><td><strong>Destination avg delay:</strong></td>
                    <td>{dest_baseline:.1f} min historically</td></tr>
                <tr><td><strong>Carrier avg delay:</strong></td>
                    <td>{carrier_baseline:.1f} min historically</td></tr>
                <tr><td colspan="2"><hr></td></tr>
                <tr><td><strong>Origin weather:</strong></td>
                    <td>{X['origin_tavg']:.1f}°C,
                        {X['origin_prcp']:.1f}mm rain,
                        {X['origin_wspd']:.0f} km/h wind
                        {'❄️' if X['origin_snow'] > 0 else ''}</td></tr>
                <tr><td><strong>Destination weather:</strong></td>
                    <td>{X['dest_tavg']:.1f}°C,
                        {X['dest_prcp']:.1f}mm rain,
                        {X['dest_wspd']:.0f} km/h wind</td></tr>
            </table>
        """)


app = App(app_ui, server)
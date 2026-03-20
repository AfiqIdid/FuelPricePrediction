import html
import os
from datetime import datetime, timedelta
from pathlib import Path

import googlemaps
import joblib
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from streamlit_searchbox import st_searchbox


load_dotenv()

st.set_page_config(
    page_title="Fuel Trip Cost Predictor",
    layout="wide",
    page_icon="car",
)

try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except (KeyError, FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

gmaps = googlemaps.Client(key=GOOGLE_API_KEY)

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "fuelconsumption3.csv"
MODEL_PATH = BASE_DIR / "fuel_model.pkl"
FUEL_ENCODER_PATH = BASE_DIR / "fuel_encoder.pkl"
CLASS_ENCODER_PATH = BASE_DIR / "class_encoder.pkl"


def inject_styles():
    st.markdown(
        """
        <style>
        
        :root {
            --background: #0b1120;
            --foreground: #f7f8fb;
            --card: rgba(15, 23, 42, 0.78);
            --card-foreground: #f8fafc;
            --muted: rgba(148, 163, 184, 0.16);
            --muted-foreground: #94a3b8;
            --accent: rgba(251, 146, 60, 0.12);
            --accent-strong: #f97316;
            --accent-soft: #fb923c;
            --border: rgba(255, 255, 255, 0.11);
            --ring: rgba(251, 146, 60, 0.38);
            --good: #4ade80;
            --warn: #facc15;
            --bad: #fb7185;
            --radius: 22px;
        }

        .stApp {
            color: var(--foreground);
            background:
                radial-gradient(circle at top left, rgba(251, 146, 60, 0.24), transparent 24%),
                radial-gradient(circle at top right, rgba(34, 197, 94, 0.12), transparent 20%),
                linear-gradient(145deg, #180d08 0%, #0f172a 42%, #020617 100%);
        }

        [data-testid="stAppViewContainer"] > .main {
            background: transparent;
        }

        .block-container {
            max-width: 1180px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        h1, h2, h3, h4, label, p, span, div {
            font-family: "Segoe UI", "Helvetica Neue", sans-serif;
        }

        .hero-shell {
            position: relative;
            overflow: hidden;
            padding: 2rem;
            margin-bottom: 1.5rem;
            border: 1px solid var(--border);
            border-radius: 30px;
            background:
                radial-gradient(circle at top right, rgba(251, 146, 60, 0.18), transparent 28%),
                linear-gradient(135deg, rgba(15, 23, 42, 0.94), rgba(30, 41, 59, 0.84));
            box-shadow: 0 30px 90px rgba(2, 6, 23, 0.42);
        }

        .hero-shell::after {
            content: "";
            position: absolute;
            inset: auto -4rem -5rem auto;
            width: 15rem;
            height: 15rem;
            border-radius: 999px;
            background: radial-gradient(circle, rgba(249, 115, 22, 0.32), transparent 68%);
            filter: blur(8px);
        }

        .hero-kicker {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            margin-bottom: 1rem;
            padding: 0.45rem 0.8rem;
            border-radius: 999px;
            background: rgba(251, 146, 60, 0.12);
            border: 1px solid rgba(251, 146, 60, 0.18);
            color: #fdba74;
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .hero-title {
            margin: 0;
            font-size: clamp(2.3rem, 4vw, 3.6rem);
            line-height: 1.02;
            font-weight: 800;
            letter-spacing: -0.04em;
            color: #f8fafc;
        }

        .hero-title span {
            background: linear-gradient(135deg, #fb923c, #fdba74);
            -webkit-background-clip: text;
            color: transparent;
        }

        .hero-copy {
            max-width: 42rem;
            margin: 1rem 0 1.4rem;
            color: #cbd5e1;
            font-size: 1rem;
            line-height: 1.7;
        }

        .hero-chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.7rem;
        }

        .hero-chip {
            padding: 0.65rem 0.9rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.08);
            color: #e2e8f0;
            font-size: 0.9rem;
        }

        .section-card {
            margin-bottom: 1.25rem;
            padding: 1.35rem;
            border-radius: 26px;
            border: 1px solid var(--border);
            background:
                linear-gradient(180deg, rgba(15, 23, 42, 0.8), rgba(15, 23, 42, 0.62));
            box-shadow: 0 18px 40px rgba(2, 6, 23, 0.24);
            backdrop-filter: blur(18px);
        }

        .section-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            margin-bottom: 1rem;
        }

        .section-title {
            margin: 0;
            color: #f8fafc;
            font-size: 1.15rem;
            font-weight: 700;
        }

        .section-subtitle {
            margin: 0.25rem 0 0;
            color: #94a3b8;
            font-size: 0.92rem;
        }

        .mini-badge {
            padding: 0.45rem 0.75rem;
            border-radius: 999px;
            background: rgba(30, 41, 59, 0.95);
            border: 1px solid rgba(255, 255, 255, 0.08);
            color: #cbd5e1;
            font-size: 0.78rem;
            font-weight: 600;
        }

        div[data-testid="stSelectbox"] label,
        div[data-testid="stSlider"] label,
        div[data-testid="stTimeInput"] label,
        div[data-testid="stNumberInput"] label {
            color: #e2e8f0 !important;
            font-weight: 700 !important;
            letter-spacing: 0.01em;
        }

        .field-label {
            margin-bottom: 0.45rem;
            color: #e2e8f0;
            font-size: 1rem;
            font-weight: 700;
            letter-spacing: 0.01em;
        }

        div[data-testid="stSelectbox"] > div[data-baseweb="select"] > div,
        div[data-testid="stTimeInput"] input,
        div[data-testid="stNumberInput"] input,
        .stSearchbox > div,
        .stSearchbox [data-baseweb="base-input"],
        .stSearchbox [data-baseweb="input"],
        .stSearchbox input,
        .stSearchbox div[data-baseweb="input"] > div {
            border-radius: 18px !important;
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            background: rgba(15, 23, 42, 0.85) !important;
            color: #f8fafc !important;
            min-height: 3.25rem !important;
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.03);
        }

        .stSearchbox input {
            padding-left: 0.95rem !important;
            padding-right: 0.95rem !important;
            font-size: 1rem !important;
        }

        div[data-testid="stSelectbox"] > div[data-baseweb="select"] > div:hover,
        div[data-testid="stTimeInput"] input:hover,
        div[data-testid="stNumberInput"] input:hover,
        .stSearchbox > div:hover,
        .stSearchbox [data-baseweb="base-input"]:hover,
        .stSearchbox input:hover {
            border-color: rgba(251, 146, 60, 0.26) !important;
        }

        div[data-testid="stSelectbox"] > div[data-baseweb="select"] > div:focus-within,
        div[data-testid="stTimeInput"] input:focus,
        div[data-testid="stNumberInput"] input:focus,
        .stSearchbox > div:focus-within,
        .stSearchbox [data-baseweb="base-input"]:focus-within,
        .stSearchbox input:focus {
            border-color: rgba(251, 146, 60, 0.45) !important;
            box-shadow: 0 0 0 4px rgba(249, 115, 22, 0.14) !important;
        }

        .stSearchbox {
            width: 100%;
        }

        .stSearchbox input::placeholder,
        div[data-testid="stTimeInput"] input::placeholder {
            color: #64748b !important;
        }

        div[data-testid="stSlider"] > div[data-baseweb="slider"] {
            padding-top: 0.9rem;
            padding-bottom: 0.6rem;
        }

        div[data-testid="stSlider"] [role="slider"] {
            background: linear-gradient(135deg, #f97316, #fb923c) !important;
            border: 0 !important;
            box-shadow: 0 2px 10px rgba(249, 115, 22, 0.35) !important;
        }

        div[data-testid="stSlider"] [data-testid="stTickBar"] {
            background: rgba(255, 255, 255, 0.16) !important;
        }

        div.stButton > button {
            width: 100%;
            min-height: 3.55rem;
            border: 0;
            border-radius: 1.2rem;
            background: linear-gradient(135deg, #ea580c, #fb923c);
            color: #fff7ed;
            font-size: 1rem;
            font-weight: 800;
            letter-spacing: 0.01em;
            box-shadow: 0 18px 35px rgba(249, 115, 22, 0.28);
            transition: transform 0.14s ease, box-shadow 0.14s ease, filter 0.14s ease;
        }

        div.stButton > button:hover {
            transform: translateY(-1px);
            filter: brightness(1.04);
            box-shadow: 0 22px 45px rgba(249, 115, 22, 0.34);
        }

        div.stButton > button:active {
            transform: scale(0.985);
        }

        [data-testid="stTabs"] [data-baseweb="tab-list"] {
            gap: 0.65rem;
            background: rgba(15, 23, 42, 0.7);
            border: 1px solid var(--border);
            border-radius: 999px;
            padding: 0.35rem;
        }

        [data-testid="stTabs"] [data-baseweb="tab"] {
            min-height: 2.85rem;
            border-radius: 999px;
            color: #cbd5e1;
            background: transparent;
            font-weight: 700;
        }

        [data-testid="stTabs"] [aria-selected="true"] {
            background: linear-gradient(135deg, rgba(249, 115, 22, 0.22), rgba(251, 146, 60, 0.26));
            color: #fff7ed !important;
        }

        .summary-card {
            position: relative;
            overflow: hidden;
            padding: 1.4rem;
            border-radius: 26px;
            border: 1px solid rgba(255, 255, 255, 0.09);
            background:
                radial-gradient(circle at top right, rgba(249, 115, 22, 0.16), transparent 26%),
                linear-gradient(135deg, rgba(15, 23, 42, 0.92), rgba(30, 41, 59, 0.82));
            box-shadow: 0 18px 42px rgba(2, 6, 23, 0.32);
        }

        .summary-label {
            color: #94a3b8;
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.09em;
            font-weight: 700;
        }

        .summary-route {
            margin: 0.45rem 0 1rem;
            color: #f8fafc;
            font-size: 1.1rem;
            font-weight: 700;
        }

        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 0.9rem;
        }

        .summary-pill {
            padding: 0.9rem 1rem;
            border-radius: 18px;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.06);
        }

        .summary-pill strong {
            display: block;
            margin-top: 0.35rem;
            color: #f8fafc;
            font-size: 1rem;
        }

        .metrics-row {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
            gap: 1rem;
            margin: 1rem 0 1.1rem;
        }

        .metric-card {
            padding: 1.1rem;
            border-radius: 22px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            background: rgba(15, 23, 42, 0.76);
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.03);
        }

        .metric-eyebrow {
            color: #94a3b8;
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.09em;
            font-weight: 700;
        }

        .metric-number {
            margin: 0.3rem 0;
            color: #f8fafc;
            font-size: 2rem;
            font-weight: 800;
            line-height: 1;
        }

        .metric-note {
            color: #cbd5e1;
            font-size: 0.92rem;
        }

        .cost-card {
            position: relative;
            overflow: hidden;
            padding: 1.35rem;
            border-radius: 26px;
            border: 1px solid rgba(96, 165, 250, 0.22);
            background: linear-gradient(135deg, #1d4ed8, #2563eb, #0f172a);
            box-shadow: 0 22px 48px rgba(37, 99, 235, 0.28);
        }

        .cost-card::after {
            content: "";
            position: absolute;
            top: -2rem;
            right: -2rem;
            width: 10rem;
            height: 10rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.14);
            filter: blur(10px);
        }

        .cost-title {
            position: relative;
            color: #dbeafe;
            font-size: 0.84rem;
            text-transform: uppercase;
            letter-spacing: 0.11em;
            font-weight: 800;
        }

        .cost-value {
            position: relative;
            margin: 0.45rem 0 0.65rem;
            color: white;
            font-size: clamp(2.2rem, 4vw, 3.2rem);
            font-weight: 900;
            line-height: 1;
        }

        .cost-copy {
            position: relative;
            color: rgba(219, 234, 254, 0.96);
            font-size: 0.98rem;
        }

        .progress-shell {
            position: relative;
            margin-top: 1rem;
        }

        .progress-meta {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            color: #dbeafe;
            font-size: 0.9rem;
            margin-bottom: 0.45rem;
        }

        .progress-bar {
            height: 0.7rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.18);
            overflow: hidden;
        }

        .progress-bar span {
            display: block;
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, #4ade80, #bfdbfe);
        }

        .table-shell {
            padding: 1rem;
            border-radius: 24px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            background: rgba(15, 23, 42, 0.78);
        }

        [data-testid="stTable"] table,
        [data-testid="stDataFrame"] {
            border-radius: 18px;
            overflow: hidden;
        }

        [data-testid="stTable"] tbody tr:nth-child(odd) {
            background: rgba(255, 255, 255, 0.02);
        }

        .empty-state {
            padding: 1.6rem;
            border-radius: 24px;
            border: 1px dashed rgba(255, 255, 255, 0.16);
            background: rgba(15, 23, 42, 0.62);
            color: #cbd5e1;
            text-align: center;
        }

        @media (max-width: 900px) {
            .hero-shell {
                padding: 1.4rem;
                border-radius: 26px;
            }

            .section-header {
                align-items: flex-start;
                flex-direction: column;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def pick_display_value(values):
    unique_values = sorted({str(value).strip() for value in values if str(value).strip()})
    preferred_values = [value for value in unique_values if not value.isupper()]
    return preferred_values[0] if preferred_values else unique_values[0]


@st.cache_data
def load_csv_data(_csv_mtime_ns):
    df = pd.read_csv(CSV_PATH, low_memory=False)
    cols = [
        "YEAR",
        "MAKE",
        "MODEL",
        "VEHICLE CLASS",
        "FUEL",
        "ENGINE SIZE",
        "CYLINDERS",
        "COMB (L/100 km)",
    ]
    df = df[cols].dropna().copy()
    for col in ["MAKE", "MODEL", "VEHICLE CLASS", "FUEL"]:
        df[col] = df[col].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
    df["MAKE_KEY"] = df["MAKE"].str.upper()
    df["MODEL_KEY"] = df["MODEL"].str.upper()
    df["MAKE_DISPLAY"] = df.groupby("MAKE_KEY")["MAKE"].transform(pick_display_value)
    df["MODEL_DISPLAY"] = df.groupby(["MAKE_KEY", "MODEL_KEY"])["MODEL"].transform(pick_display_value)
    df["YEAR"] = df["YEAR"].astype(int)
    return df.reset_index(drop=True)


@st.cache_resource
def load_ai_model():
    try:
        model = joblib.load(MODEL_PATH)
        le_fuel = joblib.load(FUEL_ENCODER_PATH)
        le_class = joblib.load(CLASS_ENCODER_PATH)
        return model, le_fuel, le_class
    except Exception:
        return None, None, None


def search_places(searchterm: str):
    if not searchterm:
        return []
    try:
        results = gmaps.places_autocomplete(
            input_text=searchterm,
            components={"country": "MY"},
        )
        return [res["description"] for res in results]
    except Exception:
        return []


def search_brands(searchterm: str):
    if len(searchterm.strip()) < 2:
        return []
    matches = brand_options[
        brand_options["MAKE_DISPLAY"].str.contains(searchterm, case=False, na=False)
    ]["MAKE_DISPLAY"].tolist()
    return matches[:12]


SEARCHBOX_STYLE = {
    "searchbox": {
        "optionEmpty": "hidden",
    }
}


def sync_fuel_price_from_slider():
    value = round(float(st.session_state.fuel_price_slider), 2)
    st.session_state.fuel_price = value
    st.session_state.fuel_price_input = value


def sync_fuel_price_from_input():
    value = round(float(st.session_state.fuel_price_input), 2)
    value = min(10.0, max(1.0, value))
    st.session_state.fuel_price = value
    st.session_state.fuel_price_slider = value
    st.session_state.fuel_price_input = value


def nudge_fuel_price(delta):
    value = round(st.session_state.fuel_price + delta, 2)
    value = min(10.0, max(1.0, value))
    st.session_state.fuel_price = value
    st.session_state.fuel_price_slider = value
    st.session_state.fuel_price_input = value


def get_route_info(start, end, dep_time: datetime):
    try:
        directions = gmaps.directions(
            start,
            end,
            mode="driving",
            departure_time=dep_time,
            traffic_model="best_guess",
        )
        if not directions:
            return None
        leg = directions[0]["legs"][0]
        dur_norm = leg["duration"]["value"] / 60
        dur_traffic = leg.get("duration_in_traffic", leg["duration"])["value"] / 60
        return {
            "dist_km": leg["distance"]["value"] / 1000,
            "duration_min": dur_norm,
            "traffic_min": dur_traffic,
            "path": [
                {"lat": p["lat"], "lon": p["lng"]}
                for p in googlemaps.convert.decode_polyline(
                    directions[0]["overview_polyline"]["points"]
                )
            ],
            "start_addr": leg["start_address"],
            "end_addr": leg["end_address"],
        }
    except Exception as exc:
        st.error(f"Maps Error: {exc}")
        return None


def get_traffic_level(traffic_ratio):
    if traffic_ratio >= 1.4:
        return "Heavy", "var(--bad)"
    if traffic_ratio >= 1.15:
        return "Moderate", "var(--warn)"
    return "Light", "var(--good)"


def render_result_panels(route, selected_car, year_val, fuel_price, model, le_fuel, le_class):
    actual_l_100km = float(selected_car["COMB (L/100 km)"])
    ai_l_100km = None

    try:
        if model is None or le_fuel is None or le_class is None:
            raise ValueError("model assets unavailable")
        f_enc = le_fuel.transform([selected_car["FUEL"]])[0]
        c_enc = le_class.transform([selected_car["VEHICLE CLASS"]])[0]
        ai_l_100km = model.predict(
            [[
                selected_car["ENGINE SIZE"],
                selected_car["CYLINDERS"],
                year_val,
                f_enc,
                c_enc,
            ]]
        )[0]
    except Exception:
        ai_l_100km = None

    # Use the selected vehicle's actual combined consumption first.
    # The AI model is kept as a fallback/reference instead of overriding known car data.
    l_100km_pred = actual_l_100km if actual_l_100km > 0 else ai_l_100km
    if l_100km_pred is None or l_100km_pred <= 0:
        l_100km_pred = actual_l_100km

    base_liters = (route["dist_km"] / 100) * l_100km_pred
    traffic_ratio = route["traffic_min"] / max(route["duration_min"], 1)
    penalty = 1 + ((traffic_ratio - 1) * 0.5)
    final_liters = base_liters * penalty
    final_cost = final_liters * fuel_price
    base_cost = base_liters * fuel_price
    traffic_extra = final_cost - (base_liters * fuel_price)
    traffic_level, traffic_color = get_traffic_level(traffic_ratio)
    efficiency_score = max(15, min(92, int(115 - (l_100km_pred * 7))))
    prediction_note = (
        f"Actual car data: {actual_l_100km:.1f} L/100 km"
        if ai_l_100km is None
        else f"Actual car data: {actual_l_100km:.1f} L/100 km | AI reference: {ai_l_100km:.1f} L/100 km"
    )

    summary_html = f"""
        <div class="summary-card">
            <div class="summary-label">Trip Summary</div>
            <div class="summary-grid">
                <div class="summary-pill">
                    <span class="summary-label">Vehicle</span>
                    <strong>{html.escape(selected_car["MAKE_DISPLAY"])} {html.escape(selected_car["MODEL_DISPLAY"])} {year_val}</strong>
                </div>
                <div class="summary-pill">
                    <span class="summary-label">Departure</span>
                    <strong>{st.session_state.departure_time.strftime("%H:%M")}</strong>
                </div>
                <div class="summary-pill">
                    <span class="summary-label">Traffic</span>
                    <strong style="color:{traffic_color};">{traffic_level}</strong>
                </div>
            </div>
        </div>
    """
    st.markdown(summary_html, unsafe_allow_html=True)

    metrics_html = f"""
        <div class="metrics-row">
            <div class="metric-card">
                <div class="metric-eyebrow">Distance</div>
                <div class="metric-number">{route["dist_km"]:.1f}</div>
                <div class="metric-note">Kilometers on this route</div>
            </div>
            <div class="metric-card">
                <div class="metric-eyebrow">Travel Time</div>
                <div class="metric-number">{route["traffic_min"]:.0f}</div>
                <div class="metric-note">Minutes with live traffic</div>
            </div>
            <div class="metric-card">
                <div class="metric-eyebrow">Fuel Needed</div>
                <div class="metric-number">{final_liters:.1f}</div>
                <div class="metric-note">Liters after traffic penalty</div>
            </div>
        </div>
    """
    st.markdown(metrics_html, unsafe_allow_html=True)

    cost_html = f"""
        <div class="cost-card">
            <div class="cost-title">Estimated Trip Cost</div>
            <div class="cost-value">RM {final_cost:.2f}</div>
            <div class="cost-copy">
                Base fuel cost: RM {base_cost:.2f}<br/>
                Traffic impact: RM {traffic_extra:.2f}<br/>
                {prediction_note}
            </div>
        </div>
    """
    st.markdown(cost_html, unsafe_allow_html=True)

    return final_cost


inject_styles()
vehicle_df = load_csv_data(CSV_PATH.stat().st_mtime_ns if CSV_PATH.exists() else 0)
model, le_fuel, le_class = load_ai_model()

if "history" not in st.session_state:
    st.session_state.history = []
if "departure_time" not in st.session_state:
    st.session_state.departure_time = datetime.now().time()
if "latest_nav_url" not in st.session_state:
    st.session_state.latest_nav_url = None
if "fuel_price" not in st.session_state:
    st.session_state.fuel_price = 1.99
if "fuel_price_slider" not in st.session_state:
    st.session_state.fuel_price_slider = st.session_state.fuel_price
if "fuel_price_input" not in st.session_state:
    st.session_state.fuel_price_input = st.session_state.fuel_price

st.markdown(
    """
    <div class="hero-shell">
        <div class="hero-kicker">Route intelligence</div>
        <h1 class="hero-title">Fuel Trip <span>Cost Predictor</span></h1>
        <p class="hero-copy">
            Plan a drive, check traffic-aware fuel cost, and compare the impact of vehicle choice,
            time of departure, and current pump price in one place.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

if vehicle_df.empty:
    st.error(f"No vehicle data could be loaded from {CSV_PATH.name}.")
    st.stop()

brand_options = (
    vehicle_df[["MAKE_KEY", "MAKE_DISPLAY"]]
    .drop_duplicates()
    .sort_values("MAKE_DISPLAY")
)
brand_keys = brand_options["MAKE_KEY"].tolist()
st.markdown(
    """
    <div class="section-card">
        <div class="section-header">
            <div>
                <p class="section-title">1. Choose Vehicle</p>
                <p class="section-subtitle">Choose the brand, model, and year that should drive the prediction.</p>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

vehicle_col1, vehicle_col2, vehicle_col3 = st.columns(3)
with vehicle_col1:
    st.markdown('<div class="field-label">Brand</div>', unsafe_allow_html=True)
    brand_display = st_searchbox(
        search_brands,
        key="brand_search",
        placeholder="Search brand",
        style_overrides=SEARCHBOX_STYLE,
    )
    brand = None
    if brand_display:
        brand_match = brand_options.loc[
            brand_options["MAKE_DISPLAY"].str.casefold() == brand_display.casefold(),
            "MAKE_KEY",
        ]
        if not brand_match.empty:
            brand = brand_match.iloc[0]
with vehicle_col2:
    if brand:
        brand_models = (
            vehicle_df.loc[vehicle_df["MAKE_KEY"] == brand, ["MODEL_KEY", "MODEL_DISPLAY"]]
            .drop_duplicates()
            .sort_values("MODEL_DISPLAY")
        )
        model_keys = brand_models["MODEL_KEY"].tolist()
        model_name = st.selectbox(
            "Model",
            model_keys,
            format_func=lambda key: brand_models.loc[
                brand_models["MODEL_KEY"] == key, "MODEL_DISPLAY"
            ].iloc[0],
        )
    else:
        st.selectbox("Model", ["Select a brand first"], index=0, disabled=True)
        brand_models = pd.DataFrame(columns=["MODEL_KEY", "MODEL_DISPLAY"])
        model_name = None
with vehicle_col3:
    if brand and model_name:
        matching_rows = vehicle_df[
            (vehicle_df["MAKE_KEY"] == brand) & (vehicle_df["MODEL_KEY"] == model_name)
        ]
        year_val = st.selectbox("Year", sorted(matching_rows["YEAR"].unique(), reverse=True))
    else:
        st.selectbox("Year", ["Select a model first"], index=0, disabled=True)
        matching_rows = pd.DataFrame()
        year_val = None

selected_car = None if matching_rows.empty or year_val is None else matching_rows[matching_rows["YEAR"] == year_val].iloc[0]

calc_tab, hist_tab = st.tabs(["Trip Planner", "Trip History"])

with calc_tab:
    st.markdown(
        """
        <div class="section-card">
            <div class="section-header">
                <div>
                    <p class="section-title">2. Choose Location</p>
                    <p class="section-subtitle">Enter your starting point, destination, and departure time.</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    location_col1, location_col2 = st.columns(2)
    with location_col1:
        st.markdown('<div class="mini-badge" style="margin-bottom:0.6rem;">Starting point</div>', unsafe_allow_html=True)
        start_addr = st_searchbox(
            search_places,
            key="start_search",
            placeholder="Search starting location",
            style_overrides=SEARCHBOX_STYLE,
        )
    with location_col2:
        st.markdown('<div class="mini-badge" style="margin-bottom:0.6rem;">Destination</div>', unsafe_allow_html=True)
        end_addr = st_searchbox(
            search_places,
            key="end_search",
            placeholder="Search destination",
            style_overrides=SEARCHBOX_STYLE,
        )

    chosen_time = st.time_input(
        "Departure Time",
        value=st.session_state.departure_time,
    )
    st.session_state.departure_time = chosen_time

    st.markdown(
        """
        <div class="section-card" style="margin-top:1rem;">
            <div class="section-header">
                <div>
                    <p class="section-title">3. Fuel Settings</p>
                    <p class="section-subtitle">Adjust the fuel price with slider, manual input, or quick step buttons.</p>
                </div>
                <div class="mini-badge">RM 1.00 to RM 10.00</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    fuel_controls_col1, fuel_controls_col2, fuel_controls_col3 = st.columns([1, 1.4, 1])
    with fuel_controls_col1:
        minus_col, plus_col = st.columns(2)

    with fuel_controls_col2:
        st.number_input(
            "Fuel Price (RM/L)",
            min_value=1.00,
            max_value=10.00,
            step=0.01,
            key="fuel_price_input",
            on_change=sync_fuel_price_from_input,
        )
    with fuel_controls_col3:
        st.markdown(
            f"""
            """,
            unsafe_allow_html=True,
        )

    st.slider(
        "Adjust Fuel Price",
        min_value=1.00,
        max_value=10.00,
        step=0.01,
        key="fuel_price_slider",
        on_change=sync_fuel_price_from_slider,
    )
    fuel_price = st.session_state.fuel_price

    run_calc = st.button("Calculate Trip Cost")

    if run_calc:
        if selected_car is None:
            st.session_state.latest_nav_url = None
            st.warning("Please search and select a brand, model, and year first.")
        elif start_addr and end_addr:
            now = datetime.now()
            dep_datetime = datetime.combine(now.date(), chosen_time)
            if dep_datetime < now:
                dep_datetime += timedelta(days=1)

            with st.spinner("Calculating route and fuel estimate..."):
                route = get_route_info(start_addr, end_addr, dep_datetime)

            if route:
                st.session_state.latest_nav_url = (
                    f"https://www.google.com/maps/dir/?api=1"
                    f"&origin={start_addr}&destination={end_addr}&travelmode=driving"
                ).replace(" ", "+")
                final_cost = render_result_panels(
                    route,
                    selected_car,
                    year_val,
                    fuel_price,
                    model,
                    le_fuel,
                    le_class,
                )
                st.session_state.history.append(
                    {
                        "Date": dep_datetime.strftime("%d/%m %H:%M"),
                        "Vehicle": f"{selected_car['MAKE_DISPLAY']} {selected_car['MODEL_DISPLAY']} {year_val}",
                        "From": start_addr,
                        "To": end_addr,
                        "Cost": f"RM {final_cost:.2f}",
                    }
                )
            else:
                st.session_state.latest_nav_url = None
                st.error("Could not find a valid driving route for the selected locations.")
        else:
            st.session_state.latest_nav_url = None
            st.warning("Please choose both the starting point and destination from the search boxes.")

if st.session_state.latest_nav_url:
    st.markdown(
        f"""
        <a href="{st.session_state.latest_nav_url}" target="_blank" style="text-decoration:none;">
            <button style="
                width: 100%;
                background-color: #4285F4;
                color: white;
                padding: 15px;
                border: none;
                border-radius: 12px;
                font-weight: bold;
                font-size: 16px;
                margin-top: 10px;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 10px;">
                Start Navigation in Google Maps
            </button>
        </a>
        """,
        unsafe_allow_html=True,
    )

with hist_tab:
    if st.session_state.history:
        st.markdown(
            """
            <div class="table-shell">
                <div class="section-header">
                    <div>
                        <p class="section-title">Recent Calculations</p>
                        <p class="section-subtitle">Latest trip estimates appear first.</p>
                    </div>
                    <div class="mini-badge">Session history</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.table(pd.DataFrame(st.session_state.history).iloc[::-1])
    else:
        st.markdown(
            """
            <div class="empty-state">
                <h3 style="margin-top:0; color:#f8fafc;">No trips yet</h3>
                <p style="margin-bottom:0;">Run a calculation in the Trip Planner tab and the result will be saved here for this session.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

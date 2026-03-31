import html
import json
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from urllib.request import Request, urlopen

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

        .fuel-header-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.9rem;
            margin-top: 1rem;
            align-items: stretch;
        }

        .fuel-header-card {
            padding: 1rem 1.1rem;
            border-radius: 22px;
            border: 1px solid rgba(255, 255, 255, 0.12);
            box-shadow: 0 10px 22px rgba(2, 6, 23, 0.18);
            min-height: 92px;
        }

        .fuel-header-label {
            color: #94a3b8;
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 700;
        }

        .fuel-header-value {
            margin-top: 0.35rem;
            color: #f8fafc;
            font-size: 1.25rem;
            font-weight: 800;
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

        div[data-testid="stRadio"] > label {
            color: #f8fafc !important;
            font-weight: 700 !important;
        }

        div[data-testid="stRadio"] [role="radiogroup"] {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.9rem;
            margin-top: 0.4rem;
        }

        div[data-testid="stRadio"] [role="radiogroup"] label {
            min-height: 92px;
            padding: 1rem 1.1rem;
            border-radius: 22px;
            border: 1px solid rgba(255, 255, 255, 0.12);
            display: flex;
            align-items: center;
            justify-content: flex-start;
            transition: transform 0.16s ease, box-shadow 0.16s ease, border-color 0.16s ease;
            box-shadow: 0 10px 22px rgba(2, 6, 23, 0.18);
        }

        div[data-testid="stRadio"] [role="radiogroup"] label > div:first-child {
            display: none !important;
        }

        div[data-testid="stRadio"] [role="radiogroup"] label p {
            color: inherit !important;
            font-weight: 800 !important;
            line-height: 1.3 !important;
        }

        div[data-testid="stRadio"] [role="radiogroup"] label:nth-of-type(1) {
            background: linear-gradient(135deg, #facc15, #f59e0b);
            color: #111827 !important;
        }

        div[data-testid="stRadio"] [role="radiogroup"] label:nth-of-type(2) {
            background: linear-gradient(135deg, #eab308, #ca8a04);
            color: #111827 !important;
        }

        div[data-testid="stRadio"] [role="radiogroup"] label:nth-of-type(3) {
            background: linear-gradient(135deg, #22c55e, #16a34a);
            color: #f8fafc !important;
        }

        div[data-testid="stRadio"] [role="radiogroup"] label:nth-of-type(4) {
            background: linear-gradient(135deg, #ef4444, #dc2626);
            color: #fef2f2 !important;
        }

        div[data-testid="stRadio"] [role="radiogroup"] label:nth-of-type(5) {
            background: linear-gradient(135deg, #fb923c, #ea580c);
            color: #fff7ed !important;
        }

        div[data-testid="stRadio"] [role="radiogroup"] label:nth-of-type(6) {
            background: linear-gradient(135deg, #111827, #000000);
            color: #f8fafc !important;
        }

        div[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) {
            transform: scale(1.04);
            border-color: rgba(255, 255, 255, 0.75);
            box-shadow: 0 18px 32px rgba(2, 6, 23, 0.28);
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

        .wheel-shell {
            position: relative;
            margin-top: 0.75rem;
            margin-bottom: 1rem;
            padding: 1rem;
            border-radius: 26px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            background:
                radial-gradient(circle at top, rgba(255, 255, 255, 0.08), transparent 42%),
                linear-gradient(180deg, rgba(15, 23, 42, 0.92), rgba(15, 23, 42, 0.76));
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04), 0 18px 36px rgba(2, 6, 23, 0.22);
            overflow: hidden;
        }

        .wheel-shell::before,
        .wheel-shell::after {
            content: "";
            position: absolute;
            left: 1rem;
            right: 1rem;
            height: 34%;
            z-index: 0;
            pointer-events: none;
        }

        .wheel-shell::before {
            top: 0;
            background: linear-gradient(180deg, rgba(2, 6, 23, 0.78), rgba(2, 6, 23, 0));
        }

        .wheel-shell::after {
            bottom: 0;
            background: linear-gradient(0deg, rgba(2, 6, 23, 0.78), rgba(2, 6, 23, 0));
        }

        .wheel-highlight {
            position: absolute;
            left: 1rem;
            right: 1rem;
            top: 50%;
            height: 3.4rem;
            transform: translateY(-50%);
            border-top: 1px solid rgba(251, 146, 60, 0.22);
            border-bottom: 1px solid rgba(251, 146, 60, 0.22);
            background: rgba(251, 146, 60, 0.08);
            border-radius: 18px;
            z-index: 0;
            pointer-events: none;
            box-shadow: 0 0 18px rgba(249, 115, 22, 0.08);
        }

        .wheel-title {
            position: relative;
            z-index: 1;
            margin-bottom: 0.9rem;
            color: #e2e8f0;
            font-size: 1rem;
            font-weight: 700;
        }

        .wheel-colon {
            position: relative;
            z-index: 1;
            margin-top: 2.4rem;
            text-align: center;
            color: #fdba74;
            font-size: 2rem;
            font-weight: 800;
            letter-spacing: 0.04em;
        }

        div[data-testid="stSelectbox"] > div[data-baseweb="select"] > div,
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
        div[data-testid="stNumberInput"] input:hover,
        .stSearchbox > div:hover,
        .stSearchbox [data-baseweb="base-input"]:hover,
        .stSearchbox input:hover {
            border-color: rgba(251, 146, 60, 0.26) !important;
        }

        div[data-testid="stSelectbox"] > div[data-baseweb="select"] > div:focus-within,
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

        .stSearchbox input::placeholder {
            color: #64748b !important;
        }

        div[data-testid="stSlider"] > div[data-baseweb="slider"] {
            padding-top: 0.9rem;
            padding-bottom: 0.6rem;
            pointer-events: none;
        }

        div[data-testid="stSlider"] [role="slider"] {
            background: linear-gradient(135deg, #f97316, #fb923c) !important;
            border: 0 !important;
            box-shadow: 0 2px 10px rgba(249, 115, 22, 0.35) !important;
            pointer-events: auto !important;
        }

        div[data-testid="stSlider"] [data-testid="stTickBar"] {
            background: rgba(255, 255, 255, 0.16) !important;
            pointer-events: none !important;
        }

        div[data-testid="stSlider"] [data-baseweb="slider"] > div:not([role="slider"]) {
            pointer-events: none !important;
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

            .fuel-header-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
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


def to_title_display(value: str) -> str:
    words = []
    for word in str(value).strip().split():
        parts = re.split(r"([-/()])", word)
        formatted_parts = []
        for part in parts:
            if not part or re.fullmatch(r"[-/()]", part):
                formatted_parts.append(part)
            elif any(char.isalpha() for char in part):
                formatted_parts.append(part[:1].upper() + part[1:].lower())
            else:
                formatted_parts.append(part)
        words.append("".join(formatted_parts))
    return " ".join(words)


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
    required_cols = [
        "YEAR",
        "MAKE",
        "MODEL",
        "ENGINE SIZE",
        "CYLINDERS",
        "COMB (L/100 km)",
    ]
    df = df[cols].copy()
    df = df.dropna(subset=required_cols)
    df["VEHICLE CLASS"] = df["VEHICLE CLASS"].fillna("Unknown")
    df["FUEL"] = df["FUEL"].fillna("Unknown")
    for col in ["MAKE", "MODEL", "VEHICLE CLASS", "FUEL"]:
        df[col] = df[col].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
    df["MAKE_KEY"] = df["MAKE"].str.upper()
    df["MODEL_KEY"] = df["MODEL"].str.upper()
    df["MAKE_DISPLAY"] = (
        df.groupby("MAKE_KEY")["MAKE"].transform(pick_display_value).map(to_title_display)
    )
    df["MODEL_DISPLAY"] = (
        df.groupby(["MAKE_KEY", "MODEL_KEY"])["MODEL"]
        .transform(pick_display_value)
        .map(to_title_display)
    )
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
    matches = brand_options[
        brand_options["MAKE_DISPLAY"].str.contains(searchterm, case=False, na=False)
    ]["MAKE_DISPLAY"].tolist()
    return matches[:4]


def search_vehicles(searchterm: str):
    if not searchterm.strip():
        return vehicle_options["VEHICLE_LABEL"].tolist()[:6]
    matches = vehicle_options[
        vehicle_options["VEHICLE_LABEL"].str.contains(searchterm, case=False, na=False)
    ]["VEHICLE_LABEL"].tolist()
    return matches[:6]


def search_option_labels(searchterm: str, options, limit=4):
    if not searchterm.strip():
        return options[:limit]
    return [
        option for option in options
        if searchterm.lower() in str(option).lower()
    ][:limit]


@st.cache_data(ttl=900)
def get_live_fuel_prices():
    def first_numeric_value(row, keys, fallback):
        for key in keys:
            value = pd.to_numeric(row.get(key), errors="coerce")
            if pd.notna(value):
                return float(value)
        return fallback

    try:
        url = "https://api.data.gov.my/data-catalogue?id=fuelprice&limit=1&sort=-date"
        request = Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 FuelTripCostPredictor/1.0"},
        )
        with urlopen(request, timeout=15) as response:
            payload = json.loads(response.read().decode("utf-8"))

        records = payload.get("data") if isinstance(payload, dict) else payload
        if not records:
            raise ValueError("Fuel price API returned no rows.")

        latest = records[0]
        ron95_price = first_numeric_value(
            latest,
            ["ron95", "ron_95", "ron95_unsubsidized"],
            1.99,
        )
        budi95_price = first_numeric_value(
            latest,
            ["ron95_budi", "ron95_bersubsidi", "ron95_subsidized"],
            1.99,
        )
        ron97_price = first_numeric_value(latest, ["ron97", "ron_97"], 4.55)
        diesel_price = first_numeric_value(
            latest,
            [
                "diesel_white_peninsula",
                "diesel_white_peninsular",
                "diesel_peninsula",
                "diesel",
            ],
            4.72,
        )

        return {
            "RON 95": ron95_price,
            "BUDI 95": budi95_price,
            "RON 97": ron97_price,
            "RON 100": 7.50,
            "V-Power Racing": 7.88,
            "Diesel": diesel_price,
        }
    except Exception:
        return {

        }


SEARCHBOX_STYLE = {
    "searchbox": {
        "optionEmpty": "hidden",
        "menuList": {
            "maxHeight": "224px",
            "overflowY": "auto",
        },
    }
}

FUEL_OPTION_META = {
    "RON 95": {"icon": "R95", "price_key": "RON 95"},
    "BUDI 95": {"icon": "B95", "price_key": "BUDI 95"},
    "RON 97": {"icon": "R97", "price_key": "RON 97"},
    "RON 100": {"icon": "R100", "price_key": "RON 100"},
    "V-Power Racing": {"icon": "VPR", "price_key": "V-Power Racing"},
    "Diesel": {"icon": "DSL", "price_key": "Diesel"},
}


def get_selected_fuel_price():
    selected_fuel = st.session_state.fuel_type_option
    return round(LIVE_PRICES[FUEL_OPTION_META[selected_fuel]["price_key"]], 2)


def apply_selected_fuel_price():
    value = get_selected_fuel_price()
    st.session_state.fuel_price = value
    st.session_state.fuel_price_slider = value
    st.session_state.fuel_price_input = value
    st.session_state.fuel_price_manual_override = False


def sync_fuel_price_from_selected_type():
    selected_price = get_selected_fuel_price()
    if not st.session_state.get("fuel_price_manual_override", False):
        st.session_state.fuel_price = selected_price
        st.session_state.fuel_price_slider = selected_price
        st.session_state.fuel_price_input = selected_price


def sync_fuel_price_from_slider():
    value = round(float(st.session_state.fuel_price_slider), 2)
    st.session_state.fuel_price = value
    st.session_state.fuel_price_input = value
    st.session_state.fuel_price_manual_override = True


def sync_fuel_price_from_input():
    value = round(float(st.session_state.fuel_price_input), 2)
    value = min(10.0, max(1.0, value))
    st.session_state.fuel_price = value
    st.session_state.fuel_price_slider = value
    st.session_state.fuel_price_input = value
    st.session_state.fuel_price_manual_override = True


def nudge_fuel_price(delta):
    value = round(st.session_state.fuel_price + delta, 2)
    value = min(10.0, max(1.0, value))
    st.session_state.fuel_price = value
    st.session_state.fuel_price_slider = value
    st.session_state.fuel_price_input = value
    st.session_state.fuel_price_manual_override = True


def sync_departure_time():
    st.session_state.departure_time = datetime.now().replace(
        hour=int(st.session_state.departure_hour),
        minute=int(st.session_state.departure_minute),
        second=0,
        microsecond=0,
    ).time()


def get_route_info(start, end, dep_time: datetime):
    try:
        directions = gmaps.directions(
            start,
            end,
            mode="driving",
            departure_time=dep_time,
            traffic_model="best_guess",
            alternatives=True,
        )
        if not directions:
            return None
        return directions
    except Exception as exc:
        st.error(f"Maps Error: {exc}")
        return None


def get_traffic_level(traffic_ratio):
    if traffic_ratio >= 1.2:
        return "Heavy", "var(--bad)"
    if traffic_ratio >= 1.05:
        return "Moderate", "var(--warn)"
    return "Light", "var(--good)"


def calculate_average_speed(dist_km, traffic_min):
    if traffic_min <= 0:
        return 0.0
    return dist_km / (traffic_min / 60)


def get_environment_profile(average_speed):
    if average_speed < 25:
        return 1.6, "Residential/School Zone - High stop-start", "var(--bad)"
    if average_speed < 55:
        return 1.4, "Mixed City Road", "var(--warn)"
    if average_speed <= 85:
        return 1.00, "Standard Flow", "var(--good)"
    return 0.90, "Highway Cruising - High Efficiency", "#93c5fd"


def get_consumption_rate(selected_car, year_val, model, le_fuel, le_class):
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

    consumption_rate = actual_l_100km if actual_l_100km > 0 else ai_l_100km
    if consumption_rate is None or consumption_rate <= 0:
        consumption_rate = actual_l_100km

    prediction_note = (
        f"Actual car data: {actual_l_100km:.1f} L/100 km"
        if ai_l_100km is None
        else f"Actual car data: {actual_l_100km:.1f} L/100 km | AI reference: {ai_l_100km:.1f} L/100 km"
    )
    return consumption_rate, prediction_note


def build_route_result(route_option, route_index, consumption_rate, fuel_price):
    leg = route_option["legs"][0]
    distance_km = leg["distance"]["value"] / 1000
    duration_mins = leg.get("duration_in_traffic", leg["duration"])["value"] / 60
    average_speed = calculate_average_speed(distance_km, duration_mins)
    multiplier, environment_label, environment_color = get_environment_profile(average_speed)
    final_liters = (distance_km / 100) * consumption_rate * multiplier
    final_cost = final_liters * fuel_price
    base_cost = (distance_km / 100) * consumption_rate * fuel_price
    route_summary = route_option.get("summary") or f"Route {route_index}"

    return {
        "summary": route_summary,
        "dist_km": distance_km,
        "traffic_min": duration_mins,
        "average_speed": average_speed,
        "env_multiplier": multiplier,
        "environment_label": environment_label,
        "environment_color": environment_color,
        "final_liters": final_liters,
        "final_cost": final_cost,
        "base_cost": base_cost,
        "environment_extra": final_cost - base_cost,
        "start_addr": leg["start_address"],
        "end_addr": leg["end_address"],
    }


def render_result_panels(route, selected_car, year_val, fuel_price, model, le_fuel, le_class):
    l_100km_pred, prediction_note = get_consumption_rate(
        selected_car, year_val, model, le_fuel, le_class
    )
    route_result = build_route_result(route, 1, l_100km_pred, fuel_price)
    average_speed = route_result["average_speed"]
    env_multiplier = route_result["env_multiplier"]
    environment_label = route_result["environment_label"]
    environment_color = route_result["environment_color"]
    final_liters = route_result["final_liters"]
    final_cost = route_result["final_cost"]
    base_cost = route_result["base_cost"]
    environment_extra = route_result["environment_extra"]

    cost_html = f"""
        <div class="cost-card">
            <div class="cost-title">Estimated Trip Cost</div>
            <div class="cost-value">RM {final_cost:.2f}</div>
            <div class="cost-copy">
                Distance: {route_result["dist_km"]:.1f} km<br/>
                Average Speed: {average_speed:.1f} km/h<br/>
                Travel Time: {route_result["traffic_min"]:.0f} mins<br/>
                Fuel needed: {final_liters:.1f} L<br/>
            </div>
        </div>
    """
    st.markdown(cost_html, unsafe_allow_html=True)

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
                    <strong style="color:{environment_color};">{environment_label}</strong>
                </div>
            </div>
        </div>
    """
    st.markdown(summary_html, unsafe_allow_html=True)

    return final_cost


inject_styles()
LIVE_PRICES = get_live_fuel_prices()
vehicle_df = load_csv_data(CSV_PATH.stat().st_mtime_ns if CSV_PATH.exists() else 0)
model, le_fuel, le_class = load_ai_model()

if "history" not in st.session_state:
    st.session_state.history = []
if "departure_time" not in st.session_state:
    st.session_state.departure_time = datetime.now().replace(second=0, microsecond=0).time()
if "departure_hour" not in st.session_state:
    st.session_state.departure_hour = f"{st.session_state.departure_time.hour:02d}"
if "departure_minute" not in st.session_state:
    st.session_state.departure_minute = f"{st.session_state.departure_time.minute:02d}"
if "latest_nav_url" not in st.session_state:
    st.session_state.latest_nav_url = None
if "pending_route_results" not in st.session_state:
    st.session_state.pending_route_results = []
if "pending_route_directions" not in st.session_state:
    st.session_state.pending_route_directions = []
if "selected_route_index" not in st.session_state:
    st.session_state.selected_route_index = None
if "selected_route_meta" not in st.session_state:
    st.session_state.selected_route_meta = {}
if "selected_route_history_saved" not in st.session_state:
    st.session_state.selected_route_history_saved = False
if "fuel_type_option" not in st.session_state:
    st.session_state.fuel_type_option = "RON 95"
elif st.session_state.fuel_type_option not in FUEL_OPTION_META:
    st.session_state.fuel_type_option = "RON 95"
if "fuel_price_manual_override" not in st.session_state:
    st.session_state.fuel_price_manual_override = False
if "fuel_price" not in st.session_state:
    st.session_state.fuel_price = round(LIVE_PRICES["RON 95"], 2)
if "fuel_price_slider" not in st.session_state:
    st.session_state.fuel_price_slider = st.session_state.fuel_price
if "fuel_price_input" not in st.session_state:
    st.session_state.fuel_price_input = st.session_state.fuel_price

sync_fuel_price_from_selected_type()

# --- CLEAN HEADER START ---
# Build cards individually to avoid join errors
cards_html = ""
color_map = {
    "RON 95": "#facc15",  # Yellow
    "BUDI 95": "#fde68a", # Soft Yellow
    "RON 97": "#4ade80",  # Green
    "RON 100": "#fb7185", # Red/Pink
    "V-Power Racing": "#f97316", # Orange
    "Diesel": "#94a3b8"   # Grey/Silver
}
for label, price in LIVE_PRICES.items():
    text_color = color_map.get(label, "#ffffff")
    cards_html += f"""
    <div class="fuel-header-card">
        <div class="fuel-header-label" style="color: {text_color};">{label}</div>
        <div class="fuel-header-value">RM {price:.2f}</div>
    </div>"""

# One single markdown call with perfectly matched tags
st.markdown(
    f"""
    <div class="hero-shell">
        <h1 class="hero-title">Fuel Trip <span>Cost Predictor</span></h1>
        <p class="hero-copy">
            Plan a drive, check traffic aware fuel cost, and compare the impact of vehicle choice,
            time of departure, and current fuel price in one place.
        </p>
        <div class="fuel-header-grid">
            {cards_html}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
# --- CLEAN HEADER END ---

if vehicle_df.empty:
    st.error(f"No vehicle data could be loaded from {CSV_PATH.name}.")
    st.stop()

brand_options = (
    vehicle_df[["MAKE_KEY", "MAKE_DISPLAY"]]
    .drop_duplicates()
    .sort_values("MAKE_DISPLAY")
)
vehicle_options = (
    vehicle_df[["MAKE_KEY", "MODEL_KEY", "MAKE_DISPLAY", "MODEL_DISPLAY"]]
    .drop_duplicates()
    .assign(
        VEHICLE_LABEL=lambda df: df["MAKE_DISPLAY"].str.strip() + " " + df["MODEL_DISPLAY"].str.strip()
    )
    .sort_values("VEHICLE_LABEL")
)
st.markdown(
    """
    <div class="section-card">
        <div class="section-header">
            <div>
                <p class="section-title">1. Choose Vehicle</p>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="field-label">Vehicle</div>', unsafe_allow_html=True)
vehicle_display = st_searchbox(
    search_vehicles,
    key="vehicle_search",
    placeholder="Search brand or model",
    style_overrides=SEARCHBOX_STYLE,
    default_options=vehicle_options["VEHICLE_LABEL"].tolist()[:6],
    style_absolute=False,
)

selected_vehicle_option = None
if vehicle_display:
    vehicle_match = vehicle_options.loc[
        vehicle_options["VEHICLE_LABEL"].str.casefold() == vehicle_display.casefold()
    ]
    if not vehicle_match.empty:
        selected_vehicle_option = vehicle_match.iloc[0]

if selected_vehicle_option is not None:
    matching_rows = vehicle_df[
        (vehicle_df["MAKE_KEY"] == selected_vehicle_option["MAKE_KEY"])
        & (vehicle_df["MODEL_KEY"] == selected_vehicle_option["MODEL_KEY"])
    ]
    year_options = sorted(matching_rows["YEAR"].unique(), reverse=True)
    year_display = st_searchbox(
        lambda term: search_option_labels(term, [str(year) for year in year_options]),
        key="year_search",
        placeholder="Search year",
        style_overrides=SEARCHBOX_STYLE,
        default_options=[str(year) for year in year_options[:4]],
        style_absolute=False,
    )
    year_val = int(year_display) if year_display else None
else:
    st.session_state.pop("year_search", None)
    st.text_input("Year", value="Select a vehicle first", disabled=True)
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
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="mini-badge" style="margin-bottom:0.6rem;">Starting point</div>', unsafe_allow_html=True)
    start_addr = st_searchbox(
        search_places,
        key="start_search",
        placeholder="Search starting location",
        style_overrides=SEARCHBOX_STYLE,
    )
    st.markdown('<div class="mini-badge" style="margin-bottom:0.6rem;">Destination</div>', unsafe_allow_html=True)
    end_addr = st_searchbox(
        search_places,
        key="end_search",
        placeholder="Search destination",
        style_overrides=SEARCHBOX_STYLE,
    )

    st.time_input(
        "Departure Time",
        key="departure_time_widget",
        on_change=sync_departure_time,
    )
    chosen_time = st.session_state.departure_time_widget

    st.markdown(
        """
        <div class="section-card" style="margin-top:1rem;">
            <div class="section-header">
                <div>
                    <p class="section-title">3. Fuel Settings</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="field-label">Fuel Type</div>', unsafe_allow_html=True)
    st.radio(
        "Fuel Type",
        options=list(FUEL_OPTION_META.keys()),
        key="fuel_type_option",
        horizontal=False,
        on_change=apply_selected_fuel_price,
        label_visibility="collapsed",
    )

    fuel_controls_col1, fuel_controls_col2, fuel_controls_col3 = st.columns([1, 1.4, 1])
 
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
            <div class="section-card" style="margin-top:1.9rem; margin-bottom:0;">
                <div class="section-title">Selected Fuel</div>
                <div class="section-subtitle">{FUEL_OPTION_META[st.session_state.fuel_type_option]["icon"]}: RM{st.session_state.fuel_price:.2f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    fuel_price = st.session_state.fuel_price

    run_calc = st.button("Calculate Trip Cost")

    if run_calc:
        if selected_car is None:
            st.session_state.latest_nav_url = None
            st.session_state.pending_route_results = []
            st.session_state.pending_route_directions = []
            st.session_state.selected_route_index = None
            st.session_state.selected_route_meta = {}
            st.session_state.selected_route_history_saved = False
            st.warning("Please search and select a brand, model, and year first.")
        elif start_addr and end_addr:
            now = datetime.now()
            dep_datetime = datetime.combine(now.date(), chosen_time)
            if dep_datetime < now:
                dep_datetime += timedelta(days=1)

            with st.spinner("Calculating route and fuel estimate..."):
                directions = get_route_info(start_addr, end_addr, dep_datetime)

            if directions:
                consumption_rate, _ = get_consumption_rate(
                    selected_car, year_val, model, le_fuel, le_class
                )
                results = []
                for idx, route_option in enumerate(directions, start=1):
                    results.append(
                        build_route_result(route_option, idx, consumption_rate, fuel_price)
                    )
                st.session_state.pending_route_results = results
                st.session_state.pending_route_directions = directions
                st.session_state.selected_route_index = None
                st.session_state.selected_route_meta = {
                    "selected_car": selected_car.to_dict(),
                    "year_val": int(year_val),
                    "fuel_price": float(fuel_price),
                    "start_addr": start_addr,
                    "end_addr": end_addr,
                    "dep_datetime": dep_datetime.strftime("%d/%m %H:%M"),
                }
                st.session_state.latest_nav_url = None
                st.session_state.selected_route_history_saved = False
                st.rerun()

                st.markdown(
                    '<div class="field-label" style="margin-top:1rem;">Route Options</div>',
                    unsafe_allow_html=True,
                )
                route_columns = st.columns(3)
                for idx, route_result in enumerate(results[:3]):
                    route_card_html = f"""
                        <div class="metric-card" style="height:100%;">
                            <div class="metric-eyebrow">{html.escape(route_result["summary"])}</div>
                            <div class="metric-number" style="font-size:1.7rem;">RM {route_result["final_cost"]:.2f}</div>
                            <div class="metric-note">{route_result["dist_km"]:.1f} km • {route_result["traffic_min"]:.0f} min</div>
                            <div class="metric-note" style="margin-top:0.45rem;">
                                {route_result["average_speed"]:.1f} km/h • {route_result["env_multiplier"]:.2f}x
                            </div>
                        </div>
                    """
                    with route_columns[idx]:
                        st.markdown(route_card_html, unsafe_allow_html=True)
                        if st.button("Choose This Route", key=f"choose_route_{idx}"):
                            st.session_state.selected_route_index = idx
                            st.session_state.latest_nav_url = (
                                f"https://www.google.com/maps/dir/?api=1"
                                f"&origin={start_addr}&destination={end_addr}&travelmode=driving"
                            ).replace(" ", "+")
                            st.session_state.selected_route_history_saved = False

            else:
                st.session_state.latest_nav_url = None
                st.session_state.pending_route_results = []
                st.session_state.pending_route_directions = []
                st.session_state.selected_route_index = None
                st.session_state.selected_route_meta = {}
                st.session_state.selected_route_history_saved = False
                st.error("Could not find a valid driving route for the selected locations.")
        else:
            st.session_state.latest_nav_url = None
            st.session_state.pending_route_results = []
            st.session_state.pending_route_directions = []
            st.session_state.selected_route_index = None
            st.session_state.selected_route_meta = {}
            st.session_state.selected_route_history_saved = False
            st.warning("Please choose both the starting point and destination from the search boxes.")

    if (
        st.session_state.pending_route_results
        and st.session_state.selected_route_index is None
    ):
        st.markdown(
            """
            <div class="section-card" style="margin-top:1rem;">
                <div class="section-header">
                    <div>
                        <p class="section-title">4. Choose Route</p>
                        <p class="section-subtitle">Pick one route option first. Then the trip summary will be shown for that route.</p>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        route_columns = st.columns(3)
        for idx, route_result in enumerate(st.session_state.pending_route_results[:3]):
            route_card_html = f"""
                <div class="metric-card" style="height:100%;">
                    <div class="metric-eyebrow">{html.escape(route_result["summary"])}</div>
                    <div class="metric-number" style="font-size:1.7rem;">RM {route_result["final_cost"]:.2f}</div>
                    <div class="metric-note">{route_result["dist_km"]:.1f} km | {route_result["traffic_min"]:.0f} min</div>
                </div>
            """
            with route_columns[idx]:
                st.markdown(route_card_html, unsafe_allow_html=True)
                if st.button("Choose This Route", key=f"choose_route_persist_{idx}"):
                    st.session_state.selected_route_index = idx
                    st.session_state.latest_nav_url = (
                        f"https://www.google.com/maps/dir/?api=1"
                        f"&origin={st.session_state.selected_route_meta['start_addr']}"
                        f"&destination={st.session_state.selected_route_meta['end_addr']}"
                        f"&travelmode=driving"
                    ).replace(" ", "+")
                    st.session_state.selected_route_history_saved = False
                    st.rerun()

    if (
        st.session_state.selected_route_index is not None
        and st.session_state.pending_route_directions
        and st.session_state.selected_route_meta
    ):
        selected_route = st.session_state.pending_route_directions[
            st.session_state.selected_route_index
        ]
        selected_car_series = pd.Series(st.session_state.selected_route_meta["selected_car"])
        final_cost = render_result_panels(
            selected_route,
            selected_car_series,
            st.session_state.selected_route_meta["year_val"],
            st.session_state.selected_route_meta["fuel_price"],
            model,
            le_fuel,
            le_class,
        )

        if not st.session_state.selected_route_history_saved:
            st.session_state.history.append(
                {
                    "Date": st.session_state.selected_route_meta["dep_datetime"],
                    "Vehicle": (
                        f"{selected_car_series['MAKE_DISPLAY']} "
                        f"{selected_car_series['MODEL_DISPLAY']} "
                        f"{st.session_state.selected_route_meta['year_val']}"
                    ),
                    "From": st.session_state.selected_route_meta["start_addr"],
                    "To": st.session_state.selected_route_meta["end_addr"],
                    "Cost": f"RM {final_cost:.2f}",
                }
            )
            st.session_state.selected_route_history_saved = True

        if st.button("Choose a Different Route", key="choose_different_route"):
            st.session_state.selected_route_index = None
            st.session_state.latest_nav_url = None
            st.session_state.selected_route_history_saved = False
            st.rerun()

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

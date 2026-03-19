import streamlit as st
import pandas as pd
import googlemaps
import pydeck as pdk
import requests
import joblib
from streamlit_searchbox import st_searchbox
from datetime import datetime, timedelta  # Added timedelta
from pathlib import Path
from typing import Dict, List, Optional
import os
from dotenv import load_dotenv

# Load the keys from the .env file
load_dotenv()

# --- 1. CONFIG & ASSETS ---
st.set_page_config(page_title="Fuel Trip Cost Predictor", layout="wide", page_icon="🚗")

# Replace with your key (Keep it safe!)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
gmaps = googlemaps.Client(key=GOOGLE_API_KEY)

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "fuelconsumption3.csv"

# --- 2. STYLING ---
def inject_styles():
    st.markdown("""
        <style>
        .big-font {
        font-size:20px !important;
        color: ##6495ED; 
        font-weight: bold;
        }
                
    .metric-container {
            background-color: #f0f2f6; /* Light grey background box */
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #dcdcdc;
        }
        .metric-label {
            color: #555555; /* Medium grey for labels */
            font-size: 14px;
            font-weight: bold;
        }
        .metric-value {
            color: #1E1E1E; /* Deep black for the numbers */
            font-size: 24px;
            font-weight: bold;
        }
        .stApp {
            background: radial-gradient(circle at top left, rgba(255, 196, 114, 0.2), transparent 25%),
                        linear-gradient(135deg, #fff8ed 0%, #eef7ff 55%, #e5f7f2 100%);
        }
        div[data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.8);
            border-radius: 15px;
            padding: 15px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        }
        .glass-card {
            background: rgba(255, 255, 255, 0.6);
            border-radius: 20px;
            padding: 25px;
            border: 1px solid rgba(255, 255, 255, 0.3);
            box-shadow: 0 8px 32px rgba(31, 38, 135, 0.07);
        }
        div.stButton > button {
            background: linear-gradient(90deg, #ef7f45 0%, #e35d3d 100%);
            color: white; border-radius: 25px; font-weight: 700; width: 100%;
            height: 3em;
        }
        </style>
        """, unsafe_allow_html=True)

# --- 3. DATA & AI LOADING ---
def pick_display_value(values):
    unique_values = sorted({str(value).strip() for value in values if str(value).strip()})
    preferred_values = [value for value in unique_values if not value.isupper()]
    return preferred_values[0] if preferred_values else unique_values[0]


@st.cache_data
def load_csv_data(_csv_mtime_ns):
    df = pd.read_csv(CSV_PATH, low_memory=False)
    cols = ["YEAR", "MAKE", "MODEL", "VEHICLE CLASS", "FUEL", "ENGINE SIZE", "CYLINDERS", "COMB (L/100 km)"]
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
        model = joblib.load('fuel_model2.pkl')
        le_fuel = joblib.load('fuel_encoder2.pkl')
        le_class = joblib.load('class_encoder2.pkl')
        return model, le_fuel, le_class
    except:
        return None, None, None

# --- 4. HELPERS ---
def search_places(searchterm: str):
    if not searchterm:
        return []
    results = gmaps.places_autocomplete(
        input_text=searchterm,
        components={"country": "MY"} 
    )
    return [res["description"] for res in results]

def get_route_info(start, end, dep_time: datetime):
    try:
        # Requesting traffic-aware directions for a specific time
        directions = gmaps.directions(
            start, end, 
            mode="driving", 
            departure_time=dep_time,
            traffic_model="best_guess"
        )
        if not directions: return None
        
        leg = directions[0]["legs"][0]
        dur_norm = leg["duration"]["value"] / 60
        dur_traffic = leg.get("duration_in_traffic", leg["duration"])["value"] / 60
        
        return {
            "dist_km": leg["distance"]["value"] / 1000,
            "duration_min": dur_norm,
            "traffic_min": dur_traffic,
            "path": [{"lat": p["lat"], "lon": p["lng"]} for p in googlemaps.convert.decode_polyline(directions[0]["overview_polyline"]["points"])],
            "start_addr": leg["start_address"],
            "end_addr": leg["end_address"]
        }
    except Exception as e:
        st.error(f"Google Error: {e}")
        return None

# --- 5. INITIALIZATION ---
inject_styles()
vehicle_df = load_csv_data(CSV_PATH.stat().st_mtime_ns)
model, le_fuel, le_class = load_ai_model()

if "history" not in st.session_state:
    st.session_state.history = []

# --- 6. SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Settings")
    fuel_price = st.number_input("Fuel Price (RM/L)", value=2.05)
    st.divider()
    st.info("AI: Trained on 2000-2022 Data")

# --- 7. MAIN UI ---
st.title("🚗 Fuel Trip Cost Predictor")

# Vehicle Selection Area
with st.expander("🚙 Select Vehicle Model", expanded=True):
    c1, c2, c3 = st.columns(3)
    brand_options = (
        vehicle_df[["MAKE_KEY", "MAKE_DISPLAY"]]
        .drop_duplicates()
        .sort_values("MAKE_DISPLAY")
    )
    brand_keys = brand_options["MAKE_KEY"].tolist()
    default_brand = "PERODUA" if "PERODUA" in brand_keys else brand_keys[0]
    brand = c1.selectbox(
        "Brand",
        brand_keys,
        index=brand_keys.index(default_brand),
        format_func=lambda make_key: brand_options.loc[
            brand_options["MAKE_KEY"] == make_key, "MAKE_DISPLAY"
        ].iloc[0]
    )
    brand_models = (
        vehicle_df.loc[vehicle_df["MAKE_KEY"] == brand, ["MODEL_KEY", "MODEL_DISPLAY"]]
        .drop_duplicates()
        .sort_values("MODEL_DISPLAY")
    )
    model_keys = brand_models["MODEL_KEY"].tolist()
    model_name = c2.selectbox(
        "Model",
        model_keys,
        format_func=lambda model_key: brand_models.loc[
            brand_models["MODEL_KEY"] == model_key, "MODEL_DISPLAY"
        ].iloc[0]
    )
    matching_rows = vehicle_df[
        (vehicle_df["MAKE_KEY"] == brand) & (vehicle_df["MODEL_KEY"] == model_name)
    ]
    year_val = c3.selectbox("Year", sorted(matching_rows["YEAR"].unique(), reverse=True))

    selected_car = matching_rows[matching_rows["YEAR"] == year_val].iloc[0]

# Tabs
calc_tab, hist_tab = st.tabs(["🚀 Predictor", "📜 Trip History"])

with calc_tab:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    r1, r2 = st.columns(2)
    with r1:
        st.write("**Starting Point**")
        start_addr = st_searchbox(search_places, key="start_search", placeholder="Search starting location...")
    with r2:
        st.write("**Destination**")
        end_addr = st_searchbox(search_places, key="end_search", placeholder="Search destination...")

    if "departure_time" not in st.session_state:
        st.session_state.departure_time = datetime.now().time()

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        # 2. Use the session_state value and update it when the user clicks
        chosen_time = st.time_input(
            "Departure Time", 
            value=st.session_state.departure_time,
            key="time_widget" # Adding a key helps Streamlit track this specific widget
        )
        # 3. Update the memory with the new choice
        st.session_state.departure_time = chosen_time
        
    with col_t2:
        st.caption("Google uses historical data to predict traffic for this specific time.")

    run_calc = st.button("Calculate Final Cost")
    st.markdown('</div>', unsafe_allow_html=True)

    if run_calc:
        if start_addr and end_addr:
            # Prepare the exact departure datetime
            now = datetime.now()
            dep_datetime = datetime.combine(now.date(), chosen_time)
            if dep_datetime < now:
                dep_datetime += timedelta(days=1)

            with st.spinner(f"Analyzing traffic for {chosen_time}..."):
                route = get_route_info(start_addr, end_addr, dep_datetime)
                
                if route:
                    try:
                        f_enc = le_fuel.transform([selected_car['FUEL']])[0]
                        c_enc = le_class.transform([selected_car['VEHICLE CLASS']])[0]
                        
                        l_100km_pred = model.predict([[
                            selected_car['ENGINE SIZE'], 
                            selected_car['CYLINDERS'], 
                            year_val, f_enc, c_enc
                        ]])[0]
                    except:
                        l_100km_pred = selected_car['COMB (L/100 km)']
                    
                    # Calculation
                    base_liters = (route["dist_km"] / 100) * l_100km_pred
                    
                    # Traffic Penalty Logic
                    traffic_ratio = route["traffic_min"] / max(route["duration_min"], 1)
                    penalty = 1 + ((traffic_ratio - 1) * 0.5) 
                    final_liters = base_liters * penalty
                    final_cost = final_liters * fuel_price

                    # Display Metrics
                    st.divider()
                    m1, m2, m3 = st.columns(3)

                    with m1:
                        st.markdown(f'<div class="metric-container"><p class="metric-label">Distance</p><p class="metric-value">{route["dist_km"]:.1f} km</p></div>', unsafe_allow_html=True)

                    with m2:
                        st.markdown(f'<div class="metric-container"><p class="metric-label">Traffic Time</p><p class="metric-value">{route["traffic_min"]:.0f} min</p></div>', unsafe_allow_html=True)

                    with m3:
                        traffic_extra = final_cost - (base_liters * fuel_price)
                        st.markdown(f'''
                            <div class="metric-container" style="background-color: #e1f5fe; border: 1px solid #01579b;">
                                <p class="metric-label" style="color: #01579b;">Predicted Cost</p>
                                <p class="metric-value" style="color: #01579b;">RM {final_cost:.2f}</p>
                                <p style="color: #d32f2f; font-size: 12px; margin-top: -10px;">+RM {traffic_extra:.2f} traffic charge</p>
                            </div>
                        ''', unsafe_allow_html=True)

                    # Map
                    st.pydeck_chart(pdk.Deck(
                        map_style='road',
                        initial_view_state=pdk.ViewState(
                            latitude=route["path"][0]["lat"], longitude=route["path"][0]["lon"], zoom=11
                        ),
                        layers=[pdk.Layer(
                            "PathLayer", data=[{"path": [[p["lon"], p["lat"]] for p in route["path"]]}], 
                            get_path="path", get_width=5, get_color=[227, 93, 61]
                        )]
                    ))
                    
                    st.session_state.history.append({
                        "Date": dep_datetime.strftime("%d/%m %H:%M"), 
                        "From": start_addr, "To": end_addr, 
                        "Cost": f"RM {final_cost:.2f}"
                    })
                else:
                    st.error("Could not find a valid driving route.")
        else:
            st.warning("Please select locations using the search boxes.")

with hist_tab:
    if st.session_state.history:
        st.table(pd.DataFrame(st.session_state.history))
    else:
        st.info("No trips calculated yet.")

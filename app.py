import streamlit as st
import joblib
import requests
from geopy.geocoders import Nominatim

# --- LOAD AI MODEL ---
model = joblib.load('D:/UIA/LAB 1 Python Intsllation with VSCODE-20250306/Machine Learning/Fuel/fuel_model.pkl')
le_fuel = joblib.load('D:/UIA/LAB 1 Python Intsllation with VSCODE-20250306/Machine Learning/Fuel/fuel_encoder.pkl')
le_class = joblib.load('D:/UIA/LAB 1 Python Intsllation with VSCODE-20250306/Machine Learning/Fuel/class_encoder.pkl')

# --- HELPER: GET DISTANCE ---
def get_km(start, end):
    try:
        geolocator = Nominatim(user_agent="myvi_app")
        l1, l2 = geolocator.geocode(start), geolocator.geocode(end)
        url = f"http://router.project-osrm.org/route/v1/driving/{l1.longitude},{l1.latitude};{l2.longitude},{l2.latitude}?overview=false"
        data = requests.get(url).json()
        return data['routes'][0]['distance'] / 1000
    except:
        return None

# --- UI INTERFACE ---
st.title("🚗 Myvi Fuel Cost Predictor")
st.write("Predict your daily commute cost using Machine Learning.")

col1, col2 = st.columns(2)
with col1:
    start_loc = st.text_input("From (e.g., Kajang)", "Kajang")
    end_loc = st.text_input("To (e.g., KLCC)", "KLCC")
with col2:
    fuel_price = st.number_input("Fuel Price (RM/L)", value=2.05)
    car_year = st.slider("Car Year", 1984, 2026, 2022)

if st.button("Calculate Cost"):
    with st.spinner("Calculating distance and fuel..."):
        dist = get_km(start_loc, end_loc)
        
        if dist:
            # Prepare inputs for AI (Using Myvi 1.5H specs)
            f_enc = le_fuel.transform(['Regular'])[0]
            c_enc = le_class.transform(['Compact Cars'])[0]
            
            # AI Prediction (returns MPG)
            pred_mpg = model.predict([[1.5, 4, car_year, f_enc, c_enc]])[0]
            
            # Conversion: MPG -> L/100km -> Total RM
            l_100km = 235.215 / pred_mpg
            total_rm = (dist / 100) * l_100km * fuel_price
            
            st.success(f"Estimated Distance: {dist:.2f} km")
            st.metric("Total Trip Cost", f"RM {total_rm:.2f}")
        else:
            st.error("Could not find locations. Please try again.")
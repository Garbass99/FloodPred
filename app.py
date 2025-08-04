import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# Load model and preprocessor
@st.cache_resource
def load_artifacts():
    model = joblib.load('flood_prediction.pkl')
    preprocessor = joblib.load('preprocessor.pkl')
    return model, preprocessor

def main():
    st.title('Flood Prediction System')
    
    model, preprocessor = load_artifacts()

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            max_temp = st.number_input('Max Temp (°C)', min_value=0.0, max_value=50.0, value=30.0)
            min_temp = st.number_input('Min Temp (°C)', min_value=0.0, max_value=40.0, value=20.0)
            rainfall = st.number_input('Rainfall (mm)', min_value=0.0, max_value=1000.0, value=50.0)
            
        with col2:
            humidity = st.number_input('Humidity (%)', min_value=0.0, max_value=100.0, value=70.0)
            wind_speed = st.number_input('Wind Speed (km/h)', min_value=0.0, value=10.0)
            cloud_cover = st.number_input('Cloud Coverage (oktas)', min_value=0.0, max_value=8.0, value=3.0)
            sunshine = st.number_input('Sunshine (hours)', min_value=0.0, max_value=12.0, value=6.0)
        
        # Geographic inputs
        latitude = st.number_input('Latitude', min_value=-90.0, max_value=90.0, value=23.7)
        longitude = st.number_input('Longitude', min_value=-180.0, max_value=180.0, value=90.4)
        alt = st.number_input('Altitude (m)', min_value=0.0, value=10.0)
        
        # Correct Period input (year.month format)
        year = st.slider('Year', 1949, 2023, 2023)
        month = st.slider('Month', 1, 12, 6)
        period = float(f"{year}.{month:02d}")  # Converts to 2023.06 format
        
        if st.form_submit_button("Predict"):
            input_data = pd.DataFrame({
                'Max_Temp': [max_temp],
                'Min_Temp': [min_temp],
                'Rainfall': [rainfall],
                'Relative_Humidity': [humidity],
                'Wind_Speed': [wind_speed],
                'Cloud_Coverage': [cloud_cover],
                'Bright_Sunshine': [sunshine],
                'LATITUDE': [latitude],
                'LONGITUDE': [longitude],
                'ALT': [alt],
                'Period': [period]  # Now in correct numeric format
            })
            
            try:
                processed_data = preprocessor.transform(input_data)
                prediction = model.predict(processed_data)
                probability = model.predict_proba(processed_data)[0, 1]
                
                st.subheader("Result")
                if prediction[0] == 1:
                    st.error(f"🚨 Flood Risk: HIGH ({probability:.1%} probability)")
                    st.write("Critical risk factors:")
                    st.write(f"- Rainfall: {rainfall}mm (threshold exceeded)" if rainfall > 150 else "")
                    st.write(f"- Humidity: {humidity}% (prolonged saturation)" if humidity > 85 else "")
                else:
                    st.success(f"✅ Flood Risk: LOW ({probability:.1%} probability)")
                    
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
                st.info("""
                Common fixes:
                1. Verify all inputs are numbers
                2. Check model expects the same features
                3. Ensure preprocessor handles Period as float
                """)

if __name__ == '__main__':
    main()
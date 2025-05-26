import streamlit as st
import pandas as pd
import joblib
from geopy.distance import geodesic

# Load model
@st.cache_resource
def load_model():
    return joblib.load("best_model.pkl")

# Streamlit app
def main():
    st.title("📦 Amazon Delivery Time Prediction")
    st.write("Predict delivery time based on distance, weather, and traffic conditions.")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            store_lat = st.number_input("Store Latitude", format="%.6f")
            store_long = st.number_input("Store Longitude", format="%.6f")
            weather = st.selectbox("Weather Condition", ["Sunny", "Rainy", "Cloudy"])

        with col2:
            drop_lat = st.number_input("Drop Latitude", format="%.6f")
            drop_long = st.number_input("Drop Longitude", format="%.6f")
            traffic = st.selectbox("Traffic Condition", ["Low", "Medium", "High"])

        submitted = st.form_submit_button("Predict Delivery Time")

        if submitted:
            # Calculate distance
            distance = geodesic((store_lat, store_long), (drop_lat, drop_long)).km
            input_df = pd.DataFrame({
                'Distance': [distance],
                'Weather': [weather],
                'Traffic': [traffic]
            })

            # Load model and predict
            model = load_model()
            prediction = model.predict(input_df)[0]
            st.success(f"📦 Estimated Delivery Time: {prediction:.2f} hours")

if __name__ == "__main__":
    main()

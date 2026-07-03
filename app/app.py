import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("models/traffic_model.pkl")

# Page Configuration
st.set_page_config(
    page_title="Smart City Traffic Forecasting",
    page_icon="🚦",
    layout="centered"
)

# Title
st.title("🚦 Smart City Traffic Forecasting")

st.markdown("""
### Traffic Volume Prediction using Machine Learning

This application predicts traffic volume at different city junctions based on:
- Junction Number
- Date Information
- Time Information

**Model Used:** Random Forest Regressor  
**Model Accuracy (R² Score):** 96.9%
""")

st.divider()

# User Inputs
junction = st.selectbox(
    "Select Junction",
    [1, 2, 3, 4]
)

year = st.number_input(
    "Year",
    min_value=2015,
    max_value=2030,
    value=2017
)

month = st.slider(
    "Month",
    min_value=1,
    max_value=12,
    value=6
)

day = st.slider(
    "Day",
    min_value=1,
    max_value=31,
    value=15
)

hour = st.slider(
    "Hour",
    min_value=0,
    max_value=23,
    value=12
)

dayofweek = st.slider(
    "Day of Week (0 = Monday, 6 = Sunday)",
    min_value=0,
    max_value=6,
    value=0
)

# Weekend Feature
weekend = 1 if dayofweek >= 5 else 0

st.divider()

# Prediction Button
if st.button("🚀 Predict Traffic Volume"):

    input_data = pd.DataFrame({
        'Junction': [junction],
        'Year': [year],
        'Month': [month],
        'Day': [day],
        'Hour': [hour],
        'DayOfWeek': [dayofweek],
        'Weekend': [weekend]
    })

    prediction = model.predict(input_data)

    st.success(
        f"🚗 Predicted Traffic Volume: {prediction[0]:.2f} vehicles"
    )

    # Traffic Level Indicator
    if prediction[0] < 10:
        st.info("🟢 Low Traffic Expected")
    elif prediction[0] < 30:
        st.warning("🟡 Moderate Traffic Expected")
    else:
        st.error("🔴 Heavy Traffic Expected")

st.divider()

st.markdown("""
### Project Information

**Project Title:** Smart City Traffic Forecasting

**Domain:** Data Science & Machine Learning

**Tech Stack:**
- Python
- Pandas
- Scikit-Learn
- Matplotlib
- Streamlit

**Developed By:** Shaik Mahek Sharfuddin
""")
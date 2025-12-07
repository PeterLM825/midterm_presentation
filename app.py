import streamlit as st
import joblib
import pandas as pd

# ---------------------------
# Load Uber & Lyft models
# ---------------------------
@st.cache_resource
def load_models():
    with open("temp_impact_cab_price.pkl", "rb") as f:
        models = joblib.load(f)
    return models["uber_model"], models["lyft_model"]

uber_model, lyft_model = load_models()

# Sidebar
st.sidebar.image("midterm/images (4).jpg", width=150)
st.sidebar.title("About Me")
st.sidebar.write("Name: John Doe")
st.sidebar.write("Role: Data Scientist")
st.sidebar.write("Email: john@example.com")
st.sidebar.write("Description: I build ML apps!")

st.title("🚕 Uber/Lyft Price Prediction Based on Temperature")

# ------- USER INPUTS -------
st.header("Enter Ride Details")

cab_type = st.selectbox("Cab Type", ["Uber", "Lyft"])

# Dynamic car type options
uber_car_types = ["UberX", "UberXL", "Uber Black", "Uber Black SUV"]
lyft_car_types = ["Lyft", "Lyft XL", "Lux", "Lux Black", "Lux Black XL"]

if cab_type == "Uber":
    car_type_options = uber_car_types
else:
    car_type_options = lyft_car_types

locations = [
    "Back Bay", "Beacon Hill", "Boston University", "Fenway",
    "Financial District", "Haymarket Square", "North End",
    "North Station", "Northeastern University", "South Station",
    "Theatre District", "West End"
]

source = st.selectbox("Pickup (source)", locations)
destination = st.selectbox("Drop-off (destination)", locations)

name = st.selectbox(
    "Car Type (based on cab selection)",
    car_type_options
)

temp = st.number_input("Temperature (°F)", min_value=10.0, max_value=95.0, step=1.0)

day_time = st.selectbox(
    "Time of Day",
    ["Morning", "Afternoon", "Evening", "Late Night"]
)

# ------- CREATE INPUT DATAFRAME -------
input_df = pd.DataFrame({
    "cab_type": [cab_type],
    "source": [source],
    "destination": [destination],
    "name": [name],
    "temp": [temp],
    "day_time": [day_time]
})

st.write("### Input Data Preview")
st.write(input_df)

# ---------------------------
# Prediction Button
# ---------------------------
if st.button("Predict Price", key="predict_button_main"):

    if source == destination:
        st.info("Pickup and drop-off are the same location.")
        st.success("Predicted Price: **$0.00**")
    else:

        if cab_type == "Uber":
            price = uber_model.predict(input_df)[0]
            st.subheader("Results")
            st.success(f"Uber Price Prediction: **${price:.2f}**")

        else:  # Lyft
            price = lyft_model.predict(input_df)[0]
            st.subheader("Results")
            st.success(f"Lyft Price Prediction: **${price:.2f}**")



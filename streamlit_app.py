import streamlit as st
import joblib
import numpy as np

# Load saved components
model = joblib.load('flight_price_model.pkl')
scaler = joblib.load('flight_price_scaler.pkl')
encoder = joblib.load('flight_price_encoder.pkl')

# Display the number of features the model expects
st.write("Model expects", model.n_features_in_, "features")

# App title and description
st.title("Flight Price Prediction")
st.write("Enter the flight details below to predict the ticket price.")

# Inputs for the prediction
# For demonstration, these are sample inputs.
# You need to adjust the options and mappings to reflect those used during training.
airline = st.selectbox("Airline", ["SpiceJet", "AirAsia", "Vistara", "GO_FIRST"])
source_city = st.selectbox("Source City", ['Delhi','Mumbai', 'Bangalore', 'Kolkata', 'Hyderabad', 'Chennai'])  
departure_time = st.selectbox("Departure Time", ["Evening", "Early_Morning", "Morning", "Afternoon"])
stops = st.selectbox("Stops", ["zero", "one", "two", "three"])
arrival_time = st.selectbox("Arrival Time", ["Night", "Morning", "Early_Morning", "Afternoon", "Evening"])
destination_city = st.selectbox("Destination City", ['Delhi', 'Mumbai', 'Bangalore', 'Kolkata', 'Hyderabad', 'Chennai'])  
flight_class = st.selectbox("Class", ["Economy", "Business"])
duration = st.number_input("Duration (in hours)", min_value=0.0, value=2.0, step=0.1)
days_left = st.number_input("Days Left", min_value=0, value=1, step=1)

# Mapping dictionaries based on your training (customize these to match your LabelEncoder mappings)
airline_mapping = {"SpiceJet": 0, "AirAsia": 1, "Vistara": 2, "GO_FIRST": 3}
source_city_mapping = {
    "Delhi": 0,
    "Mumbai": 1,
    "Bangalore": 2,
    "Kolkata": 3,
    "Hyderabad": 4,
    "Chennai": 5
}
departure_time_mapping = {"Evening": 0, "Early_Morning": 1, "Morning": 2, "Afternoon": 3}
stops_mapping = {"zero": 0, "one": 1, "two": 2, "three": 3}
arrival_time_mapping = {"Night": 0, "Morning": 1, "Early_Morning": 2, "Afternoon": 3, "Evening": 4}
destination_city_mapping = {
    "Delhi": 0,
    "Mumbai": 1,
    "Bangalore": 2,
    "Kolkata": 3,
    "Hyderabad": 4,
    "Chennai": 5
}
class_mapping = {"Economy": 0, "Business": 1}

# Encode the inputs using the dictionaries above
# (In a more complex case, you might use the loaded encoder but here we show manual mappings for clarity.)
encoded_airline = airline_mapping[airline]
encoded_source_city = source_city_mapping[source_city]
encoded_departure_time = departure_time_mapping[departure_time]
encoded_stops = stops_mapping[stops]
encoded_arrival_time = arrival_time_mapping[arrival_time]
encoded_destination_city = destination_city_mapping[destination_city]
encoded_class = class_mapping[flight_class]

# Prepare the input data array in the same order as training features
# Order: airline, source_city, departure_time, stops, arrival_time, destination_city, class, duration, days_left
input_data = np.array([
    encoded_airline,
    encoded_source_city,
    encoded_departure_time,
    encoded_stops,
    encoded_arrival_time,
    encoded_destination_city,
    encoded_class,
    duration,
    days_left
]).reshape(1, -1)

# Scale the numeric features using the same scaler used in training
input_data_scaled = scaler.transform(input_data)

# Predict the flight price when the "Predict Price" button is pressed
if st.button("Predict Price"):
    prediction = model.predict(input_data_scaled)
    st.success(f"The predicted flight price is: Rs. {prediction[0]:.2f}")

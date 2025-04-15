import streamlit as st
import joblib
import numpy as np
import os
from pathlib import Path

# Set page config for better appearance
st.set_page_config(page_title="Flight Price Predictor", page_icon="✈️", layout="wide")

# Custom CSS for better styling
st.markdown("""
<style>
    .stNumberInput, .stSelectbox {
        margin-bottom: 1rem;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_components():
    """Load model components with error handling"""
    try:
        # Use Path for more reliable file paths
        model_path = Path('flight_price_model.pkl')
        scaler_path = Path('flight_price_scaler.pkl')
        encoder_path = Path('flight_price_encoder.pkl')
        
        if not all([model_path.exists(), scaler_path.exists(), encoder_path.exists()]):
            st.error("Model files not found! Please ensure all .pkl files are in the correct directory.")
            return None, None, None
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        encoder = joblib.load(encoder_path)
        return model, scaler, encoder
        
    except Exception as e:
        st.error(f"Error loading model components: {str(e)}")
        return None, None, None

# Load components
model, scaler, encoder = load_components()

# App title and description
st.title("✈️ Flight Price Prediction")
st.write("Enter the flight details below to predict the ticket price.")

# Only show inputs if model loaded successfully
if model is not None:
    # Display model info in expander
    with st.expander("Model Information", expanded=False):
        st.write(f"Model expects {model.n_features_in_} features")
        st.write(f"Model type: {type(model).__name__}")
    
    # Create two columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        airline = st.selectbox("Airline", ["SpiceJet", "AirAsia", "Vistara", "GO_FIRST"])
        source_city = st.selectbox("Source City", ['Delhi', 'Mumbai', 'Bangalore', 'Kolkata', 'Hyderabad', 'Chennai'])
        departure_time = st.selectbox("Departure Time", ["Evening", "Early_Morning", "Morning", "Afternoon", "Night"])
        stops = st.selectbox("Stops", ["zero", "one", "two", "three"])
        flight_class = st.selectbox("Class", ["Economy", "Business"])
    
    with col2:
        destination_city = st.selectbox("Destination City", ['Delhi', 'Mumbai', 'Bangalore', 'Kolkata', 'Hyderabad', 'Chennai'])
        arrival_time = st.selectbox("Arrival Time", ["Night", "Morning", "Early_Morning", "Afternoon", "Evening"])
        duration = st.number_input("Duration (hours)", min_value=0.0, value=2.0, step=0.1, help="Flight duration in hours")
        days_left = st.number_input("Days Until Departure", min_value=0, value=1, step=1, help="Number of days until flight departure")
    
    # Mapping dictionaries (should match your training)
    mapping_dicts = {
        "airline": {"SpiceJet": 0, "AirAsia": 1, "Vistara": 2, "GO_FIRST": 3},
        "source_city": {"Delhi": 0, "Mumbai": 1, "Bangalore": 2, "Kolkata": 3, "Hyderabad": 4, "Chennai": 5},
        "departure_time": {"Evening": 0, "Early_Morning": 1, "Morning": 2, "Afternoon": 3, "Night": 4},
        "stops": {"zero": 0, "one": 1, "two": 2, "three": 3},
        "arrival_time": {"Night": 0, "Morning": 1, "Early_Morning": 2, "Afternoon": 3, "Evening": 4},
        "destination_city": {"Delhi": 0, "Mumbai": 1, "Bangalore": 2, "Kolkata": 3, "Hyderabad": 4, "Chennai": 5},
        "class": {"Economy": 0, "Business": 1}
    }
    
    # Prepare input data
    try:
        input_data = np.array([
            mapping_dicts["airline"][airline],
            mapping_dicts["source_city"][source_city],
            mapping_dicts["departure_time"][departure_time],
            mapping_dicts["stops"][stops],
            mapping_dicts["arrival_time"][arrival_time],
            mapping_dicts["destination_city"][destination_city],
            mapping_dicts["class"][flight_class],
            duration,
            days_left
        ]).reshape(1, -1)
        
        # Scale the input data
        input_data_scaled = scaler.transform(input_data)
        
        # Prediction button with better styling
        if st.button("Predict Price", type="primary", use_container_width=True):
            with st.spinner("Calculating..."):
                prediction = model.predict(input_data_scaled)
                st.balloons()
                
                # Display prediction with nice formatting
                st.markdown(f"""
                <div class="prediction-box">
                    <h3>Prediction Result</h3>
                    <p style="font-size: 24px; color: #0068c9;">
                        Estimated Price: <strong>₹{prediction[0]:,.2f}</strong>
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
    except Exception as e:
        st.error(f"Error processing input: {str(e)}")

else:
    st.warning("""
    The model is not loaded correctly. Please ensure:
    1. All required .pkl files are in the correct directory
    2. The files are named correctly (flight_price_model.pkl, flight_price_scaler.pkl, flight_price_encoder.pkl)
    3. The joblib package is installed (pip install joblib)
    """)

# Add footer
st.markdown("---")
st.caption("Flight Price Prediction App | Made with Streamlit")
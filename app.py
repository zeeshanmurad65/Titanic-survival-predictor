import streamlit as st
import pandas as pd
import pickle
import os

# --- 1. CONFIGURATION & SETUP ---
st.set_page_config(page_title="Titanic AI", page_icon="🚢", layout="centered")

# Custom CSS to make it look nicer
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
    }
    .success-msg {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        color: #155724;
        font-weight: bold;
        text-align: center;
    }
    .error-msg {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8d7da;
        color: #721c24;
        font-weight: bold;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# --- 2. LOAD SAVED MODEL ---
@st.cache_resource
def load_model():
    # Make sure 'titanic_model.pkl' is in the same folder as this script
    if not os.path.exists('titanic_model.pkl'):
        st.error("Error: 'titanic_model.pkl' not found. Please save your model first!")
        return None
    
    with open('titanic_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# --- 3. TITLE & DESCRIPTION ---
st.title("🚢 Titanic Survival Predictor")
st.write("Enter passenger details to predict if they would have survived the Titanic disaster.")
st.markdown("---")

# --- 4. USER INPUTS ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Passenger Info")
    
    # Sex Input
    sex_input = st.selectbox("Gender", ["Male", "Female"])
    
    # Age Input
    age_input = st.number_input("Age", min_value=0.0, max_value=100.0, value=30.0, step=1.0)
    
    # Pclass Input
    pclass_input = st.radio("Ticket Class", ["1st Class (Upper)", "2nd Class (Middle)", "3rd Class (Lower)"])

with col2:
    st.subheader("Trip Details")
    
    # Fare Input
    fare_input = st.number_input("Ticket Fare ($)", min_value=0.0, max_value=600.0, value=32.0, step=0.1)
    
    # Embarked Input
    embarked_input = st.selectbox("Port of Embarkation", ["Southampton (S)", "Cherbourg (C)", "Queenstown (Q)"])

# --- 5. PREDICTION LOGIC ---
if st.button("Predict Survival"):
    if model:
        # A. Manual Mapping (Matching LabelEncoder Logic)
        # Sex: Female=0, Male=1 (Alphabetical order)
        sex_map = {"Female": 0, "Male": 1}
        sex_val = sex_map[sex_input]
        
        # Embarked: C=0, Q=1, S=2 (Alphabetical order)
        emb_map = {"Cherbourg (C)": 0, "Queenstown (Q)": 1, "Southampton (S)": 2}
        emb_val = emb_map[embarked_input]
        
        # Pclass: 1, 2, 3
        pclass_map = {"1st Class (Upper)": 1, "2nd Class (Middle)": 2, "3rd Class (Lower)": 3}
        pclass_val = pclass_map[pclass_input]
        
        # B. Create DataFrame
        # IMPORTANT: These column names must match your training data exactly!
        input_data = pd.DataFrame({
            'Pclass': [pclass_val],
            'Sex': [sex_val],
            'Age': [age_input],
            'Fare': [fare_input],
            'Embarked': [emb_val]
        })
        
        # C. Predict
        try:
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1] # Probability of "1" (Survival)
            
            # D. Display Results
            st.markdown("---")
            if prediction == 1:
                st.markdown(f'<div class="success-msg">🎉 Prediction: SURVIVED<br>Confidence: {probability:.1%}</div>', unsafe_allow_html=True)
                st.balloons()
            else:
                st.markdown(f'<div class="error-msg">💀 Prediction: DID NOT SURVIVE<br>Survival Chance: {probability:.1%}</div>', unsafe_allow_html=True)
                
            # Optional: Show the data used for prediction
            with st.expander("See Raw Data"):
                st.dataframe(input_data)
                
        except Exception as e:
            st.error(f"Error during prediction: {e}")
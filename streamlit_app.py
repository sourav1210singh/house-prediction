import streamlit as st
import numpy as np
import catboost
import pickle
from pyngrok import ngrok

# Load the trained model
with open("catboost_model.pkl", "rb") as file:
    loaded_model = pickle.load(file)

# Title of the app
st.title("🏠 House Price Prediction App")

st.markdown("### Enter the details below to predict house prices.")

# Define feature names
feature_names = ["CRIM", "INDUS", "NOX", "RM", "AGE", "DIS", "TAX", "PTRATIO", "B", "LSTAT"]

# Collect user input for each feature (manual input without min/max)
user_input = []
for feature in feature_names:
    value = st.number_input(f"Enter {feature}:", value=0.0)
    user_input.append(value)

# Predict button
if st.button("Predict"):

    if max(user_input)==0:
        st.error("Provide Input First !")
    else:
        # Convert input to NumPy array and reshape
        user_array = np.array(user_input).reshape(1, -1)
        
        # Get prediction
        prediction = loaded_model.predict(user_array)[0]
        
        # Display result
        st.success(f"🏡 Predicted House Price: ${prediction * 1000:.2f}")


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import joblib 
import numpy as np 
import streamlit as st 

# Load Model and Scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Set Page Layout
st.set_page_config('Diabetes Prediction', page_icon="🩺")

st.title("🩺 Diabetes Prediction")

st.write("Enter the Features Value Below for Diabetes Prediction")

# User Inputs 
pregnancies = st.number_input('Pregnancies', min_value=0.0, format="%.2f")
glucose = st.number_input('Glucose', min_value=0.0, format="%.2f")
BMI = st.number_input('BMI', min_value=0.0, format="%.2f")
age = st.number_input('Age', min_value=0.0, format= "%.2f")
diabetespedigreefunction = st.number_input("Diabetes Pedigree Function", min_value=0.0, format="%.4f")

# bloodpressure = st.number_input('Blood Pressure', min_value=0.0, format="%.2f")
# skinthickness = st.number_input("Skin Thickness", min_value=0.0, format="%.2f")
# insulin = st.number_input("Insulin", min_value=0.0, format="%.4f")

if st.button("Predict Diabeties"):

    input_data = np.array([[
        pregnancies,
        glucose,
        BMI,
        age,
        diabetespedigreefunction       
    ]])

    # scaled the input
    input_scaled = scaler.transform(input_data)

    # predict the model 
    prediction = model.predict(input_scaled)

    # Display Result
    if prediction[0]==1:
        st.error("The person is Diabetic.")
    elif prediction[0]==0:
        st.success("The person is not Diabetic.")
    else: 
        st.error("Something went wrong")


# In[ ]:





# In[ ]:





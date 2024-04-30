#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import OneHotEncoder

model = joblib.load('XGB_model.pkl')
gender_encoded = joblib.load('gender_encode.pkl')
geography_encoded = joblib.load('geography_oneHot.pkl')

def main():
    st.title("Churn Prediction - Model Deployment")
    Gender = st.radio("Gender", ['Male', 'Female'])
    Age = st.number_input("Age", 0, 100)
    Geography = st.radio("Country", ['France', 'Spain', 'Germany'])
    Tenure = st.number_input("The period of time you holds a position (in years)", 0, 100)
    IsActiveMember = st.radio("Choose status member", 0, 1)
    HasCrCard = st.radio('Do you have a Credit Card', 0, 1)
    CreditScore = st.number_input('Total Credit Score', 0, 1000)
    EstimatedSalary = st.number_input('Number of your estimated salary', 0, 1000000000)
    Balance = st.number_input('Total Balance', 0, 10000000000)
    NumOfProducts = st.number_input('Number of products', 0, 100)

    data = {'Gender': Gender, 'Age': int(Age), 'Geography': Geography, 'Tenure': int(Tenure),
         'IsActiveMember': int(IsActiveMember), 'HasCrCard': int(HasCrCard), 'CreditScore': int(CreditScore),
         'EstimatedSalary': int(EstimatedSalary), 'Balance': int(Balance),
          'NumOfProducts': int(NumOfProducts)}

    df = pd.DataFrame([list(data.values())], columns=['Gender', 'Age', 'Geography', 'Tenure',
                                                      'IsActiveMember', 'HasCrCard', 'CreditScore',
                                                     'EstimatedSalary', 'Balance', 'NumOfProducts'])
    
    # Encoding categorical variables
    df = df.replace(gender_encoded)
    df = df.replace(geography_encoded)

    if st.button('Make Prediction'):
        features=df  
        result = make_prediction(features)
        st.success(f'The prediction is: {result}')

def make_prediction(features):
    input_array = np.array(features).reshape(1, -1)
    prediction = model.predict(input_array)
    return prediction[0]

if __name__ == '__main__':
    main()


import streamlit as st
import pandas as pd 
import numpy as np
import pickle
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

model = tf.keras.models.load_model('churn_model.h5')

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)   
with open('label_encoder.pkl', 'rb') as le_file:
    label_encoder = pickle.load(le_file)
with open('ohe.pkl', 'rb') as ohe_file:
    ohe = pickle.load(ohe_file) 

st.title("Customer Churn Prediction")

geography = st.selectbox("Select Geography", ohe.categories_[0])
gender = st.selectbox("Select Gender", label_encoder.classes_)
age = st.number_input("Enter Age", min_value=18, max_value=100, value=30)
balance = st.number_input("Enter Balance", min_value=0.0, value=1000.0)
credit_score = st.number_input("Enter Credit Score", min_value=300, max_value=850, value=600)
estimated_salary = st.number_input("Enter Estimated Salary")
tenure = st.slider('Tenure', 0, 1)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has credit card', [0, 1])
is_active_member = st.selectbox('Is active member', [0, 1])

input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender' : [label_encoder.transform([gender])[0]],
    'Age' : [age],
    'Tenure' : [tenure],
    'Balance' : [balance],
    'NumOfProducts' : [num_of_products],
    'HasCrCard' : [has_cr_card],
    'IsActiveMember' : [is_active_member],
    'EstimatedSalary' : [estimated_salary]
})

geo_encoded = ohe.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=ohe.get_feature_names_out(['Geography']))

input_data = pd.concat([input_data.reset_index(drop = True), geo_encoded_df], axis = 1)

input_data_scaled = scaler.transform(input_data)

prediction = model.predict(input_data_scaled)
prediction_prob = prediction[0][0]

st.write(f'Churn Probability: {prediction_prob:.2f}')
if prediction_prob > 0.5:
    st.write('The customer is likey to churn')
else:
    st.write('The customer is not likely to churn')
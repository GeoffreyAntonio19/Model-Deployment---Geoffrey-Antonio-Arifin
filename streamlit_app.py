import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.title('Obesity Classification Prediction')
st.info('This application can predict your obesity class based on some parameters!')

with st.expander('**Data**'):
  st.write('This is a raw data')
  df = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')
  df

with st.expander('**Data Visualization**'):
  st.scatter_chart(data=df, x='Height', y='Weight', color='NObeyesdad')

st.subheader('**Input Features**')
gender = st.selectbox("Gender", ("Male", "Female"))
age = st.slider("Age (years)", min_value=0, max_value=70, value=40)
height = st.slider("Height (meters)", min_value=1.30, max_value=2.00, value=1.65)
weight = st.slider("Weight (kg)", min_value=30.00, max_value=200.00, value=70.00)
family_history = st.selectbox("Family History with Overweight", ("Yes", "No"))
favc = st.selectbox("Frequent High Caloric Food Consumption (FAVC)", ("Yes", "No"))
fcvc = st.slider("Frequency of Vegetable Consumption (FCVC)", min_value=1.00, max_value=3.00, value=2.00)
ncp = st.slider("Number of Main Meals per Day (NCP)", min_value=1.00, max_value=4.00, value=2.50)
caec = st.selectbox("Consumption of Alcohol (CAEC)", ("Sometimes", "Frequently", "Always", "No"))
smoke = st.selectbox("Smoking Habit (SMOKE)", ("Yes", "No"))
ch2o = st.slider("Daily Water Consumption (CH2O, in liters)", min_value=1.00, max_value=3.00, value=2.00)
scc = st.selectbox("Caloric Consumption Monitoring (SCC)", ("Yes", "No"))
faf = st.slider("Physical Activity Frequency (FAF, times per week)", min_value=0.00, max_value=3.00, value=1.50)
tue = st.slider("Time Using Electronic Devices (TUE, hours per day)", min_value=0.00, max_value=2.00, value=1.00)
calc = st.selectbox("Consumption of Alcoholic Beverages (CALC)", ("Yes", "No"))
mtrans = st.selectbox(
    "Main Mode of Transportation (MTRANS)", 
    ("Public Transportation", "Walking", "Automobile", "Motorbike", "Bike")
)

# Menyimpan input dalam DataFrame
user_data = pd.DataFrame({
    "Gender": [gender],
    "Age (years)": [age],
    "Height (m)": [height],
    "Weight (kg)": [weight],
    "Family History with Overweight": [family_history],
    "Frequent High Caloric Food (FAVC)": [favc],
    "Vegetable Consumption Frequency (FCVC)": [fcvc],
    "Number of Main Meals (NCP)": [ncp],
    "Alcohol Consumption (CAEC)": [caec],
    "Smoking Habit (SMOKE)": [smoke],
    "Daily Water Consumption (CH2O, liters)": [ch2o],
    "Caloric Consumption Monitoring (SCC)": [scc],
    "Physical Activity Frequency (FAF, per week)": [faf],
    "Time Using Electronic Devices (TUE, hours/day)": [tue],
    "Alcoholic Beverage Consumption (CALC)": [calc],
    "Mode of Transportation (MTRANS)": [mtrans]
})

# Menampilkan data dalam bentuk tabel
st.subheader("Data Input by User")
st.dataframe(user_data, use_container_width=True)

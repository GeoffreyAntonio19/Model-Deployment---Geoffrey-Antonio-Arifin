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

st.write('**Input Features**')
gender = st.selectbox('Gender', ('Male', 'Female'))
age = st.slider('Age', min_value = 0, max_value = 70, value = 40)
height = st.slider('Height', min_value = 1.30, max_value = 2.00, value = 1.65)
weight = st.slider('Weight', min_value = 30, max_value = 200, value = 70)
family_history = st.selectbox('Family Member History with Overweight', ('Yes', 'No'))
favc = st.selectbox('FAVC', ('Yes', 'No'))
caec = st.selectbox('CAEC', ('Sometimes', 'Frequently', 'Always', 'No'))
smoke = st.selectbox('SMOKE', ('Yes', 'No'))
scc = st.selectbox('SCC', ('Yes', 'No'))
calc = st.selectbox('CALC', ('Yes', 'No'))
mtrans = st.selectbox('MTRANS', ('Public Transportation', 'Walking', 'Automobile', 'Motorbike', 'Bike'))

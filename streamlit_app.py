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
weight = st.slider('Weight', min_value = 30.00, max_value = 200.00, value = 70.00)
family_history = st.selectbox('Family Member History with Overweight', ('Yes', 'No'))
favc = st.selectbox('FAVC', ('Yes', 'No'))
fcvc = st.slider('FCVC', min_value = 1.00, max_value = 3.00, value = 2.00)
ncp = st.slider('NCP', min_value = 1.00, max_value = 4.00, value = 2.50)
caec = st.selectbox('CAEC', ('Sometimes', 'Frequently', 'Always', 'No'))
smoke = st.selectbox('SMOKE', ('Yes', 'No'))
ch2o = st.slider('CH2O', min_value = 1.00, max_value = 3.00, value = 2.00)
scc = st.selectbox('SCC', ('Yes', 'No'))
faf = st.slider('FAF', min_value = 0.00, max_value = 3.00, value = 1.50)
tue = st.slider('TUE', min_value = 0.00, max_value = 2.00, value = 1.00)
calc = st.selectbox('CALC', ('Yes', 'No'))
mtrans = st.selectbox('MTRANS', ('Public Transportation', 'Walking', 'Automobile', 'Motorbike', 'Bike'))

if 'user_inputs' not in st.session_state:
    st.session_state['user_inputs'] = {}

st.session_state['user_inputs'] = {
    "Gender": gender,
    "Age": age,
    "Height": height,
    "Weight": weight,
    "Family History with Overweight": family_history,
    "FAVC": favc,
    "FCVC": fcvc,
    "NCP": ncp,
    "CAEC": caec,
    "SMOKE": smoke,
    "CH2O": ch2o,
    "SCC": scc,
    "FAF": faf,
    "TUE": tue,
    "CALC": calc,
    "MTRANS": mtrans
}

st.subheader("Data Input User:")
st.write(st.session_state['user_inputs'])

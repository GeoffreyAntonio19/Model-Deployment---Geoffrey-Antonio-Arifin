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

with st.expander('**Input Features**'):
  st.header('Input Categorical Features')
  gender = st.selectbox('1. Gender', ('Male', 'Female'))

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

st.markdown(
    """
    <style>
    .streamlit-expanderContent {
        max-height: 100px; /* Sesuaikan tinggi */
        overflow-y: auto;
    }
    </style>
    """,
    unsafe_allow_html=True
)

with st.container():
    st.subheader('Input Features')  # Menambahkan judul bagian
    
    st.write('**Categorical Features**')
    gender = st.selectbox('Gender', ('Male', 'Female'))
    family_history = st.selectbox('Family Member History with Overweight', ('Yes', 'No'))
    favc = st.selectbox('FAVC', ('Yes', 'No'))
    caec = st.selectbox('CAEC', ('Sometimes', 'Frequently', 'Always', 'No'))
    smoke = st.selectbox('SMOKE', ('Yes', 'No'))
    scc = st.selectbox('SCC', ('Yes', 'No'))
    calc = st.selectbox('CALC', ('Yes', 'No'))
    mtrans = st.selectbox('MTRANS', ('Yes', 'No'))


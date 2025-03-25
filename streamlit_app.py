taskkill /IM streamlit.exe /F
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# ---- Title ----
st.title('Obesity Classification Prediction')
st.info('This application predicts your obesity class based on some health and lifestyle parameters.')

# ---- Load Data ----
@st.cache_data
def load_data():
    df = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')
    return df

df = load_data()

# ---- Data Exploration ----
with st.expander('**Dataset Preview**'):
    st.write('This is the raw dataset:')
    st.dataframe(df)

with st.expander('**Data Visualization**'):
    st.scatter_chart(data=df, x='Height', y='Weight', color='NObeyesdad')

# ---- Data Preprocessing ----
def preprocess_data(df):
    label_encoders = {}

    # Pastikan nama kolom yang sesuai dengan dataset
    categorical_columns = [
        "Gender", "family_history_with_overweight", "FAVC", "CAEC", 
        "SMOKE", "SCC", "CALC", "MTRANS", "NObeyesdad"
    ]

    # Proses label encoding untuk kolom kategorikal
    for col in categorical_columns:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le

    return df, label_encoders

df, label_encoders = preprocess_data(df)

# ---- Train Model ----
X = df.drop(columns=["NObeyesdad"])  # Features
y = df["NObeyesdad"]  # Target variable

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ---- User Input Features ----
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
mtrans = st.selectbox("Main Mode of Transportation (MTRANS)", ("Public Transportation", "Walking", "Automobile", "Motorbike", "Bike"))

# ---- Convert Input Data to DataFrame ----
user_data = pd.DataFrame({
    "Gender": [gender],
    "Age": [age],
    "Height": [height],
    "Weight": [weight],
    "family_history_with_overweight": [family_history],
    "FAVC": [favc],
    "FCVC": [fcvc],
    "NCP": [ncp],
    "CAEC": [caec],
    "SMOKE": [smoke],
    "CH2O": [ch2o],
    "SCC": [scc],
    "FAF": [faf],
    "TUE": [tue],
    "CALC": [calc],
    "MTRANS": [mtrans]
})

# ---- Encode User Input ----
for col in label_encoders:
    if col in user_data.columns:
        user_data[col] = label_encoders[col].transform(user_data[col])

# ---- Display User Input ----
st.subheader("User Input Data")
st.dataframe(user_data, use_container_width=True)

# ---- Make Prediction ----
if st.button("Predict Obesity Class"):
    prediction = model.predict(user_data)
    predicted_class = label_encoders["NObeyesdad"].inverse_transform(prediction)[0]
    st.success(f"Predicted Obesity Class: **{predicted_class}**")

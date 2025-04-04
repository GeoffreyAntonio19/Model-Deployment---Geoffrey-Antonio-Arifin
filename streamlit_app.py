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
    categorical_columns = [
        "Gender", "family_history_with_overweight", "FAVC", "CAEC", 
        "SMOKE", "SCC", "CALC", "MTRANS", "NObeyesdad"
    ]

    for col in categorical_columns:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le

    return df, label_encoders

df, label_encoders = preprocess_data(df)

# ---- Train Model ----
X = df.drop(columns=["NObeyesdad"])
y = df["NObeyesdad"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ---- User Input Features ----
st.subheader('**Input Features**')
user_data = pd.DataFrame({
    "Gender": [st.selectbox("Gender", ("Male", "Female"))],
    "Age": [st.slider("Age (years)", min_value=0, max_value=70, value=40)],
    "Height": [st.slider("Height (meters)", min_value=1.30, max_value=2.00, value=1.65)],
    "Weight": [st.slider("Weight (kg)", min_value=30.00, max_value=200.00, value=70.00)],
    "family_history_with_overweight": [st.selectbox("Family History with Overweight", ("yes", "no"))],
    "FAVC": [st.selectbox("Frequent High Caloric Food Consumption (FAVC)", ("yes", "no"))],
    "FCVC": [st.slider("Frequency of Vegetable Consumption (FCVC)", min_value=1.00, max_value=3.00, value=2.00)],
    "NCP": [st.slider("Number of Main Meals per Day (NCP)", min_value=1.00, max_value=4.00, value=2.50)],
    "CAEC": [st.selectbox("Consumption of Alcohol (CAEC)", ("Sometimes", "Frequently", "Always", "no"))],
    "SMOKE": [st.selectbox("Smoking Habit (SMOKE)", ("yes", "no"))],
    "CH2O": [st.slider("Daily Water Consumption (CH2O, in liters)", min_value=1.00, max_value=3.00, value=2.00)],
    "SCC": [st.selectbox("Caloric Consumption Monitoring (SCC)", ("yes", "no"))],
    "FAF": [st.slider("Physical Activity Frequency (FAF, times per week)", min_value=0.00, max_value=3.00, value=1.50)],
    "TUE": [st.slider("Time Using Electronic Devices (TUE, hours per day)", min_value=0.00, max_value=2.00, value=1.00)],
    "CALC": [st.selectbox("Consumption of Alcoholic Beverages (CALC)", ("yes", "no"))],
    "MTRANS": [st.selectbox("Main Mode of Transportation (MTRANS)", ("Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"))]
})

# ---- Display User Input (Original Data) ----
st.subheader("User Input Data")
user_data_original = user_data.copy() # Simpan data asli sebelum encoding
st.dataframe(user_data_original, use_container_width=True)

# ---- Encode User Input (One-by-One Handling) ----
user_data_encoded = user_data.copy()
for col in label_encoders:
    if col in user_data_encoded.columns:
        le = label_encoders[col]
        if user_data_encoded[col][0] not in le.classes_:
            user_data_encoded[col] = le.transform([le.classes_[0]])  # Gunakan default (kelas pertama)
        else:
            user_data_encoded[col] = le.transform(user_data_encoded[col])

# ---- Make Prediction ----
if st.button("Predict Obesity Class"):
    # Prediksi kelas dan probabilitas
    prediction_proba = model.predict_proba(user_data_encoded)
    prediction = model.predict(user_data_encoded)
    predicted_class = label_encoders["NObeyesdad"].inverse_transform(prediction)[0]

    # Ambil nama kelas dari encoder
    class_names = label_encoders["NObeyesdad"].classes_

    # Buat dataframe probabilitas dengan nama kelas sebagai header
    df_proba = pd.DataFrame(prediction_proba, columns=class_names)

    # Tampilkan hasil prediksi dan tabel probabilitas
    st.subheader("Obesity Prediction")
    st.dataframe(df_proba.style.format("{:.4f}"), use_container_width=True)

    st.success(f"Predicted Obesity Class: **{predicted_class}**")

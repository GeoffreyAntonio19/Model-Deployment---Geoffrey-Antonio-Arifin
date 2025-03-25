import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# ---- Title ----
st.title('Obesity Classification Prediction')

# ---- Load Data ----
@st.cache_data
def load_data():
    df = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')
    return df

df = load_data()

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

# ---- Function to Handle Unknown Categories ----
def encode_input(value, le):
    if value in le.classes_:
        return le.transform([value])[0]
    else:
        return le.transform([le.classes_[0]])[0]  # Default ke nilai pertama

# ---- User Input Features ----
st.subheader('**Input Features**')
gender = st.selectbox("Gender", ("Male", "Female"))
mtrans = st.selectbox("Main Mode of Transportation (MTRANS)", ("Public Transportation", "Walking", "Automobile", "Motorbike", "Bike"))

# ---- Convert Input Data to DataFrame ----
user_data = pd.DataFrame({"Gender": [gender], "MTRANS": [mtrans]})

# ---- Encode User Input ----
for col in label_encoders:
    if col in user_data.columns:
        user_data[col] = user_data[col].apply(lambda x: encode_input(x, label_encoders[col]))

# ---- Make Prediction ----
if st.button("Predict Obesity Class"):
    prediction = model.predict(user_data)
    predicted_class = label_encoders["NObeyesdad"].inverse_transform(prediction)[0]
    st.success(f"Predicted Obesity Class: **{predicted_class}**")

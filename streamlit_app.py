import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import os

# Load Dataset
class DataHandler:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.label_encoders = {}
        self.scaler = MinMaxScaler()
    
    def load_data(self):
        self.data = pd.read_csv(self.file_path)
        return self.data
    
    def preprocess_data(self):
        """ Preprocessing tanpa mengubah label sebelum prediksi """
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col != "NObeyesdad":  # Jangan transform kolom target dulu
                self.label_encoders[col] = LabelEncoder()
                self.data[col] = self.label_encoders[col].fit_transform(self.data[col])
        
        numerical_cols = self.data.select_dtypes(include=['int64', 'float64']).columns
        self.data[numerical_cols] = self.scaler.fit_transform(self.data[numerical_cols])
    
    def transform_user_input(self, user_df):
        """ Transformasi input user agar sesuai dengan format model """
        for col in self.label_encoders:
            user_df[col] = self.label_encoders[col].transform(user_df[col])
        return self.scaler.transform(user_df)

# Load Trained Model
class ModelHandler:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)
    
    def predict(self, features):
        """ Pastikan input berbentuk array 2D """
        features = features.values.reshape(1, -1)
        prediction_prob = self.model.predict_proba(features)[0]
        predicted_class = self.model.classes_[np.argmax(prediction_prob)]
        return predicted_class, prediction_prob

# Streamlit App Class
class ObesityClassificationApp:
    def __init__(self, data_handler, model_handler):
        self.data_handler = data_handler
        self.model_handler = model_handler
    
    def display_raw_data(self):
        st.subheader("1. Menampilkan Raw Data")
        st.write(self.data_handler.data.head())
    
    def display_visualization(self):
        st.subheader("2. Data Visualization")
        fig, ax = plt.subplots()
        sns.scatterplot(data=self.data_handler.data, x="Height", y="Weight", hue="NObeyesdad", palette="Set1")
        st.pyplot(fig)
    
    def user_input_features(self):
        st.subheader("3 & 4. Input Data Numerik dan Kategorikal")
        features = {}
        for col in self.data_handler.data.columns:
            if col != "NObeyesdad":  # Jangan tampilkan kolom target
                if col in self.data_handler.label_encoders:
                    features[col] = st.selectbox(f"{col}", self.data_handler.label_encoders[col].classes_)
                else:
                    features[col] = st.slider(f"{col}", float(self.data_handler.data[col].min()), float(self.data_handler.data[col].max()), float(self.data_handler.data[col].mean()))
        
        user_df = pd.DataFrame([features])
        user_df = pd.DataFrame(self.data_handler.transform_user_input(user_df), columns=user_df.columns)
        return user_df
    
    def display_prediction(self, user_input):
        st.subheader("6 & 7. Prediksi dan Probabilitas Klasifikasi")
        predicted_class, prediction_prob = self.model_handler.predict(user_input)
        prob_df = pd.DataFrame({'Class': self.model_handler.model.classes_, 'Probability': prediction_prob})
        st.write(prob_df)
        st.write(f"### Prediksi Akhir: {predicted_class}")
    
    def run(self):
        st.title("Aplikasi Klasifikasi Obesitas dengan Streamlit")
        self.display_raw_data()
        self.display_visualization()
        user_input = self.user_input_features()
        st.subheader("5. Data yang Diinputkan User")
        st.write(user_input)
        self.display_prediction(user_input)

# Main Program
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "ObesityDataSet_raw_and_data_sinthetic.csv")
MODEL_PATH = os.path.join(BASE_DIR, "trained_model.pkl")

# Pastikan file ada sebelum dijalankan
if not os.path.exists(DATA_PATH):
    st.error(f"File data tidak ditemukan: {DATA_PATH}")
elif not os.path.exists(MODEL_PATH):
    st.error(f"File model tidak ditemukan: {MODEL_PATH}")
else:
    data_handler = DataHandler(DATA_PATH)
    data_handler.load_data()
    data_handler.preprocess_data()

    model_handler = ModelHandler(MODEL_PATH)

    app = ObesityClassificationApp(data_handler, model_handler)
    app.run()

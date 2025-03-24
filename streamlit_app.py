import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Load Dataset & Model
DATA_PATH = "/mnt/data/ObesityDataSet_raw_and_data_sinthetic.csv"
MODEL_PATH = "/mnt/data/trained_model.pkl"

class DataHandler:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.label_encoders = {}
        self.scaler = MinMaxScaler()

    def load_data(self):
        """Memuat dataset dari file CSV"""
        self.data = pd.read_csv(self.file_path)

    def preprocess_data(self):
        """Melakukan encoding kolom kategorikal dan normalisasi kolom numerik"""
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            self.data[col] = le.fit_transform(self.data[col])
            self.label_encoders[col] = le  # Simpan encoder untuk nanti

        numerical_cols = self.data.select_dtypes(include=['int64', 'float64']).columns
        self.data[numerical_cols] = self.scaler.fit_transform(self.data[numerical_cols])

    def get_processed_data(self):
        """Mengembalikan data setelah preprocessing"""
        return self.data

class ModelHandler:
    def __init__(self, model_path):
        """Memuat model yang telah dilatih"""
        with open(model_path, "rb") as file:
            self.model = pickle.load(file)

    def predict(self, features):
        """Memprediksi input dari pengguna"""
        prediction_prob = self.model.predict_proba(features)[0]
        predicted_class = self.model.classes_[np.argmax(prediction_prob)]
        return predicted_class, prediction_prob

# Streamlit App
class ObesityClassificationApp:
    def __init__(self, data_handler, model_handler):
        self.data_handler = data_handler
        self.model_handler = model_handler
        self.user_input = None

    def display_raw_data(self):
        """1. Menampilkan raw data"""
        st.subheader("1. Raw Data")
        st.write(self.data_handler.data.head())

    def display_visualization(self):
        """2. Visualisasi Data"""
        st.subheader("2. Data Visualization")

        fig, ax = plt.subplots(1, 2, figsize=(12, 5))

        # Histogram usia
        sns.histplot(self.data_handler.data["Age"], kde=True, bins=20, ax=ax[0])
        ax[0].set_title("Distribusi Usia")

        # Distribusi kelas obesitas
        sns.countplot(x="NObeyesdad", data=self.data_handler.data, ax=ax[1])
        ax[1].set_title("Distribusi Kelas Obesitas")
        ax[1].set_xticklabels(self.data_handler.label_encoders["NObeyesdad"].classes_, rotation=45)

        st.pyplot(fig)

    def user_input_features(self):
        """3 & 4. Input Data Numerik dan Kategorikal"""
        st.subheader("3 & 4. Input Data")

        features = {}
        for col in self.data_handler.data.columns:
            if col == "NObeyesdad":
                continue
            elif col in self.data_handler.label_encoders:
                options = list(self.data_handler.label_encoders[col].classes_)
                features[col] = st.selectbox(f"{col}", options)
            else:
                min_val, max_val = float(self.data_handler.data[col].min()), float(self.data_handler.data[col].max())
                features[col] = st.slider(f"{col}", min_val, max_val, (min_val + max_val) / 2)

        # Konversi input ke DataFrame
        input_df = pd.DataFrame([features])

        # Encode categorical features
        for col in self.data_handler.label_encoders:
            input_df[col] = self.data_handler.label_encoders[col].transform(input_df[col])

        # Normalisasi data numerik
        numerical_cols = self.data_handler.data.select_dtypes(include=['int64', 'float64']).columns
        input_df[numerical_cols] = self.data_handler.scaler.transform(input_df[numerical_cols])

        return input_df

    def display_user_input(self):
        """5. Menampilkan data yang diinputkan user"""
        st.subheader("5. Data yang Dimasukkan")
        st.write(self.user_input)

    def display_prediction(self):
        """6 & 7. Menampilkan probabilitas klasifikasi dan hasil prediksi"""
        st.subheader("6 & 7. Hasil Prediksi")

        predicted_class, prediction_prob = self.model_handler.predict(self.user_input)
        prob_df = pd.DataFrame({
            "Class": self.model_handler.model.classes_,
            "Probability": prediction_prob
        })
        st.write("### Probabilitas Prediksi")
        st.write(prob_df)

        st.write(f"### Prediksi Akhir: {predicted_class}")

    def run(self):
        st.title("Aplikasi Klasifikasi Obesitas dengan Streamlit")
        self.display_raw_data()
        self.display_visualization()
        self.user_input = self.user_input_features()
        self.display_user_input()
        self.display_prediction()

# Load Data & Model
data_handler = DataHandler(DATA_PATH)
data_handler.load_data()
data_handler.preprocess_data()

model_handler = ModelHandler(MODEL_PATH)

# Jalankan Aplikasi
app = ObesityClassificationApp(data_handler, model_handler)
app.run()

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
        self.categorical_cols = []
        self.numerical_cols = []
    
    def load_data(self):
        self.data = pd.read_csv(self.file_path)
        self.categorical_cols = self.data.select_dtypes(include=['object']).columns.tolist()
        self.numerical_cols = self.data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        return self.data
    
    def preprocess_data(self):
        """ Menyiapkan encoder untuk kolom kategorikal dan scaler untuk numerik """
        for col in self.categorical_cols:
            if col != "NObeyesdad":  # Jangan ubah target
                self.label_encoders[col] = LabelEncoder()
                self.data[col] = self.label_encoders[col].fit_transform(self.data[col])
        
        self.data[self.numerical_cols] = self.scaler.fit_transform(self.data[self.numerical_cols])

    def transform_user_input(self, user_df):
        """ Hanya transformasi untuk keperluan prediksi """
        for col in self.label_encoders:
            user_df[col] = self.label_encoders[col].transform(user_df[col])
        user_df[self.numerical_cols] = self.scaler.transform(user_df[self.numerical_cols])
        return user_df

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
                if col in self.data_handler.categorical_cols:
                    features[col] = st.selectbox(f"{col}", self.data_handler.label_encoders[col].classes_)
                else:
                    min_val = self.data_handler.data[col].min()
                    max_val = self.data_handler.data[col].max()
                    mean_val = self.data_handler.data[col].mean()
                    features[col] = st.slider(f"{col}", float(min_val), float(max_val), float(mean_val))
        
        # Buat dataframe tanpa transformasi untuk ditampilkan ke user
        user_df_original = pd.DataFrame([features])

        # Transformasi hanya untuk keperluan prediksi
        user_df_transformed = self.data_handler.transform_user_input(user_df_original.copy())
        return user_df_original, user_df_transformed
    
    def display_prediction(self, user_input_original, user_input_transformed):
        st.subheader("6 & 7. Prediksi dan Probabilitas Klasifikasi")
        predicted_class, prediction_prob = self.model_handler.predict(user_input_transformed)
        prob_df = pd.DataFrame({'Class': self.model_handler.model.classes_, 'Probability': prediction_prob})
        st.write(prob_df)
        st.write(f"### Prediksi Akhir: {predicted_class}")
    
    def run(self):
        st.title("Aplikasi Klasifikasi Obesitas dengan Streamlit")
        self.display_raw_data()
        self.display_visualization()
        user_input_original, user_input_transformed = self.user_input_features()
        st.subheader("5. Data yang Diinputkan User (Nilai Asli)")
        st.write(user_input_original)
        self.display_prediction(user_input_original, user_input_transformed)

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

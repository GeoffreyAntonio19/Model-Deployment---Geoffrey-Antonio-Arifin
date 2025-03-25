import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import os
import sklearn

# Load Dataset
class DataHandler:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data_original = None
        self.data_processed = None
        self.label_encoders = {}
        self.scaler = MinMaxScaler()
        self.categorical_cols = []
        self.numerical_cols = []
    
    def load_data(self):
        self.data_original = pd.read_csv(self.file_path)
        self.categorical_cols = self.data_original.select_dtypes(include=['object']).columns.tolist()
        self.numerical_cols = self.data_original.select_dtypes(include=['int64', 'float64']).columns.tolist()
        return self.data_original
    
    def preprocess_data(self):
        self.data_processed = self.data_original.copy()
        for col in self.categorical_cols:
            if col != "NObeyesdad":
                self.label_encoders[col] = LabelEncoder()
                self.data_processed[col] = self.label_encoders[col].fit_transform(self.data_original[col])
        self.data_processed[self.numerical_cols] = self.scaler.fit_transform(self.data_original[self.numerical_cols])

    def transform_user_input(self, user_df):
        transformed_df = user_df.copy()
        for col in self.label_encoders:
            try:
                transformed_df[col] = self.label_encoders[col].transform([user_df[col].values[0]])[0]
            except ValueError:
                st.warning(f"Nilai '{user_df[col].values[0]}' di {col} tidak dikenal. Gunakan kategori lain.")
                return None
        transformed_df[self.numerical_cols] = self.scaler.transform(user_df[self.numerical_cols])
        return transformed_df

# Load Trained Model
class ModelHandler:
    def __init__(self, model_path):
        self.model = self.load_model(model_path)
    
    def load_model(self, model_path):
        try:
            return joblib.load(model_path)
        except Exception as e:
            st.error(f"Gagal memuat model: {e}")
            return None
    
    def predict(self, features):
        if features is None or self.model is None:
            return None, None
        try:
            features = features.fillna(0)
            features = features.to_numpy().reshape(1, -1)
            if hasattr(self.model, 'predict_proba'):
                prediction_prob = self.model.predict_proba(features)[0]
                predicted_class = self.model.classes_[np.argmax(prediction_prob)]
            else:
                prediction_prob = None
                predicted_class = self.model.predict(features)[0]
            return predicted_class, prediction_prob
        except AttributeError as e:
            st.error(f"Terjadi error saat melakukan prediksi. Pastikan model kompatibel dengan versi Scikit-learn terbaru: {e}")
            return None, None

# Streamlit App Class
class ObesityClassificationApp:
    def __init__(self, data_handler, model_handler):
        self.data_handler = data_handler
        self.model_handler = model_handler
    
    def display_raw_data(self):
        st.subheader("1. Menampilkan Raw Data")
        st.write(self.data_handler.data_original.head())
    
    def display_visualization(self):
        st.subheader("2. Data Visualization")
        fig, ax = plt.subplots()
        sns.scatterplot(data=self.data_handler.data_original, x="Height", y="Weight", hue="NObeyesdad", palette="Set1")
        st.pyplot(fig)
    
    def user_input_features(self):
        st.subheader("3 & 4. Input Data Numerik dan Kategorikal")
        features = {}
        for col in self.data_handler.data_original.columns:
            if col != "NObeyesdad":
                if col in self.data_handler.categorical_cols:
                    features[col] = st.selectbox(f"{col}", self.data_handler.data_original[col].unique())
                else:
                    min_val = self.data_handler.data_original[col].min()
                    max_val = self.data_handler.data_original[col].max()
                    mean_val = self.data_handler.data_original[col].mean()
                    features[col] = st.slider(f"{col}", float(min_val), float(max_val), float(mean_val))
        user_df_original = pd.DataFrame([features])
        user_df_transformed = self.data_handler.transform_user_input(user_df_original.copy())
        return user_df_original, user_df_transformed
    
    def display_prediction(self, user_input_original, user_input_transformed):
        st.subheader("6 & 7. Prediksi dan Probabilitas Klasifikasi")
        predicted_class, prediction_prob = self.model_handler.predict(user_input_transformed)
        if predicted_class is not None:
            st.write(f"### Prediksi Akhir: {predicted_class}")
            if prediction_prob is not None:
                prob_df = pd.DataFrame({'Class': self.model_handler.model.classes_, 'Probability': prediction_prob})
                prob_df = prob_df.sort_values(by="Probability", ascending=False)
                st.write(prob_df)
            else:
                st.info("Model tidak mendukung probabilitas klasifikasi.")
        else:
            st.error("Prediksi gagal. Pastikan input valid.")

    def run(self):
        st.title("Aplikasi Klasifikasi Obesitas dengan Streamlit")
        self.display_raw_data()
        self.display_visualization()
        user_input_original, user_input_transformed = self.user_input_features()
        st.subheader("5. Data yang Diinputkan User (Nilai Asli)")
        st.write(user_input_original)
        if user_input_transformed is not None:
            self.display_prediction(user_input_original, user_input_transformed)

# Main Program
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "ObesityDataSet_raw_and_data_sinthetic.csv")
MODEL_PATH = os.path.join(BASE_DIR, "trained_model.pkl")

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

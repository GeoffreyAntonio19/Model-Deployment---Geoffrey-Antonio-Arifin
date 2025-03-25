import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
import os
import sklearn

# Cek versi Scikit-learn
def check_sklearn_version():
    return sklearn.__version__

st.write(f"Versi Scikit-learn (Runtime): {check_sklearn_version()}")

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

# Latih ulang model dan simpan ulang dalam format yang kompatibel
def retrain_and_save_model(data_handler, model_path):
    X = data_handler.data_processed.drop(columns=["NObeyesdad"])
    y = data_handler.data_original["NObeyesdad"]
    
    model = DecisionTreeClassifier()
    model.fit(X, y)
    
    model._sklearn_version = check_sklearn_version()  # Simpan versi sklearn dalam model
    joblib.dump(model, model_path)
    st.sidebar.success("✅ Model telah dilatih ulang dan disimpan kembali!")

# Load Trained Model
class ModelHandler:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.load_model()
    
    def load_model(self):
        if not os.path.exists(self.model_path):
            st.sidebar.warning("⚠️ Model tidak ditemukan, melatih ulang...")
            retrain_and_save_model(data_handler, self.model_path)
        
        self.model = joblib.load(self.model_path)
        model_sklearn_version = getattr(self.model, '_sklearn_version', 'Unknown')
        st.sidebar.write(f"📌 Model dimuat dengan versi Scikit-learn: {model_sklearn_version}")
        
        # Cek kompatibilitas versi Scikit-learn
        if model_sklearn_version != check_sklearn_version():
            st.sidebar.warning("⚠️ Versi Scikit-learn berbeda! Melatih ulang model...")
            retrain_and_save_model(data_handler, self.model_path)
            self.model = joblib.load(self.model_path)
    
    def predict(self, features):
        if features is None or self.model is None:
            return None, None
        
        features = features.to_numpy().reshape(1, -1)
        predicted_class = self.model.predict(features)[0]
        return predicted_class

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
        height = st.slider("Height", float(self.data_handler.data_original["Height"].min()), float(self.data_handler.data_original["Height"].max()), float(self.data_handler.data_original["Height"].mean()))
        weight = st.slider("Weight", float(self.data_handler.data_original["Weight"].min()), float(self.data_handler.data_original["Weight"].max()), float(self.data_handler.data_original["Weight"].mean()))
        
        user_df = pd.DataFrame({"Height": [height], "Weight": [weight]})
        return user_df
    
    def display_prediction(self, user_input):
        st.subheader("6 & 7. Prediksi dan Probabilitas Klasifikasi")
        prediction = self.model_handler.predict(user_input)
        if prediction is not None:
            st.write(f"### Prediksi Akhir: {prediction}")
        else:
            st.error("Prediksi gagal. Pastikan input valid dan model kompatibel.")
    
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

data_handler = DataHandler(DATA_PATH)
data_handler.load_data()
data_handler.preprocess_data()
model_handler = ModelHandler(MODEL_PATH)
app = ObesityClassificationApp(data_handler, model_handler)
app.run()

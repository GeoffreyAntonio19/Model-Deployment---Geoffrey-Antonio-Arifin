import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
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
            model = joblib.load(model_path)
            model_sklearn_version = getattr(model, '_sklearn_version', 'Unknown')
            st.sidebar.write(f"Versi Scikit-learn (Model): {model_sklearn_version}")
            return model
        except Exception as e:
            st.error(f"Gagal memuat model: {e}")
            return None
    
    def predict(self, features):
        if features is None or self.model is None:
            return None, None
        try:
            features = features.fillna(0)
            features = features.to_numpy().reshape(1, -1)
            prediction = self.model.predict(features)
            prediction_proba = self.model.predict_proba(features)
            return prediction[0], prediction_proba[0]
        except AttributeError as e:
            st.error(f"Terjadi error saat melakukan prediksi: {e}. Pastikan model kompatibel dengan versi terbaru Scikit-learn.")
            return None, None

# Streamlit App Class
class ObesityClassificationApp:
    def __init__(self, data_handler, model_handler):
        self.data_handler = data_handler
        self.model_handler = model_handler
    
    def display_prediction(self, user_input_original, user_input_transformed):
        st.subheader("Prediksi dan Probabilitas Klasifikasi")
        predicted_class, prediction_prob = self.model_handler.predict(user_input_transformed)
        if predicted_class is not None:
            obesity_categories = np.array(['Normal_Weight', 'Overweight_Level_I', 'Overweight_Level_II',
                                           'Obesity_Type_I', 'Insufficient_Weight', 'Obesity_Type_II',
                                           'Obesity_Type_III'])
            st.success(f"Prediksi: {obesity_categories[predicted_class]}")
            prob_df = pd.DataFrame(prediction_prob.reshape(1, -1), columns=obesity_categories)
            st.dataframe(prob_df.style.format('{:.4f}'))
        else:
            st.error("Prediksi gagal. Pastikan input valid dan model kompatibel.")

    def run(self):
        st.title("Aplikasi Klasifikasi Obesitas dengan Streamlit")
        user_input = {
            "Gender": st.selectbox("Gender", ['Male', 'Female']),
            "Age": st.slider("Age", 10, 80, 25),
            "Height": st.slider("Height", 1.2, 2.2, 1.7),
            "Weight": st.slider("Weight", 30, 200, 70),
            "family_history_with_overweight": st.selectbox("Family History", ['yes', 'no'])
        }
        user_input_df = pd.DataFrame([user_input])
        user_input_transformed = self.data_handler.transform_user_input(user_input_df)
        if user_input_transformed is not None:
            self.display_prediction(user_input_df, user_input_transformed)

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

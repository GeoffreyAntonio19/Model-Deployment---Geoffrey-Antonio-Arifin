import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Load Dataset
class DataHandler:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.label_encoders = {}
        self.scaler = MinMaxScaler()
    
    def load_data(self):
        self.data = pd.read_csv(self.file_path)
    
    def preprocess_data(self):
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            self.label_encoders[col] = LabelEncoder()
            self.data[col] = self.label_encoders[col].fit_transform(self.data[col])
        
        numerical_cols = self.data.select_dtypes(include=['int64', 'float64']).columns
        self.data[numerical_cols] = self.scaler.fit_transform(self.data[numerical_cols])
    
    def get_features_and_target(self, target_column='NObeyesdad'):
        return self.data.drop(target_column, axis=1), self.data[target_column]

# Load Trained Model
class ModelHandler:
    def __init__(self, model_path):
        with open(model_path, 'rb') as file:
            self.model = pickle.load(file)
    
    def predict(self, features):
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
        sns.histplot(self.data_handler.data['Age'], kde=True, bins=20, ax=ax)
        st.pyplot(fig)
    
    def user_input_features(self):
        st.subheader("3 & 4. Input Data Numerik dan Kategorikal")
        features = {}
        for col in self.data_handler.data.columns:
            if col in self.data_handler.label_encoders:
                selected_option = st.selectbox(f"{col}", self.data_handler.label_encoders[col].classes_)
                features[col] = self.data_handler.label_encoders[col].transform([selected_option])[0]
            else:
                min_val = float(self.data_handler.data[col].min())
                max_val = float(self.data_handler.data[col].max())
                features[col] = st.slider(f"{col}", min_val, max_val, (min_val + max_val) / 2)
        
        return pd.DataFrame([features])
    
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
DATA_PATH = '/mnt/data/ObesityDataSet_raw_and_data_sinthetic.csv'
MODEL_PATH = '/mnt/data/trained_model.pkl'

data_handler = DataHandler(DATA_PATH)
data_handler.load_data()
data_handler.preprocess_data()

model_handler = ModelHandler(MODEL_PATH)

app = ObesityClassificationApp(data_handler, model_handler)
app.run()

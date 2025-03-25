import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

st.title('Obesity Classification Prediction')
st.info('This application predicts your obesity class based on some parameters!')

# Load dataset asli
@st.cache_data
def load_data():
    return pd.read_csv("ObesityDataSet_raw_and_data_sinthetic.csv")  # Ganti dengan nama file dataset Anda

df = load_data()

# Menampilkan dataset
with st.expander('**Data**'):
    st.write('This is the raw dataset')
    st.dataframe(df, use_container_width=True)

# Visualisasi Data
with st.expander('**Data Visualization**'):
    st.scatter_chart(data=df, x='Height', y='Weight', color='NObeyesdad')

st.subheader('**Input Features**')

# **User Input**
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
mtrans = st.selectbox("Main Mode of Transportation (MTRANS)", 
                      ("Public Transportation", "Walking", "Automobile", "Motorbike", "Bike"))

# Menyimpan input dalam DataFrame
user_data = pd.DataFrame({
    "Gender": [gender], "Age": [age], "Height": [height], "Weight": [weight],
    "Family_History": [family_history], "FAVC": [favc], "FCVC": [fcvc], "NCP": [ncp],
    "CAEC": [caec], "SMOKE": [smoke], "CH2O": [ch2o], "SCC": [scc],
    "FAF": [faf], "TUE": [tue], "CALC": [calc], "MTRANS": [mtrans]
})

# Menampilkan input user dalam bentuk tabel
st.subheader("User Input Data")
st.dataframe(user_data, use_container_width=True)

# **Preprocessing Data**
def preprocess_data(df):
    label_encoders = {}
    categorical_columns = ["Gender", "family_history_with_overweight", "FAVC", "CAEC", "SMOKE", "SCC", "CALC", "MTRANS"]
    
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    return df, label_encoders

df, label_encoders = preprocess_data(df)

# **Split Data**
X = df.drop(columns=["NObeyesdad"])  # Ganti dengan nama target kolom sesuai dataset
y = df["NObeyesdad"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# **Normalisasi Data**
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# **Train Model**
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# **Simpan model untuk pemakaian ulang**
joblib.dump(model, "model_obesity.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")

# **Load Model & Scaler**
model = joblib.load("model_obesity.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# **Preprocessing Input User**
for col in label_encoders:
    user_data[col] = label_encoders[col].transform(user_data[col])

# **Normalisasi Input**
user_data_scaled = scaler.transform(user_data)

# **Prediksi Probabilitas**
probabilities = model.predict_proba(user_data_scaled)[0]
categories = model.classes_

# **Menampilkan Probabilitas Tiap Kelas**
df_result = pd.DataFrame(probabilities.reshape(1, -1), columns=categories)
st.write("### Obesity Prediction Probability")
st.dataframe(df_result)

# **Menentukan Prediksi Akhir**
predicted_index = np.argmax(probabilities)
predicted_label = categories[predicted_index]

# **Tampilkan Hasil Prediksi**
st.write(f"**Predicted Obesity Class:** {predicted_label}")

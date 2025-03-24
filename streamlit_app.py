import streamlit as st
import joblib

def load_model(filename):
    model = joblib.load(filename)
    return model

def main():
    st.title('Obesity Classification')
    st.write('This app using Machine Learning')
    model_filename = 'trained_model.pkl'
    model = load_model(model_filename)

if __name__ == "__main__":
    main()

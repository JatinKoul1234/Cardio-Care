import streamlit as st
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Load the dataset
@st.cache_data
def load_data():
    data = pd.read_csv('framingham.csv')
    return data

# Handle missing values
def preprocess_data(df):
    df.fillna(df.mean(), inplace=True)
    df.fillna(df.mode().iloc[0], inplace=True)
    return df

# Load and train the model
@st.cache_resource
def load_model():
    categorical_features = ['male', 'currentSmoker', 'prevalentHyp', 'diabetes', 'prevalentStroke', 'BPMeds', 'education']
    numeric_features = ['age', 'cigsPerDay', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']

    preprocessor = make_column_transformer(
        (SimpleImputer(strategy='most_frequent'), categorical_features),
        (SimpleImputer(strategy='mean'), numeric_features),
        (OrdinalEncoder(), categorical_features),
        remainder='passthrough'
    )
    
    model = make_pipeline(
        preprocessor,
        DecisionTreeClassifier(random_state=42)
    )

    df = preprocess_data(load_data())
    x = df.drop(columns=["TenYearCHD"])
    y = df["TenYearCHD"]
    x_train, _, y_train, _ = train_test_split(x, y, test_size=0.2, random_state=42)
    model.fit(x_train, y_train)

    return model

# UI Inputs
def user_input_features():
    with st.expander("📝 Fill Patient Information"):
        gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
        age = st.slider("Age", 30, 80, 50)
        smoker = st.radio("Current Smoker?", ["Yes", "No"], horizontal=True)
        cigs_per_day = st.slider("Cigarettes per Day", 0, 50, 10)
        totChol = st.slider("Total Cholesterol", 100, 400, 200)
        sysBP = st.slider("Systolic Blood Pressure", 90, 200, 120)
        diaBP = st.slider("Diastolic Blood Pressure", 60, 130, 80)
        BMI = st.slider("Body Mass Index (BMI)", 15, 40, 25)
        heart_rate = st.slider("Heart Rate", 50, 120, 70)
        glucose = st.slider("Glucose Level", 50, 200, 100)
        prevalentHyp = st.radio("Hypertension?", ["Yes", "No"], horizontal=True)
        diabetes = st.radio("Diabetes?", ["Yes", "No"], horizontal=True)
        stroke = st.radio("History of Stroke?", ["Yes", "No"], horizontal=True)

    # Mapping to numerical
    data = {
        'male': 1 if gender == "Male" else 0,
        'age': age,
        'currentSmoker': 1 if smoker == "Yes" else 0,
        'cigsPerDay': cigs_per_day,
        'totChol': totChol,
        'sysBP': sysBP,
        'diaBP': diaBP,
        'BMI': BMI,
        'heartRate': heart_rate,
        'glucose': glucose,
        'prevalentHyp': 1 if prevalentHyp == "Yes" else 0,
        'diabetes': 1 if diabetes == "Yes" else 0,
        'prevalentStroke': 1 if stroke == "Yes" else 0,
        'BPMeds': 0,       # default
        'education': 1     # default
    }

    return pd.DataFrame(data, index=[0])

# Main App
def main():
    st.set_page_config(page_title="Cardio Care Risk Predictor", page_icon="💓", layout="centered")
    st.title("💓 Coronary Heart Disease Risk Predictor")
    st.markdown("This app predicts the **10-year risk** of developing Coronary Heart Disease using medical parameters.")

    st.divider()
    st.subheader("📋 Patient Medical Information")
    df = user_input_features()

    # Predict Button
    if st.button("🔍 Predict CHD Risk"):
        with st.spinner("Analyzing..."):
            model = load_model()
            prediction = model.predict(df)

        st.success("✅ Prediction Complete")
        st.subheader("📈 Prediction Result")
        if prediction[0] == 1:
            st.error("⚠️ High Risk of Coronary Heart Disease")
        else:
            st.success("🟢 Low Risk of Coronary Heart Disease")

if __name__ == "__main__":
    main()

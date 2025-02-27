import streamlit as st
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
@st.cache_data
def load_data():
    data = pd.read_csv('framingham.csv')
    return data

# Handle missing values
def preprocess_data(df):
    # Check for NaNs and fill them with appropriate values
    df.fillna(df.mean(), inplace=True)  # Fill numeric NaNs with the mean
    df.fillna(df.mode().iloc[0], inplace=True)  # Fill categorical NaNs with the mode
    return df

# Load and train the model
def load_model():
    # Define the preprocessing for both numeric and categorical columns
    categorical_features = ['male', 'currentSmoker', 'prevalentHyp', 'diabetes', 'prevalentStroke', 'BPMeds', 'education']
    numeric_features = ['age', 'cigsPerDay', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']

    # Preprocessing pipelines
    preprocessor = make_column_transformer(
        (SimpleImputer(strategy='most_frequent'), categorical_features),
        (SimpleImputer(strategy='mean'), numeric_features),
        (OrdinalEncoder(), categorical_features),
        remainder='passthrough'
    )
    
    # Create the pipeline
    model = make_pipeline(
        preprocessor,
        DecisionTreeClassifier(random_state=42)
    )
    
    # Load dataset and preprocess data
    df = preprocess_data(load_data())
    x = df.drop(columns=["TenYearCHD"])
    y = df["TenYearCHD"]
    
    # Train the model
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    model.fit(x_train, y_train)
    
    return model

# User input for the Streamlit interface
def user_input_features():
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.slider("Age", 30, 80, 50)
    smoker = st.selectbox("Are you a current smoker?", ["Yes", "No"])
    cigs_per_day = st.slider("Cigarettes per day", 0, 50, 10)
    totChol = st.slider("Total Cholesterol", 100, 400, 200)
    sysBP = st.slider("Systolic Blood Pressure", 90, 200, 120)
    diaBP = st.slider("Diastolic Blood Pressure", 60, 130, 80)
    BMI = st.slider("Body Mass Index (BMI)", 15, 40, 25)
    heart_rate = st.slider("Heart Rate", 50, 120, 70)
    glucose = st.slider("Glucose Level", 50, 200, 100)
    prevalentHyp = st.selectbox("Do you have hypertension?", ["Yes", "No"])
    diabetes = st.selectbox("Do you have diabetes?", ["Yes", "No"])
    stroke = st.selectbox("Have you had a stroke?", ["Yes", "No"])

    # Default values for missing features
    BPmeds = 0  # Assuming '0' means 'No' for 'BPMeds'
    education = 1  # A default educational level, adjust as necessary

    # Mapping categorical values to numerical values for prediction
    gender = 1 if gender == "Male" else 0
    smoker = 1 if smoker == "Yes" else 0
    prevalentHyp = 1 if prevalentHyp == "Yes" else 0
    diabetes = 1 if diabetes == "Yes" else 0
    stroke = 1 if stroke == "Yes" else 0

    # Create a dataframe for the inputs
    data = {
        'male': gender, 'age': age, 'currentSmoker': smoker, 'cigsPerDay': cigs_per_day,
        'totChol': totChol, 'sysBP': sysBP, 'diaBP': diaBP, 'BMI': BMI, 
        'heartRate': heart_rate, 'glucose': glucose, 'prevalentHyp': prevalentHyp,
        'diabetes': diabetes, 'prevalentStroke': stroke, 'BPMeds': BPmeds, 'education': education
    }

    features = pd.DataFrame(data, index=[0])
    return features

# Streamlit app definition
def main():
    st.title("Coronary Heart Disease Risk Prediction")

    st.sidebar.header("User Input Parameters")
    df = user_input_features()

    st.subheader("User Input Parameters")
    st.write(df)

    # Load the model and predict
    model = load_model()
    prediction = model.predict(df)

    st.subheader("Prediction")
    st.write(f"Risk of CHD: {'Yes' if prediction[0] == 1 else 'No'}")

if __name__ == "__main__":
    main()
# to run -> streamlit run app.py in the dir.
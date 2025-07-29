import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64
import os

# 🔽 CSV download link
def get_binary_file_downloader_html(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="PredictedHeart.csv">📥 Download Predictions CSV</a>'
    return href

# 🎯 Model names and display titles
algonames = ['Decision Trees', 'Logistic Regression', 'Random Forest', 'Support Vector Machine', 'Grid Random']
modelnames = ['DecisionTreeC.pkl', 'LogisticR.pkl', 'RandomForestC.pkl', 'SupportVM.pkl', 'gridrf.pkl']

# 🧠 Prediction engine
def predict_heart_disease(data):
    predictions = []
    for modelname in modelnames:
        try:
            with open(modelname, 'rb') as f:
                model = pickle.load(f)
            prediction = model.predict(data)
            predictions.append(prediction)
        except Exception as e:
            predictions.append([None])
            st.error(f"Error with model '{modelname}': {e}")
    return predictions

# 🧭 Main layout
st.title("💓 Heart Disease Predictor")
tab1, tab2, tab3 = st.tabs(['🧪 Predict', '📂 Bulk Predict', '📊 Model Information'])

# 🔸 Individual Prediction Tab
with tab1:
    age = st.number_input("Age (years)", min_value=0, max_value=150)
    sex = st.selectbox("Sex", ["Male", "Female"])
    chest_pain = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, max_value=300)
    cholesterol = st.number_input("Serum Cholesterol (mg/dl)", min_value=0)
    fasting_bs = st.selectbox("Fasting Blood Sugar", ["<=120 mg/dl", ">120 mg/dl"])
    resting_ecg = st.selectbox("Resting ECG Results", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
    max_hr = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=202)
    exercise_angina = st.selectbox("Exercise-Induced Angina", ["Yes", "No"])
    oldpeak = st.number_input("Oldpeak (ST Depression)", min_value=0.0, max_value=10.0)
    st_slope = st.selectbox("Slope of Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"])

    # Convert categorical to numeric
    sex = 0 if sex == "Male" else 1
    chest_pain = ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"].index(chest_pain)
    fasting_bs = 1 if fasting_bs == ">120 mg/dl" else 0
    resting_ecg = ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"].index(resting_ecg)
    exercise_angina = 1 if exercise_angina == "Yes" else 0
    st_slope = ["Upsloping", "Flat", "Downsloping"].index(st_slope)

    # Create input DataFrame
    input_data = pd.DataFrame({
        'Age': [age],
        'Sex': [sex],
        'ChestPainType': [chest_pain],
        'RestingBP': [resting_bp],
        'Cholesterol': [cholesterol],
        'FastingBS': [fasting_bs],
        'RestingECG': [resting_ecg],
        'MaxHR': [max_hr],
        'ExerciseAngina': [exercise_angina],
        'Oldpeak': [oldpeak],
        'ST_Slope': [st_slope]
    })

    if st.button("📊 Submit"):
        st.subheader("🔎 Prediction Results")
        st.markdown('---')
        results = predict_heart_disease(input_data)

        for i in range(len(results)):
            st.subheader(algonames[i])
            if results[i][0] is None:
                st.warning("⚠️ Prediction failed due to model error.")
            elif results[i][0] == 0:
                st.success("✅ No heart disease detected.")
            else:
                st.error("❗ Heart disease detected.")
            st.markdown('---')

# 🔹 Bulk Prediction Tab
with tab2:
    st.subheader("📁 Upload CSV File for Bulk Prediction")
    st.info("""
    ✅ Instructions before uploading:
    - No missing values (NaNs) allowed.
    - Exactly 11 columns, in this order:
        'Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol',
        'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope'
    - Feature values must be encoded as expected:
        - Sex: 0 = Male, 1 = Female
        - ChestPainType: 3 = Typical Angina, 0 = Atypical Angina, etc.
        - RestingECG: 0 = Normal, etc.
        - ST_Slope: 0 = Upsloping, 1 = Flat, 2 = Downsloping
    """)

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        input_data = pd.read_csv(uploaded_file)

        expected_columns = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol',
                            'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina',
                            'Oldpeak', 'ST_Slope']

        if all(col in input_data.columns for col in expected_columns):
            try:
                with open("LogisticR.pkl", "rb") as f:
                    model = pickle.load(f)
                input_data['Prediction LR'] = model.predict(input_data)
                st.subheader("🧾 Predictions:")
                st.dataframe(input_data)
                st.markdown(get_binary_file_downloader_html(input_data), unsafe_allow_html=True)
            except Exception as e:
                st.error(f"❌ Error loading model or predicting: {e}")
        else:
            st.warning("⚠️ Please make sure your CSV has the correct columns.")
    else:
        st.info("📤 Upload a properly formatted CSV file to begin.")

# 🔸 Model Info Tab
with tab3:
    import plotly.express as px

    data = {
        'Decision Trees': 80.97,
        'Logistic Regression': 85.86,
        'Random Forest': 84.23,
        'Support Vector Machine': 84.22,
        'GridRF': 89.75
    }

    df = pd.DataFrame(list(data.items()), columns=['Models', 'Accuracies'])
    fig = px.bar(df, x='Models', y='Accuracies', title='Model Accuracies Comparison')
    st.plotly_chart(fig)

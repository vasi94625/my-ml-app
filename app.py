import streamlit as st
import pandas as pd
import joblib

# Load the trained model and scaler
model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')

# Define the numerical and binary columns based on X_train
# These lists must match the columns used during training, in the correct order.
numerical_cols = ['Age', 'BMI', 'GenHlth', 'MentHlth', 'PhysHlth']
binary_cols = ['Sex', 'HighChol', 'CholCheck', 'Smoker', 'HeartDiseaseorAttack',
               'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 'DiffWalk',
               'Stroke', 'HighBP']

# Streamlit App Title
st.title('Diabetes Prediction App')
st.write('Enter patient details to predict the likelihood of Diabetes.')

# --- User Input Section ---
st.header('Patient Information')

# Collect inputs for each feature
# Numerical inputs
age = st.number_input('Age (years, 18-99)', min_value=18, max_value=99, value=50)
bmi = st.number_input('BMI', min_value=15.0, max_value=50.0, value=25.0, step=0.1)
gen_hlth = st.selectbox('General Health (1=Excellent, 2=Very Good, 3=Good, 4=Fair, 5=Poor)', options=[1, 2, 3, 4, 5], index=2) # Default to Good
ment_hlth = st.number_input('Mental Health (days of poor mental health in last 30 days)', min_value=0, max_value=30, value=0)
phys_hlth = st.number_input('Physical Health (days of poor physical health in last 30 days)', min_value=0, max_value=30, value=0)

# Binary inputs
sex = st.selectbox('Sex (0=Female, 1=Male)', options=[0, 1], index=0)
high_chol = st.selectbox('High Cholesterol (0=No, 1=Yes)', options=[0, 1], index=0)
chol_check = st.selectbox('Cholesterol Check in last 5 years (0=No, 1=Yes)', options=[0, 1], index=1)
smoker = st.selectbox('Smoker (0=No, 1=Yes)', options=[0, 1], index=0)
heart_disease_or_attack = st.selectbox('Heart Disease or Attack (0=No, 1=Yes)', options=[0, 1], index=0)
phys_activity = st.selectbox('Physical Activity in last 30 days (0=No, 1=Yes)', options=[0, 1], index=1)
fruits = st.selectbox('Consume Fruits daily (0=No, 1=Yes)', options=[0, 1], index=1)
veggies = st.selectbox('Consume Vegetables daily (0=No, 1=Yes)', options=[0, 1], index=1)
hvy_alcohol_consump = st.selectbox('Heavy Alcohol Consumption (0=No, 1=Yes)', options=[0, 1], index=0)
diff_walk = st.selectbox('Difficulty Walking or Climbing Stairs (0=No, 1=Yes)', options=[0, 1], index=0)
stroke = st.selectbox('Ever had a Stroke (0=No, 1=Yes)', options=[0, 1], index=0)
high_bp = st.selectbox('High Blood Pressure (0=No, 1=Yes)', options=[0, 1], index=0)

# Create a DataFrame from user inputs
input_data = pd.DataFrame([[age, sex, high_chol, chol_check, bmi, smoker, heart_disease_or_attack,
                            phys_activity, fruits, veggies, hvy_alcohol_consump, gen_hlth,
                            ment_hlth, phys_hlth, diff_walk, stroke, high_bp]],
                          columns=['Age', 'Sex', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'HeartDiseaseorAttack',
                                   'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 'GenHlth',
                                   'MentHlth', 'PhysHlth', 'DiffWalk', 'Stroke', 'HighBP'])


# Scale numerical features
input_data_scaled = input_data.copy()
input_data_scaled[numerical_cols] = scaler.transform(input_data_scaled[numerical_cols])

# Make prediction
if st.button('Predict Diabetes'):
    prediction = model.predict(input_data_scaled)

    if prediction[0] == 1:
        st.error('Prediction: \nHigh likelihood of Diabetes. Please consult a medical professional.')
    else:
        st.success('Prediction: \nLow likelihood of Diabetes. Keep up healthy habits!')


st.write("\n\nNote: This prediction is based on a model trained on a very small and unbalanced dataset and should not be used as medical advice.")

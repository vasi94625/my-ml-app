import streamlit as st
import pickle

st.title("My ML App")

model = pickle.load(open("model.pkl", "rb"))

x = st.number_input("Enter a value")

if st.button("Predict"):
    result = model.predict([[x]])
    st.write("Prediction:", result)

import streamlit as st
import pandas as pd
import joblib

model = joblib.load("models/customer_Churn_model.pkl")
feature_names=joblib.load("models/feature_names.pkl")

st.title("Customer Churn Prediction")
st.write("predict whether a customer is likely to churn based on customer details.")

st.write("Enter Customer Details")
gender=st.selectbox("Gender",["Male","Female"])
senior_citizen=st.selectbox("senior citizen",[0,1])
partner=st.selectbox("partner",["Yes","No"])
dependents=st.selectbox("Dependents",["Yes","No"])

phone_service=st.selectbox("phone service",["Yes","No"])
internet_service=st.selectbox("internet service",["DSL","Fiber optic","No"])
contract=st.selectbox("contract",["month-to-month","one year","two year"])
        
paperless_billing=st.selectbox("paperless billing",["Yes","No"])
payment_method=st.selectbox("payment method",["Electronic check","mailed check",
"bank transfer (automatic)","Credit card(automatic)"])

tenure = st.number_input("Tenure", min_value=0)
monthly_charges = st.number_input("Monthly Charges")
total_charges = st.number_input("Total Charges")

if st.button("Predict Churn"):

    input_data = pd.DataFrame(0, index=[0], columns=feature_names)

    input_data["tenure"] = tenure
    input_data["MonthlyCharges"] = monthly_charges
    input_data["TotalCharges"] = total_charges

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("Customer may churn")
        st.warning("Recommendation: review the customer's service details and retention strategy.")
    else:
        st.success("Prediction: Customer is likely to stay")
        st.balloons()
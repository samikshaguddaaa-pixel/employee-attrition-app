import os
import pickle
import streamlit as st
import pandas as pd
import numpy as np

# ---------------- LOAD MODEL ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = pickle.load(open(os.path.join(BASE_DIR, "attrition_model.pkl"), "rb"))
features = pickle.load(open(os.path.join(BASE_DIR, "features.pkl"), "rb"))

# ---------------- UI ----------------
st.set_page_config(page_title="Attrition Prediction", layout="centered")

st.title("🏢 Employee Attrition Prediction App")
st.write("Advanced HR Risk Analysis System")

# ---------------- INPUTS ----------------

age = st.slider("Age", 18, 60, 30)

gender = st.selectbox("Gender", ["Male", "Female"])
gender_val = 1 if gender == "Male" else 0

marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
marital_map = {"Single": 0, "Married": 1, "Divorced": 2}
marital_val = marital_map[marital_status]

education = st.selectbox("Education", ["Below College", "College", "Bachelor", "Master", "Doctor"])
edu_map = {"Below College": 0, "College": 1, "Bachelor": 2, "Master": 3, "Doctor": 4}
education_val = edu_map[education]

education_field = st.selectbox("Education Field", ["Life Sciences", "Medical", "Marketing", "Technical", "Other"])
edu_field_map = {"Life Sciences": 0, "Medical": 1, "Marketing": 2, "Technical": 3, "Other": 4}
education_field_val = edu_field_map[education_field]

job_role = st.selectbox("Job Role", ["Sales Executive", "Manager", "Developer", "HR", "Research Scientist"])
job_role_map = {"Sales Executive": 0, "Manager": 1, "Developer": 2, "HR": 3, "Research Scientist": 4}
job_role_val = job_role_map[job_role]

department = st.selectbox("Department", ["Sales", "R&D", "HR"])
dept_map = {"Sales": 0, "R&D": 1, "HR": 2}
department_val = dept_map[department]

job_level = st.slider("Job Level", 1, 5, 2)

years_at_company = st.slider("Years at Company", 0, 40, 3)
years_in_role = st.slider("Years in Current Role", 0, 20, 2)
years_since_promo = st.slider("Years Since Promotion", 0, 15, 1)

monthly_income = st.number_input("Monthly Income", 1000, 20000, 5000)
salary_hike = st.slider("Percent Salary Hike", 0, 30, 12)

stock_option = st.slider("Stock Option Level", 0, 3, 1)

overtime = st.selectbox("OverTime", ["Yes", "No"])
overtime_val = 1 if overtime == "Yes" else 0

distance_home = st.slider("Distance From Home", 1, 30, 5)

work_life_balance = st.slider("Work Life Balance", 1, 4, 3)

# ---------------- BUILD INPUT FRAME ----------------
input_data = pd.DataFrame(np.zeros((1, len(features))), columns=features)

def set_if_exists(col, val):
    if col in input_data.columns:
        input_data[col] = val

# ---------------- MAP FEATURES ----------------
set_if_exists("Age", age)
set_if_exists("Gender", gender_val)
set_if_exists("MaritalStatus", marital_val)
set_if_exists("Education", education_val)
set_if_exists("EducationField", education_field_val)
set_if_exists("JobRole", job_role_val)
set_if_exists("Department", department_val)
set_if_exists("JobLevel", job_level)
set_if_exists("YearsAtCompany", years_at_company)
set_if_exists("YearsInCurrentRole", years_in_role)
set_if_exists("YearsSinceLastPromotion", years_since_promo)
set_if_exists("MonthlyIncome", monthly_income)
set_if_exists("PercentSalaryHike", salary_hike)
set_if_exists("StockOptionLevel", stock_option)
set_if_exists("OverTime", overtime_val)
set_if_exists("DistanceFromHome", distance_home)
set_if_exists("WorkLifeBalance", work_life_balance)

# ---------------- PREDICTION ----------------
if st.button("Predict Attrition"):
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠️ High Risk: Employee may leave the company")
    else:
        st.success("✅ Low Risk: Employee will stay")

    st.write("Prediction:", prediction[0])

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Page Config
st.set_page_config(page_title="Loan Prediction App", layout="wide")

# Load data for EDA
@st.cache_data
def load_data():
    df = pd.read_csv('loan_prediction.csv')
    return df

df = load_data()

# Load model and encoders (assuming they are in the same folder)
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home & EDA", "Loan Prediction"])

# --- Home & EDA ---
if page == "Home & EDA":
    st.title("🏦 Loan Prediction Analysis")
    st.write("This application predicts loan approval status and provides insights into the dataset.")
    
    st.subheader("Data Overview")
    st.dataframe(df.head())
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Loan Status Distribution")
        fig, ax = plt.subplots()
        sns.countplot(data=df, x='Loan_Status', ax=ax, palette='viridis')
        st.pyplot(fig)

    with col2:
        st.subheader("Gender vs Loan Status")
        fig, ax = plt.subplots()
        sns.countplot(data=df, x='Gender', hue='Loan_Status', ax=ax, palette='Set2')
        st.pyplot(fig)

    st.subheader("Financial Distributions")
    col3, col4 = st.columns(2)
    with col3:
        st.write("Applicant Income Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df['ApplicantIncome'], kde=True, ax=ax, color='blue')
        st.pyplot(fig)
    with col4:
        st.write("Loan Amount Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df['LoanAmount'].dropna(), kde=True, ax=ax, color='green')
        st.pyplot(fig)

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    # Select numeric columns for correlation
    numeric_df = df.select_dtypes(include=[np.number])
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# --- Loan Prediction ---
elif page == "Loan Prediction":
    st.title("💳 Loan Eligibility Prediction")
    st.write("Enter the following details to check loan eligibility:")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            married = st.selectbox("Married", ["No", "Yes"])
            dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
            education = st.selectbox("Education", ["Graduate", "Not Graduate"])
            self_employed = st.selectbox("Self Employed", ["No", "Yes"])
            
        with col2:
            applicant_income = st.number_input("Applicant Income", min_value=0, value=5000)
            coapplicant_income = st.number_input("Coapplicant Income", min_value=0, value=0)
            loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0, value=150)
            loan_term = st.selectbox("Loan Amount Term", [360, 180, 120, 84, 60, 36])
            credit_history = st.selectbox("Credit History", [1.0, 0.0])
            property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
            
        submit = st.form_submit_button("Predict Status")
        
        if submit:
            # Preprocessing Input
            # 1. Gender: Female=0, Male=1 (Alphabetical)
            gender_val = 1 if gender == "Male" else 0
            # 2. Married: No=0, Yes=1
            married_val = 1 if married == "Yes" else 0
            # 3. Dependents: Replace 3+ with 3
            dep_val = 3 if dependents == "3+" else int(dependents)
            # 4. Education: Graduate=0, Not Graduate=1
            edu_val = 0 if education == "Graduate" else 1
            # 5. Self_Employed: No=0, Yes=1
            se_val = 1 if self_employed == "Yes" else 0
            # 6. Property_Area: Rural=0, Semiurban=1, Urban=2
            prop_mapping = {"Rural": 0, "Semiurban": 1, "Urban": 2}
            prop_val = prop_mapping[property_area]
            
            # Combine all features into an array
            input_data = np.array([[gender_val, married_val, dep_val, edu_val, se_val, 
                                    applicant_income, coapplicant_income, loan_amount, 
                                    loan_term, credit_history, prop_val]])
            
            # Predict
            prediction = model.predict(input_data)
            
            st.divider()
            if prediction[0] == 1:
                st.success("✅ Congratulations! Your loan is likely to be **Approved**.")
            else:
                st.error("❌ Sorry, your loan is likely to be **Rejected**.")
                
st.sidebar.markdown("---")
st.sidebar.info("Built for Loan Prediction Analysis")

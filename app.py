import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("loan_default_pipe.pkl")

st.set_page_config(page_title="Loan Default Predictor", layout="centered")
st.title("💳 Loan Default Prediction App")
st.markdown("Enter applicant's details:")

with st.form("loan_form"):
    col1, col2 = st.columns(2)

    with col1:
        Age = st.number_input("Age", min_value=18, max_value=100, value=35)
        Income = st.number_input("Annual Income", min_value=1000, value=50000)
        LoanAmount = st.number_input("Loan Amount", min_value=1000, value=10000)
        CreditScore = st.slider("Credit Score", min_value=300, max_value=850, value=700)
        InterestRate = st.slider("Interest Rate (%)", min_value=0.0, max_value=100.0, value=8.5)
        LoanTerm = st.selectbox("Loan Term (months)", [12, 24, 36, 48, 60], index=2)

    with col2:
        MonthsEmployed = st.number_input("Months Employed", min_value=0, value=24)
        NumCreditLines = st.number_input("Number of Credit Lines", min_value=0, value=4)
        DTIRatio = st.number_input("Debt-to-Income Ratio", min_value=0.0, max_value=1.0, value=0.3)
        Education = st.selectbox("Education Level", ["High School", "Bachelor's", "Graduate", "Post-Graduate", "Other"])
        EmploymentType = st.selectbox("Employment Type", ["Full-time", "Part-time", "Salaried", "Self-Employed", "Unemployed", "Retired"])
        MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])
        HasMortgage = st.selectbox("Has Mortgage?", ["Yes", "No"])
        HasDependents = st.selectbox("Has Dependents?", ["Yes", "No"])
        LoanPurpose = st.selectbox("Loan Purpose", ["Personal", "Debt Consolidation", "Car", "Education", "Medical", "Vacation", "Other"])
        HasCoSigner = st.selectbox("Has Co-Signer?", ["Yes", "No"])

    submitted = st.form_submit_button("🔍 Predict")

if submitted:
    input_data = pd.DataFrame([{
        'Age': Age,
        'Income': Income,
        'LoanAmount': LoanAmount,
        'CreditScore': CreditScore,
        'MonthsEmployed': MonthsEmployed,
        'NumCreditLines': NumCreditLines,
        'InterestRate': InterestRate,
        'LoanTerm': LoanTerm,
        'DTIRatio': DTIRatio,
        'Education': Education,
        'EmploymentType': EmploymentType,
        'MaritalStatus': MaritalStatus,
        'HasMortgage': HasMortgage,
        'HasDependents': HasDependents,
        'LoanPurpose': LoanPurpose,
        'HasCoSigner': HasCoSigner
    }])

    try:
        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0][1]

        st.subheader("📊 Prediction Result")
        if prediction == 1:
            st.error(f"❌ Loan is likely to be **rejected**.\n\n🔢 Confidence: {proba:.2%}")
        else:
            st.success(f"✅ Loan is likely to be **approved**.\n\n🔢 Confidence: {1 - proba:.2%}")


        with st.expander("View Input Data"):
            st.dataframe(input_data)

    except Exception as e:
        st.error(f"Prediction failed: {e}")

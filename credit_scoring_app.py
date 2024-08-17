import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

# Sample trained logistic regression model
feature_names = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
                 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
                 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4',
                 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3',
                 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']

# Creating a dummy logistic regression model
X_train = pd.DataFrame(np.random.rand(100, len(feature_names)), columns=feature_names)
y_train = np.random.randint(2, size=100)

# Create a pipeline that scales the data then applies logistic regression
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
model.fit(X_train, y_train)

# Streamlit UI
st.title("Credit Scoring Explainable AI Prototype")

# User Inputs with Explanations
st.header("Input Your Financial Data")

LIMIT_BAL = st.number_input("Credit Limit Balance", min_value=0, help="Total amount of credit available to you.")
SEX = st.selectbox("Sex", options=[1, 2], help="1 = Male, 2 = Female")
EDUCATION = st.selectbox("Education Level", options=[1, 2, 3, 4], help="1 = Graduate school, 2 = University, 3 = High school, 4 = Others")
MARRIAGE = st.selectbox("Marital Status", options=[1, 2, 3], help="1 = Married, 2 = Single, 3 = Others")
AGE = st.number_input("Age", min_value=18, max_value=100, help="Your current age.")
PAY_0 = st.number_input("Most Recent Payment Status", min_value=-2, max_value=8, help="-1 = Payment on time, 1 = 1 month late, 2 = 2 months late, etc.")
PAY_2 = st.number_input("Payment Status in 2nd Last Month", min_value=-2, max_value=8, help="Payment status for the second last month.")
PAY_3 = st.number_input("Payment Status in 3rd Last Month", min_value=-2, max_value=8, help="Payment status for the third last month.")

# Add more inputs as needed...

# Button to Trigger Prediction
if st.button("Get Prediction"):
    # Create a data frame with the input values
    user_data = pd.DataFrame({
        'LIMIT_BAL': [LIMIT_BAL],
        'SEX': [SEX],
        'EDUCATION': [EDUCATION],
        'MARRIAGE': [MARRIAGE],
        'AGE': [AGE],
        'PAY_0': [PAY_0],
        'PAY_2': [PAY_2],
        'PAY_3': [PAY_3],
        'PAY_4': [0],  # Default to 0 or allow user input
        'PAY_5': [0],  # Default to 0 or allow user input
        'PAY_6': [0],  # Default to 0 or allow user input
        'BILL_AMT1': [0],  # Default to 0 or allow user input
        'BILL_AMT2': [0],  # Default to 0 or allow user input
        'BILL_AMT3': [0],  # Default to 0 or allow user input
        'BILL_AMT4': [0],  # Default to 0 or allow user input
        'BILL_AMT5': [0],  # Default to 0 or allow user input
        'BILL_AMT6': [0],  # Default to 0 or allow user input
        'PAY_AMT1': [0],  # Default to 0 or allow user input
        'PAY_AMT2': [0],  # Default to 0 or allow user input
        'PAY_AMT3': [0],  # Default to 0 or allow user input
        'PAY_AMT4': [0],  # Default to 0 or allow user input
        'PAY_AMT5': [0],  # Default to 0 or allow user input
        'PAY_AMT6': [0]  # Default to 0 or allow user input
    })

    # Predict probability of default
    prob_default = model.predict_proba(user_data)[0][1]

    st.subheader(f"Predicted Probability of Default: {prob_default:.2f}")

    # Explanation
    logistic_model = model.named_steps['logisticregression']
    importance = logistic_model.coef_[0]

    # Sorting the features by importance
    indices = np.argsort(importance)[::-1]
    sorted_feature_names = np.array(feature_names)[indices]
    sorted_importance = importance[indices]

    # Plotting feature importance
    fig, ax = plt.subplots()
    ax.barh(sorted_feature_names, sorted_importance, color='skyblue')
    ax.set_xlabel('Feature Importance')
    ax.set_title('Feature Importance from Logistic Regression Model')
    ax.invert_yaxis()
    st.pyplot(fig)

    st.write("""
    ### Interpretation:
    - The higher the bar for a feature, the more impact it has on your credit score.
    - Features like recent payment status (PAY_0) have the most significant effect on your default risk.
    """)

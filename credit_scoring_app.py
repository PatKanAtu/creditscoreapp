import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

def create_model():
    feature_names = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
                     'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
                     'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4',
                     'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3',
                     'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']

    # Consider using more realistic training data instead of random
    X_train = pd.DataFrame(np.random.rand(100, len(feature_names)), columns=feature_names)
    y_train = np.random.randint(2, size=100)

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    model.fit(X_train, y_train)
    
    # Debugging: Check model coefficients
    st.write("Logistic Regression Coefficients:")
    st.write(model.named_steps['logisticregression'].coef_)
    
    return model, feature_names

def get_user_input(feature_names):
    user_input = {}
    user_input['LIMIT_BAL'] = st.number_input("Credit Limit Balance", min_value=0, help="Total amount of credit available to you.")
    user_input['SEX'] = st.selectbox("Sex", options=[1, 2], help="1 = Male, 2 = Female")
    user_input['EDUCATION'] = st.selectbox("Education Level", options=[1, 2, 3, 4], help="1 = Graduate school, 2 = University, 3 = High school, 4 = Others")
    user_input['MARRIAGE'] = st.selectbox("Marital Status", options=[1, 2, 3], help="1 = Married, 2 = Single, 3 = Others")
    user_input['AGE'] = st.number_input("Age", min_value=18, max_value=100, help="Your current age.")
    
    # Payment status inputs
    for i in range(7):
        user_input[f'PAY_{i}'] = st.number_input(f"Payment Status in {i}th Last Month", min_value=-2, max_value=8, help=f"Payment status for the {i}th last month.")
    
    # Bill amounts and payment amounts
    for i in range(1, 7):
        user_input[f'BILL_AMT{i}'] = st.number_input(f"Bill Amount for Month {i}", min_value=0, help=f"Bill amount for the {i}th last month.")
        user_input[f'PAY_AMT{i}'] = st.number_input(f"Payment Amount for Month {i}", min_value=0, help=f"Payment amount for the {i}th last month.")
    
    user_data = pd.DataFrame(user_input, index=[0])
    
    return user_data

def ensure_valid_input(user_data, feature_names):
    for feature in feature_names:
        if feature not in user_data.columns:
            user_data[feature] = 0  # Fill missing features with default value 0
    user_data = user_data[feature_names]  # Ensure the correct order of features
    return user_data

def display_feature_importance(logistic_model, feature_names):
    importance = logistic_model.coef_[0]
    indices = np.argsort(importance)[::-1]
    sorted_feature_names = np.array(feature_names)[indices]
    sorted_importance = importance[indices]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(sorted_feature_names, sorted_importance, color='skyblue')
    ax.set_xlabel('Feature Importance')
    ax.set_title('Feature Importance from Logistic Regression Model')
    ax.invert_yaxis()
    st.pyplot(fig)

    st.write("""
    ### Interpretation:
    - The higher the bar for a feature, the more impact it has on your credit score.
    - Features like recent payment status (PAY_0) have the most significant effect on your default risk.
    - Understanding these impacts can help you manage your credit better.
    """)

def main():
    st.title("Credit Scoring Explainable AI Prototype")
    st.header("Input Your Financial Data")

    model, feature_names = create_model()
    
    user_data = get_user_input(feature_names)
    
    user_data = ensure_valid_input(user_data, feature_names)  # Ensure all required features are present

    if st.button("Get Prediction"):
        prob_default = model.predict_proba(user_data)[0][1]
        st.subheader(f"Predicted Probability of Default: {prob_default:.2f}")
        
        logistic_model = model.named_steps['logisticregression']
        display_feature_importance(logistic_model, feature_names)

if __name__ == "__main__":
    main()

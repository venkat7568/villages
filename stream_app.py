import streamlit as st
import joblib
import pandas as pd

# Load the model
model = joblib.load('Linear_regression2.joblib')

# Define the Streamlit app
def main():
    st.title('Linear Regression Prediction')

    # Using columns to better organize the inputs
    st.write("## Input Data\nPlease enter the following details:")
    col1, col2 = st.columns(2)
    
    with col1:
        NUMBER_OF_SHGS_FEDERATED_INTO_VILLAGE_ORGANISATIONS_VOS = st.number_input('Number of SHGs Federated into Village Organisations (VOS)', min_value=0)
        NUMBER_OF_SHGS_WHICH_ACCESSED_BANK_LOANS = st.number_input('Number of SHGs Which Accessed Bank Loans', min_value=0)
        NUMBER_OF_BENEFICIARIES_RECEIVING_BENEFITS_UNDER_AAYUSHMAN_BHARAT = st.number_input('Number of Beneficiaries Under Aayushman Bharat-PMJAY or State Health Scheme', min_value=0)
    
    with col2:
        TOTAL_NUMBER_OF_HOUSEHOLDS_RECEIVING_FOOD_GRAINS = st.number_input('Total Households Receiving Food Grains from Fair Price Shops', min_value=0)
        TOTAL_NUMBER_OF_FARMERS = st.number_input('Total Number of Farmers', min_value=0)
        TOTAL_EXPENDITURE_NRM = st.number_input('Total Expenditure Approved Under NRM (2018-19)', min_value=0.0, format='%.2f')

    # Button to make predictions
    if st.button('Predict'):
        data = {
            'NUMBER OF SHGS FEDERATED INTO VILLAGE ORGANISATIONS (VOS)': [NUMBER_OF_SHGS_FEDERATED_INTO_VILLAGE_ORGANISATIONS_VOS],
            'NUMBER OF SHGS WHICH ACCESSED BANK LOANS': [NUMBER_OF_SHGS_WHICH_ACCESSED_BANK_LOANS],
            'NUMBER OF BENEFICIARIES RECEIVING BENEFITS UNDER AAYUSHMAN BHARAT': [NUMBER_OF_BENEFICIARIES_RECEIVING_BENEFITS_UNDER_AAYUSHMAN_BHARAT],
            'TOTAL NUMBER OF HOUSEHOLDS RECEIVING FOOD GRAINS': [TOTAL_NUMBER_OF_HOUSEHOLDS_RECEIVING_FOOD_GRAINS],
            'TOTAL NUMBER OF FARMERS': [TOTAL_NUMBER_OF_FARMERS],
            'TOTAL EXPENDITURE APPROVED UNDER NRM': [TOTAL_EXPENDITURE_NRM]
        }
        df = pd.DataFrame(data)
        prediction = model.predict(df)
        st.success(f"Predicted Budget: {prediction[0]:,.2f}")

if __name__ == '__main__':
    main()

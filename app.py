from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
app = Flask(__name__)

# Load the model
# model = joblib.load('Linear_regression.joblib')
model = joblib.load('Linear_regression2.joblib')
# Define the API endpoint for predictions
@app.route('/Linear_predict', methods=['POST'])
def Linear_predict():
    # Get data from the request
    data = request.json
    
    # Convert data to DataFrame
    df = pd.DataFrame(data, index=[0])
    
    # Ensure the columns are in the correct order
    df = df[['NUMBER OF SHGS FEDERATED INTO VILLAGE ORGANISATIONS (VOS)', 
             'NUMBER OF SHGS WHICH ACCESSED BANK LOANS', 
             'NUMBER OF BENEFICIARIES RECEIVING BENEFITS UNDER AAYUSHMAN BHARAT-PRADHAN MANTRI JAN AROGYA YOJANA OR ANY STATE GOVT HEALTH SCHEME', 
             'TOTAL NUMBER OF HOUSEHOLDS RECEIVING FOOD GRAINS FROM FAIR PRICE SHOPS ', 
             'TOTAL NUMBER OF FARMERS ', 
             'TOTAL EXPENDITURE APPROVED UNDER NRM IN THE LABOUR BUDGET FOR THE YEAR 2018-19)',
             ]]
    
    # Make prediction
    prediction = model.predict(df)
    
    # Return the prediction
    return jsonify({'prediction': prediction.tolist()})


@app.route('/')
def Linear_regression():
    return render_template('Linear_regression.html')


if __name__ == '__main__':
    app.run(debug=True)

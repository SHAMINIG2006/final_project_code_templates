import pickle
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the trained model (adjust path if needed)
with open("flask/payments.pkl", "rb") as f:
    model = pickle.load(f)

# Route for home page
@app.route('/')
def home():
    return render_template('home.html')

# Route for prediction form
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            step = float(request.form['step'])
            type_ = request.form['type']
            amount = float(request.form['amount'])
            oldbalanceOrg = float(request.form['oldbalanceOrg'])
            newbalanceOrig = float(request.form['newbalanceOrig'])
            oldbalanceDest = float(request.form['oldbalanceDest'])
            newbalanceDest = float(request.form['newbalanceDest'])
            isFlaggedFraud = int(request.form['isFlaggedFraud'])

            # Manual label encoding for 'type' feature
            type_mapping = {
                'CASH_OUT': 0,
                'TRANSFER': 1,
                'PAYMENT': 2,
                'DEBIT': 3,
                'CASH_IN': 4
            }
            type_encoded = type_mapping.get(type_, -1)

            features = np.array([[step, type_encoded, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest, isFlaggedFraud]])
            prediction = model.predict(features)

            result = "Fraud" if prediction[0] == 1 else "Not Fraud"
            return render_template('result.html', prediction=result)
        except Exception as e:
            return f"Error: {str(e)}"

# Route to thank you page
@app.route('/exit')
def exit_page():
    return render_template('exit.html')

if __name__ == '__main__':
    app.run(debug=True)

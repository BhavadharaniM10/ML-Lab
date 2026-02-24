from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model and encoders
model = pickle.load(open("loan_model.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = [
            request.form['Gender'],
            request.form['Married'],
            request.form['Dependents'],
            request.form['Education'],
            request.form['Self_Employed'],
            float(request.form['ApplicantIncome']),
            float(request.form['CoapplicantIncome']),
            float(request.form['LoanAmount']),
            float(request.form['Loan_Amount_Term']),
            request.form['Credit_History'],
            request.form['Property_Area']
        ]

        # Encode categorical values
        feature_names = [
            'Gender', 'Married', 'Dependents', 'Education',
            'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome',
            'LoanAmount', 'Loan_Amount_Term',
            'Credit_History', 'Property_Area'
        ]

        for i in range(len(data)):
            if feature_names[i] in encoders:
                data[i] = encoders[feature_names[i]].transform([data[i]])[0]

        final_features = [np.array(data)]
        prediction = model.predict(final_features)

        result = "Loan Approved ✅" if prediction[0] == 1 else "Loan Rejected ❌"

        return render_template("index.html", prediction_text=result)

    except Exception as e:
        return render_template("index.html", prediction_text="Error: " + str(e))

if __name__ == "__main__":
    app.run(debug=True)

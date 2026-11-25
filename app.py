from flask import Flask, render_template, request
import pandas as pd
import joblib


app = Flask(__name__)


# Load the trained model
model = joblib.load('house_price_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=["POST"])
def predict():

    area = float(request.form['area'])
    bedrooms = int(request.form['bedrooms'])
    bathrooms = int(request.form['bathrooms'])
    age = int(request.form['age'])

    input_data = pd.DataFrame({
        'area': [area],
        'bedrooms': [bedrooms],
        'bathrooms': [bathrooms],
        'age': [age]
    })

    # Make prediction using the loaded model
    prediction = model.predict(input_data)[0]

    # Render the result on the web page
    return render_template(
        'result.html', 
        prediction=f"{prediction:,.2f}")

if __name__ == '__main__':
    app.run(debug=True)


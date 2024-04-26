from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the form
    age = int(request.form['age'])
    estimated_salary = int(request.form['estimated_salary'])
    
    # Make prediction
    prediction = model.predict([[age, estimated_salary]])
    result = "will purchase the product" if prediction[0] == 1 else "will not purchase the product"
    
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)


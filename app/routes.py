from app import app
from flask import request
from app.controllers import genderClassification


@app.route('/predict', methods=['POST'])
def predict():
    return genderClassification.result()

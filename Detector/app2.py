from flask import Flask, render_template, request
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
import os
from Model import ObjectDetector
from Model import ObjectDetector
from model2 import Predictor

app = Flask(__name__)


@app.route('/')
def index():
    predictor = Predictor()
    predictor.predict_images()
    path = '/Users/akhilareddy/License-Plate-Geolocator copy/Detector/results/images_predictions.txt'
    predictions=predictor.read_predictions_from_file(path)
    print(predictions)
    return render_template('results2.html', predictions=predictions)

if __name__ == '__main__':
    app.run(debug=True)


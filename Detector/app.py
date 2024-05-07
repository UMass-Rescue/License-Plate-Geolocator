from flask import Flask, render_template, request
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
import os
from Model import ObjectDetector
from Model import ObjectDetector
from model2 import Predictor

app = Flask(__name__)
app.config['SECRET_KEY'] = 'super_secretive_key'
app.config['UPLOAD_FOLDER'] = 'static/files'

class UploadFileForm(FlaskForm):
    file = FileField("File")
    submit = SubmitField("Get Plates")

@app.route('/', methods=["GET", "POST"])
@app.route('/home', methods=["GET", "POST"])
def home():
    form = UploadFileForm()
    if form.validate_on_submit():
        file = request.files['file']  # Retrieve file from form submission
        if file:  # Check if file exists
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            detector = ObjectDetector()
            res = detector.detect_objects()
            if res:
                predictor = Predictor()
                predictor.predict_images()
                path = '/Users/akhilareddy/License-Plate-Geolocator copy/Detector/results/images_predictions.txt'
                predictions=predictor.read_predictions_from_file(path)
                print(predictions)
                return render_template('results2.html', predictions=predictions)
                
    return render_template('upload.html', form=form)

if __name__ == '__main__':
    app.run(debug=True, port=5000)

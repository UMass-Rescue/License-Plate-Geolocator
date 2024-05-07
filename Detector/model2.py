from fastai.vision import load_learner, open_image
from pathlib import Path
from torch import topk
import os
import warnings
from typing import List

warnings.filterwarnings("ignore", category=UserWarning)

class Predictor:
    def __init__(self):
        self.model_path = '/Users/akhilareddy/License-Plate-Geolocator copy/Detector/models'
        self.images_path = '/Users/akhilareddy/License-Plate-Geolocator copy/Detector/detections/images'
        self.output_dir = '/Users/akhilareddy/License-Plate-Geolocator copy/Detector/results'

    def predict_images(self):
        model = load_learner(self.model_path, device='cpu')
        output_file = os.path.split(self.images_path)[-1] + '_predictions.txt'
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        with open(os.path.join(self.output_dir, output_file), "w") as f:
            for img in Path(self.images_path).glob('*'):

                # Check if the file is an image
                if img.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']:
                    pred = model.predict(open_image(img))
                    _, top5_states = topk(pred[2], k=5)
                    f.write('Predictions for: ' + str(os.path.split(img)[-1]) + ' ' + str([model.data.classes[i] for i in list(top5_states)]) + '\n')
                    prediction =  [model.data.classes[i] for i in list(top5_states)]



    def read_predictions_from_file(self, file_path):
        predictions = []
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if line.startswith('Predictions for'):
                    parts = line.split(':')
                    if len(parts) >= 2:
                        file_name = parts[1].strip().split()[0]
                        preds = parts[1].split('[')[1].split(']')[0].strip().split(', ')
                        predictions.append({'file_name': file_name, 'predictions': preds})
        return predictions

if __name__ == '__main__':
    predictor = Predictor()
    predictor.predict_images()
    path = '/Users/akhilareddy/License-Plate-Geolocator copy/Detector/results/images_predictions.txt'
    print(predictor.read_predictions_from_file(path))




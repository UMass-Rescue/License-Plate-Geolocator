from absl import app, flags
from absl.flags import FLAGS
from fastai.callbacks import *
from fastai.vision import *
from pathlib import Path
from torch import topk
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

flags.DEFINE_string('model', None, 'trained model to use for prediction')
flags.DEFINE_string('images', None, 'images to test on')

def main(_argv):
    model = load_learner(FLAGS.model, device='cpu')
    output_file = os.path.split(FLAGS.images)[-1] + '_predictions.txt'
    with open(os.path.join('./State-Classifier/results/',output_file), "w") as f:

        for img in Path(FLAGS.images).glob('*'):
            pred = model.predict(open_image(img))
            _, top5_states = topk(pred[2], k=5)
            f.write('Predictions for: ' + str(os.path.split(img)[-1]) + ' ' + str([model.data.classes[i] for i in list(top5_states)]) + '\n')

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
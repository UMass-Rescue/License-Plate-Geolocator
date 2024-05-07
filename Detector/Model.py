import time
import tensorflow as tf
import os
from core.functions import *
from absl import app, flags, logging
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

class ObjectDetector:
    def __init__(self ):
        self.framework = 'tf'
        self.weights = './checkpoints/custom-416'
        self.size = 416
        self.tiny = False
        self.model = 'yolov4'
        self.video = '/Users/akhilareddy/License-Plate-Geolocator/Detector/static/files/video.mp4'
        self.output = './detections/results.avi'
        self.output_format = 'XVID'
        self.iou = 0.45
        self.score = 0.25
        self.dont_show = False
        self.crop = True

    def load_config(self):
        config_params = {
            'framework': self.framework,
            'weights': self.weights,
            'size': self.size,
            'tiny': self.tiny,
            'model': self.model,
            'video': self.video,
            'output': self.output,
            'output_format': self.output_format,
            'iou': self.iou,
            'score': self.score,
            'dont_show': self.dont_show,
            'crop': self.crop
        }
        return config_params

    def detect_objects(self):
        config_params = self.load_config()
        config = ConfigProto()
        config.gpu_options.allow_growth = True
        session = InteractiveSession(config=config)
        STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(config_params)
        input_size = config_params['size']
        video_path = config_params['video']

        video_name = video_path.split('/')[-1]
        video_name = video_name.split('.')[0]

        if config_params['framework'] == 'tflite':
            interpreter = tf.lite.Interpreter(model_path=config_params['weights'])
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            print(input_details)
            print(output_details)
        else:
            saved_model_loaded = tf.saved_model.load(config_params['weights'], tags=[tag_constants.SERVING])
            infer = saved_model_loaded.signatures['serving_default']

        # begin video capture
        try:
            vid = cv2.VideoCapture(int(config_params['video']))
        except:
            vid = cv2.VideoCapture(config_params['video'])

        out = None

        if config_params['output']:
            # by default VideoCapture returns float instead of int
            width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(vid.get(cv2.CAP_PROP_FPS))
            codec = cv2.VideoWriter_fourcc(*config_params['output_format'])
            out = cv2.VideoWriter(config_params['output'], codec, fps, (width, height))

        frame_num = 0
        while True:
            return_value, frame = vid.read()
            if return_value:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_num += 1
                image = Image.fromarray(frame)
            else:
                print('Video has ended or failed, try a different video format!')
                break

            frame_size = frame.shape[:2]
            image_data = cv2.resize(frame, (input_size, input_size))
            image_data = image_data / 255.
            image_data = image_data[np.newaxis, ...].astype(np.float32)
            start_time = time.time()

            if config_params['framework'] == 'tflite':
                interpreter.set_tensor(input_details[0]['index'], image_data)
                interpreter.invoke()
                pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
                if config_params['model'] == 'yolov3' and config_params['tiny'] == True:
                    boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                    input_shape=tf.constant([input_size, input_size]))
                else:
                    boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                    input_shape=tf.constant([input_size, input_size]))
            else:
                batch_data = tf.constant(image_data)
                pred_bbox = infer(batch_data)
                for key, value in pred_bbox.items():
                    boxes = value[:, :, 0:4]
                    pred_conf = value[:, :, 4:]

            boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
                boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
                scores=tf.reshape(
                    pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
                max_output_size_per_class=50,
                max_total_size=50,
                iou_threshold=config_params['iou'],
                score_threshold=config_params['score']
            )
            pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
            image = utils.draw_bbox(frame, pred_bbox)

            # read in all class names from config
            class_names = utils.read_class_names(cfg.YOLO.CLASSES)

            allowed_classes = list(class_names.values())

            # if crop flag is enabled, crop each detection and save it as new image
            if config_params['crop']:
                crop_rate = 20 # capture images every so many frames (ex. crop photos every 150 frames)
                crop_path = os.path.join(os.getcwd(), 'static', 'images1')  # New path
                print(crop_path)
                try:
                    os.mkdir(crop_path)
                except FileExistsError:
                    pass
                if frame_num % crop_rate == 0:
                    final_path = os.path.join(crop_path)
                    try:
                        os.mkdir(final_path)
                    except FileExistsError:
                        pass          
                    crop_objects(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), pred_bbox, final_path, allowed_classes, 'video',frame_num)
                else:
                    pass

        #     fps = 1.0 / (time.time() - start_time)
        #     result = np.asarray(image)
        #     cv2.namedWindow("result", cv2.WINDOW_NORMAL)

        #     result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
        #     if not config_params['dont_show']:
        #         cv2.imshow("result", result)
            
        #     if config_params['output']:
        #         out.write(result)
        #     if cv2.waitKey(1) & 0xFF == ord('q'): break
        # cv2.destroyAllWindows()
        return True 


if __name__ == '__main__':
    detector = ObjectDetector()
    res = detector.detect_objects()
    print(res)

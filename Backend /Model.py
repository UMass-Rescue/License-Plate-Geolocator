import tensorflow as tf
import core.utils as utils
from core.config import cfg
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

class LicensePlateDetector:
    def __init__(self):
        self.framework = 'tf'
        self.weights_path = './checkpoints/yolov4-416'
        self.size = 416
        self.tiny = False
        self.model = 'yolov4'
        self.images_path = ['./data/images/kite.jpg']
        self.output_path = './detections/'
        self.iou_threshold = 0.45
        self.score_threshold = 0.25
        self.dont_show = False

    def detect_license_plates(self):
        # Initialize TensorFlow session
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        config = ConfigProto()
        config.gpu_options.allow_growth = True
        session = InteractiveSession(config=config)

        # Load YOLO config
        STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config({
            'size': self.size,
            'tiny': self.tiny,
            'model': self.model
        })

        # Load model
        if self.framework == 'tflite':
            interpreter = tf.lite.Interpreter(model_path=self.weights_path)
        else:
            saved_model_loaded = tf.saved_model.load(self.weights_path, tags=[tag_constants.SERVING])

        # Loop through images and run YOLOv4 model on each
        for count, image_path in enumerate(self.images_path, 1):
            original_image = cv2.imread(image_path)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

            image_data = cv2.resize(original_image, (self.size, self.size))
            image_data = image_data / 255.

            images_data = np.expand_dims(image_data, axis=0).astype(np.float32)

            if self.framework == 'tflite':
                interpreter.allocate_tensors()
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
                interpreter.set_tensor(input_details[0]['index'], images_data)
                interpreter.invoke()
                pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
                if self.model == 'yolov3' and self.tiny:
                    boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=self.score_threshold,
                                                    input_shape=tf.constant([self.size, self.size]))
                else:
                    boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=self.score_threshold,
                                                    input_shape=tf.constant([self.size, self.size]))
            else:
                infer = saved_model_loaded.signatures['serving_default']
                batch_data = tf.constant(images_data)
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
                iou_threshold=self.iou_threshold,
                score_threshold=self.score_threshold
            )
            pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]

            # Read class names from config
            class_names = utils.read_class_names(cfg.YOLO.CLASSES)

            # By default allow all classes in .names file
            allowed_classes = list(class_names.values())

            image = utils.draw_bbox(original_image, pred_bbox, allowed_classes=allowed_classes)

            image = Image.fromarray(image.astype(np.uint8))
            if not self.dont_show:
                image.show()
            image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
            cv2.imwrite(self.output_path + 'detection' + str(count) + '.png', image)

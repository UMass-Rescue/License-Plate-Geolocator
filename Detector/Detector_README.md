This detector was adapted from: https://github.com/theAIGuysCode/tensorflow-yolov4-tflite/tree/master and slightly modified.

The weights for this model are found at: https://drive.google.com/drive/folders/1MqEfknZQ5Q0qX5GkdeTdtCzTq3zVf1i6?usp=sharing
To run locally move variables.data-00000-of-00001 into Detector/checkpoints/custom-416/variables/ and move custom.weights into Detector/data

Command to run model on an image (change directory into Detector folder):
python ./detect.py --weights ./checkpoints/custom-416 --size 416 --model yolov4 --images ./data/images/image_name.jpg

Command to run model on a video (change directory into Detector folder):
python ./detect_video.py --weights ./checkpoints/custom-416 --size 416 --model yolov4 --video ./data/video/video_name.mp4
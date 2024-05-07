This detector was taken from: https://github.com/theAIGuysCode/tensorflow-yolov4-tflite/tree/master and slightly modified.

Use command "git lfs pull" when working with files in this folder in order to view and edit files directly (instead of raw)

The weights for this model are found at: https://drive.google.com/drive/folders/1MqEfknZQ5Q0qX5GkdeTdtCzTq3zVf1i6?usp=sharing
(Note the file is too large to store directly with Github)

Run model on image command:
python ./detect.py --weights ./checkpoints/custom-416 --size 416 --model yolov4 --images ./data/images/image_name.jpg

Run model on video command:
python ./detect_video.py --weights ./checkpoints/custom-416 --size 416 --model yolov4 --video ./data/video/video_name.mp4
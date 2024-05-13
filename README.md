# License-Plate-Geolocator
Our goal is to build a geolocator that can recognize license plates in images/videos in order to aid law enforcement in locating persons of interest. This is done using two separate machine learning models sequentially:
* The first model detects licence plates and isolates them from images/videos (adapted from https://github.com/theAIGuysCode/tensorflow-yolov4-tflite/tree/master)
* The second model takes the isolated images from the first model and predicts the top 5 most probable states that a plate is from (using ResNet: https://github.com/tornadomeet/ResNet/tree/master)

**Scripts for training and setting up each of the models can be found in the branch labeled "ml"**
## Technical Approch 
1) Extract all the images from Video
2) Find all contours in the image frame
3) Pass every image to the Classification model to identify the top 5 states


## Setup Instructions
1) Clone the repository
2) Download the model files from the Google Drive from the repository and save them in the model folder. 
3) Run pip3 install -r requirements.txt. This installs all the necessary libraries.
4) Run python3 app.py
5) Open http://127.0.0.1:5000/ on any browser to view the UI.


### Implementation (Completed, On-going, Future Plans)
Completed 

# License-Plate-Geolocator
Our goal is to build a geolocator that can recognize license plates in images/videos in order to aid law enforcement in locating persons of interest. This is done using two separate machine learning models sequentially:
* The first model detects licence plates and isolates them from images/videos (adapted from https://github.com/theAIGuysCode/tensorflow-yolov4-tflite/tree/master)
* The second model takes the isolated images from the first model and predicts the top 5 most probable states that a plate is from (using ResNet: https://github.com/tornadomeet/ResNet/tree/master)

**Scripts for training and setting up each of the models can be found in the branch labeled "ml"**



## Setup Instructions
1) Clone the repository
2) Download the model files from the Google Drive from the repository and save them in the model folder.
4) Run pip3 install -r requirements.txt. This installs all the necessary libraries.
5) Run python3 app.py
6) Open http://127.0.0.1:5000/ on any browser to view the UI.

Google Drive link :  https://drive.google.com/drive/folders/1XJRzu6WLNre8euyKdlRqHElFOu0mZO_t?usp=sharing
## Implementation (Completed, On-going, Future Plans)
### Completed: 
1) The Detector model is fully implemented according to our specific needs and incorporated with the most recent State Classifier model
2) The State Classifier model works with high accuracy on current test data
3) Frontend is fully implemented. Upload, results pages works as intented
4) Backend operations include server-side tasks, data processing, and interfacing with ML models
5) Frontend provides the user interface for interacting with the application

### Future Works
1) Have a profile page (multi user application)
2) Keep track of User history and have a specific profile page=
3) Integrate backend with Immich
4) Find more data for training the State Classifier, particularly videos to be run through the first model, as well as realistically bad quality data

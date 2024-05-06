INSTRUCTIONS FOR RUNNING LOCALLY (using resnet models)
1. First, move test data from into State-Classifier/data/images
2. Next, move the model you wish to use from https://drive.google.com/drive/folders/1JU_sQ7SvfklGEbh08_qw0s5Gvu6cFOmu?usp=sharing
into State-Classifier/models
(The most up to date model currently is export.pkl.)
3. Run the command python ./State-Classifier/classify.py --model ./State-Classifier/models --images ./State-Classifier/images
(The test images path can be changed to any folder with images to be classified)

The data used for training/testing can be found at: https://drive.google.com/drive/folders/1RSvWruc5AvOmGoB3RRCimcfynTBXTYxU?usp=sharing
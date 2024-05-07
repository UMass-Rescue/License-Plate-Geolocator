INSTRUCTIONS FOR RUNNING LOCALLY (Using resnet models)
1. First, move test data from into State-Classifier/data/images
2. Next, move the model you wish to use from https://drive.google.com/drive/folders/1JU_sQ7SvfklGEbh08_qw0s5Gvu6cFOmu?usp=sharing
into State-Classifier/models
(The most up to date model currently is export.pkl.)
3. Run the command python c
(The test images path can be changed to any folder with images to be classified)

The data used for training/testing can be found at: https://drive.google.com/drive/folders/1RSvWruc5AvOmGoB3RRCimcfynTBXTYxU?usp=sharing

Model 1 was trained using the initial_model_class with 1 epoch and .01 learning rate:
Exact Prediction - 5.04%
Correct Prediction in Top 3 - 10.86%
Correct Prediction in Top 5 - 16.69%

Model 2 was trained using the initial_model_class with 20 epochs and .001 learning rate:
Exact Prediction - 53.08%
Correct Prediction in Top 3 - 65.4%
Correct Prediction in Top 5 - 69.88%

Model 3 (export.pkl) was trained with the resnet_model_train script, using ResNet50, 50 epochs, and an optimized learning rate:
Exact Prediction Accuracy - 66.67%
Top 3 Prediction Accuracy - 88.89%
Top 5 Prediction Accuracy - 88.89%
from fastai.vision import *
from pathlib import Path
from fastai.metrics import error_rate
from fastai.callbacks import *
import torch
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Read in data
path = Path('D:\geolocator_temp\images_with_reduced_res_split')
dataset = ImageDataBunch.from_folder(path, size=224, num_workers=0, train='train', test='test', valid='val').normalize(imagenet_stats)

# Train the ResNet CNN
learn = create_cnn(dataset, models.resnet101, metrics=error_rate)
# learn.data.batch_size = 2
early_stop = EarlyStoppingCallback(learn, patience=20)
save_best_model = SaveModelCallback(learn, name='resnet_model2')
defaults.device = torch.device('cuda') # makes sure the gpu is used
learn.fit_one_cycle(50, callbacks=[early_stop, save_best_model])

# Load the best trained model
learn.load('resnet_model2')

# Find the best learning rate for the model
learn.unfreeze()
def find_appropriate_lr(model:Learner, lr_diff:int = 15, loss_threshold:float = .05, adjust_value:float = 1, plot:bool = False) -> float:
    #Run the Learning Rate Finder
    model.lr_find()
    
    #Get loss values and their corresponding gradients, and get lr values
    losses = np.array(model.recorder.losses)
    min_loss_index = np.argmin(losses)
    
    #loss_grad = np.gradient(losses)
    lrs = model.recorder.lrs
    
    #return the learning rate that produces the minimum loss divide by 10   
    return lrs[min_loss_index] / 10

optimal_lr = find_appropriate_lr(learn)

# Retrain using best learning rate and reload newly-optimized model
learn.fit_one_cycle(50, max_lr=slice(optimal_lr/10, optimal_lr), callbacks=[early_stop, save_best_model])
learn.load('resnet_model2')

# Save the optimized model
learn.path = Path('./State-Classifier/models')
#learn.save('resnet_model1')
learn.export()

# Compute accuracies for test images
model = load_learner('./State-Classifier/models', device='cpu')

total = 0
total_noReduced = 0
total_90 = 0
total_93 = 0
total_97 = 0

correct_1_total = 0
correct_3_total = 0
correct_5_total = 0
correct_1_noReduced = 0
correct_3_noReduced = 0
correct_5_noReduced = 0
correct_1_90 = 0
correct_3_90 = 0
correct_5_90 = 0
correct_1_93 = 0
correct_3_93 = 0
correct_5_93 = 0
correct_1_97 = 0
correct_3_97 = 0
correct_5_97 = 0

for state in Path('D:\\geolocator_temp\\images_with_reduced_res_split\\test').glob('*'):
    for img in Path(str(state)).glob('*'):
        pred = model.predict(open_image(img))
        true_state = str(state).split('\\')[-1]


        predicted1 = torch.argmax(pred[2])
        __, predicted3 = torch.topk(pred[2], k=3)
        ___, predicted5 = torch.topk(pred[2], k=5)

        total += 1
        if '_90pc' not in str(img) and '_93pc' not in str(img) and '_97pc' not in str(img):
            total_noReduced += 1
        elif '_90pc' in str(img):
            total_90 += 1
        elif '_93pc' in str(img):
            total_93 += 1
        elif '_97pc' in str(img):
            total_97 += 1
        
        if true_state == model.data.classes[predicted1]:
            correct_1_total += 1
            if '_90pc' not in str(img) and '_93pc' not in str(img) and '_97pc' not in str(img):
                correct_1_noReduced += 1
            elif '_90pc' in str(img):
                correct_1_90 += 1
            elif '_93pc' in str(img):
                correct_1_93 += 1
            elif '_97pc' in str(img):
                correct_1_97 += 1

        if true_state in [model.data.classes[i] for i in list(predicted3)]:
            correct_3_total += 1
            if '_90pc' not in str(img) and '_93pc' not in str(img) and '_97pc' not in str(img):
                correct_3_noReduced += 1
            elif '_90pc' in str(img):
                correct_3_90 += 1
            elif '_93pc' in str(img):
                correct_3_93 += 1
            elif '_97pc' in str(img):
                correct_3_97 += 1

        if true_state in [model.data.classes[i] for i in list(predicted5)]:
            correct_5_total += 1
            if '_90pc' not in str(img) and '_93pc' not in str(img) and '_97pc' not in str(img):
                correct_5_noReduced += 1
            elif '_90pc' in str(img):
                correct_5_90 += 1
            elif '_93pc' in str(img):
                correct_5_93 += 1
            elif '_97pc' in str(img):
                correct_5_97 += 1

print('Results over all images: ')
print('Exact Prediction Accuracy: ' + str(round(100 * correct_1_total / total, 2)) + '%')
print('Top 3 Prediction Accuracy: ' + str(round(100 * correct_3_total / total, 2)) + '%')
print('Top 5 Prediction Accuracy: ' + str(round(100 * correct_5_total / total, 2)) + '%')

print('\n')
print('Results for non-reduced resolution images: ')
print('Exact Prediction Accuracy: ' + str(round(100 * correct_1_noReduced / total_noReduced, 2)) + '%')
print('Top 3 Prediction Accuracy: ' + str(round(100 * correct_3_noReduced / total_noReduced, 2)) + '%')
print('Top 5 Prediction Accuracy: ' + str(round(100 * correct_5_noReduced / total_noReduced, 2)) + '%')

print('\n')
print('Results for 90% ' + 'reduced resolution images: ')
print('Exact Prediction Accuracy: ' + str(round(100 * correct_1_90 / total_90, 2)) + '%')
print('Top 3 Prediction Accuracy: ' + str(round(100 * correct_3_90 / total_90, 2)) + '%')
print('Top 5 Prediction Accuracy: ' + str(round(100 * correct_5_90 / total_90, 2)) + '%')

print('\n')
print('Results for 93% ' + 'reduced resolution images: ')
print('Exact Prediction Accuracy: ' + str(round(100 * correct_1_93 / total_93, 2)) + '%')
print('Top 3 Prediction Accuracy: ' + str(round(100 * correct_3_93 / total_93, 2)) + '%')
print('Top 5 Prediction Accuracy: ' + str(round(100 * correct_5_93 / total_93, 2)) + '%')

print('\n')
print('Results for 97% ' + 'reduced resolution images: ')
print('Exact Prediction Accuracy: ' + str(round(100 * correct_1_97 / total_97, 2)) + '%')
print('Top 3 Prediction Accuracy: ' + str(round(100 * correct_3_97 / total_97, 2)) + '%')
print('Top 5 Prediction Accuracy: ' + str(round(100 * correct_5_97 / total_97, 2)) + '%')

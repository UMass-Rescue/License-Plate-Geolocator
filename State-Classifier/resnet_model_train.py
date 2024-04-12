from fastai.vision import *
from pathlib import Path
from fastai.metrics import error_rate
from fastai.callbacks import *
import torch
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Read in data
path = Path('D:\geolocator_temp\images_split')
data = ImageDataBunch.from_folder(path, size=224, num_workers=0, train='train', test='test', valid='val').normalize(imagenet_stats)

# Train the ResNet CNN
learn = create_cnn(data, models.resnet50, metrics=error_rate)
# learn.data.batch_size = 10
early_stop = EarlyStoppingCallback(learn, patience=20)
save_best_model = SaveModelCallback(learn, name='resnet_model1')
defaults.device = torch.device('cuda') # makes sure the gpu is used
learn.fit_one_cycle(50, callbacks=[early_stop, save_best_model])

# Load the best trained model
learn.load('resnet_model1')

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
learn.load('resnet_model1')

# Save the optimized model
learn.path = Path('./State-Classifier/models')
#learn.save('resnet_model1')
learn.export()

# Compute accuracies for test images
model = load_learner('./State-Classifier/models', device='cpu')

total = 0
correct_1 = 0
correct_3 = 0
correct_5 = 0

for state in Path('D:\\geolocator_temp\\images_sample_split\\test').glob('*'):
    for img in Path(str(state)).glob('*'):
        pred = model.predict(open_image(img))
        true_state = str(state).split('\\')[-1]


        predicted1 = torch.argmax(pred[2])
        __, predicted3 = torch.topk(pred[2], k=3)
        ___, predicted5 = torch.topk(pred[2], k=5)

        total += 1
        if true_state == model.data.classes[predicted1]:
            correct_1 += 1
        if true_state in [model.data.classes[i] for i in list(predicted3)]:
            correct_3 += 1
        if true_state in [model.data.classes[i] for i in list(predicted5)]:
            correct_5 += 1
            
exact_accuracy = round(100 * correct_1 / total, 2)
top3_accuracy = round(100 * correct_3 / total, 2)
top5_accuracy = round(100 * correct_5 / total, 2)

print('Exact Prediction Accuracy: ' + str(exact_accuracy) + '%')
print('Top 3 Prediction Accuracy: ' + str(top3_accuracy) + '%')
print('Top 5 Prediction Accuracy: ' + str(top5_accuracy) + '%')

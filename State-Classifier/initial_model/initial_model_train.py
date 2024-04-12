import torch
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from initial_model_class import state_classifier as sc
import initial_model_classify as cy

# This script was used to convert the dataset from Kaggle into a usable form

# Extract the data and create a tuple of the classes (U.S. states)
path = './State-Classifier/data/images'
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((128, 128))
])
dataset = ImageFolder(root=path, transform=transform)
classes= ("Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware", "Florida", "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada", "NewHampshire", "NewJersey", "NewMexico", "NewYork", "NorthCarolina", "NorthDakota", "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "RhodeIsland", "SouthCarolina", "SouthDakota", "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington", "WashingtonDC", "WestVirginia", "Wisconsin", "Wyoming")

# Divide the data into train and test sets and create dataloaders
train_length=int(0.8* len(dataset))
test_length=len(dataset)-train_length
train_dataset,test_dataset=random_split(dataset,(train_length,test_length))
train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

# Train and save the model
# model = sc()
# model.fit(model.parameters(), train_dataloader, 20, .001)
# model.save_model('model1')

acc1, acc3, acc5 = cy.accuracy('model1', test_dataloader)

print('Exact Guess Accuracy: ' + str(acc1) + ' %')
print('Top 3 Accuracy: ' + str(acc3) + ' %')
print('Top 5 Accuracy: ' + str(acc5) + ' %')
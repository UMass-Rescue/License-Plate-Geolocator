import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

class state_classifier(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(13456, 5000)
        self.fc2 = nn.Linear(5000, 1000)
        self.fc3 = nn.Linear(1000, 51)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def fit(self, params, dataloader, num_epochs, learning_rate):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(params, lr=learning_rate)

        for epoch in range(num_epochs):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(dataloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0

    def save_model(self, model_str):
        torch.save({'model_state_dict': self.state_dict()}, "./State-Classifier/initial_model/models/" + model_str + ".pt")
    
    def load(self, model_str):
        checkpoint = torch.load("./State-Classifier/initial_model/models/" + model_str + ".pt")
        self.load_state_dict(checkpoint["model_state_dict"])

    def predict(self, data):
        return self.forward(data)

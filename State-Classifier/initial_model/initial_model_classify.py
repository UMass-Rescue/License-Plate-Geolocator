from initial_model_class import state_classifier as sc
import torch

# Makes predictions on a dataset given a dataloader and the model name as a string
def make_predictions(model_str, dataloader):
    model = sc()
    model.load(model_str)

    predictions = []

    with torch.no_grad():
        for data in dataloader:
            image, label = data
            # calculate outputs by running images through the network
            output = model.predict(image)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(output.data, dim=1)
            predictions.append(predicted)

    return predictions

# Compute the accuracy of predicting on a dataset by counting the percentage of exact predictions as well as in the top 3 and top 5 classifications
def accuracy(model_str, dataloader):
    
    model = sc()
    model.load(model_str)

    correct_1 = 0
    correct_3 = 0
    correct_5 = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            image, label = data
            # calculate outputs by running images through the network
            output = model.predict(image)
            # the class with the highest energy is what we choose as prediction
            _, predicted1 = torch.max(output.data, dim=1)
            __, predicted3 = torch.topk(output.data, k=3, dim=1)
            ___, predicted5 = torch.topk(output.data, k=5, dim=1)
            total += 1
            if label == predicted1:
                correct_1 += 1
            if label in predicted3:
                correct_3 += 1
            if label in predicted5:
                correct_5 += 1

    exact_accuracy = round(100 * correct_1 / total, 2)
    top3_accuracy = round(100 * correct_3 / total, 2)
    top5_accuracy = round(100 * correct_5 / total, 2)

    return exact_accuracy, top3_accuracy, top5_accuracy
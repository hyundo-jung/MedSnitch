import torch
from torch import nn
import torch.optim as optim
import pandas as pd
import xgboost
from xgboost import XGBClassifier
import os

# Data for both models
dataset = pd.read_csv("data/processed/processed_claims.csv")
shuffled_dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)

print(dataset)

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data.values, dtype=torch.float32)
        self.labels = torch.tensor(labels.values, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

data_size = dataset.shape[0]
train_size = int(0.6 * data_size)
val_size = int(0.2 * data_size)

train_params, train_labels = shuffled_dataset[:train_size, :-1], shuffled_dataset[:train_size, -1]
val_params, val_labels = shuffled_dataset[train_size:(train_size + val_size), :-1], shuffled_dataset[train_size:(train_size + val_size), -1]
test_params, test_labels = shuffled_dataset[(train_size + val_size):, :-1], shuffled_dataset[(train_size + val_size):, -1]

train_tensor = CustomDataset(train_params, train_labels)
val_tensor = CustomDataset(val_params, val_labels)
test_tensor = CustomDataset(test_params, test_labels)

# Neural Network Model
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(14, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.layers(x)

criterion = nn.BCEWithLogitsLoss()

def train(epochs, train_loader, val_loader):
    neural_network = Model()
    optimizer = optim.Adam(neural_network.parameters(), lr=0.1)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    for i in range(epochs):
        training_loss = 0.0
        neural_network.train()
        for j, data in enumerate(train_loader):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = neural_network(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            training_loss += loss.item()

        valid_loss = 0.0
        neural_network.eval()
        with torch.no_grad():
            for j, data in enumerate(val_loader):
                inputs, labels = data
                outputs = neural_network(inputs)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()

        scheduler.step(valid_loss)

        print(f"Epoch {i + 1} completed.") 
        print(f"Training Loss: {training_loss / len(train_loader)}")
        print(f"Validation Loss: {valid_loss / len(val_loader)}")
    print("Finished Training")

    return neural_network


new_nn = train(5, train_tensor, val_tensor)
test_loss = 0.0
new_nn.eval()
with torch.no_grad():
    for i, data in enumerate(test_tensor):
        inputs, labels = data
        outputs = new_nn(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
    print(f"Test Loss: {test_loss / len(test_tensor)}")

torch.save(new_nn.state_dict(), "model.pth")


# XGBoost Model
bst = XGBClassifier(n_estimators=2, max_depth=2, learning_rate=1, objective='binary:logistic')

# fit model
bst.fit(train_params, train_labels)

# make predictions
preds = bst.predict(test_params)

correct = 0
for i in range(len(preds)):
    if preds[i] == test_labels[i]:
        correct += 1
print(f"Accuracy: {correct / len(preds)}")

bst.save_model('xg.json')
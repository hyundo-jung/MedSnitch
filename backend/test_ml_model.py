import torch
from torch import nn, sigmoid
import torch.optim as optim
import pandas as pd
import xgboost
from xgboost import XGBClassifier
import os
import joblib
scaler = joblib.load('data/processed/scaler.joblib')

# Data for both models
dataset = pd.read_csv("data/processed/processed_claims.csv")

dataset["first_procedure"] = dataset["first_procedure"].fillna(0)
dataset["first_diagnosis"] = dataset["first_diagnosis"].astype("category").cat.codes
# print(dataset.dtypes)

# print(scaler.feature_names_in_)

feature_columns = ['claimType', 'StayDuration', 'cost', 'num_diagnoses', 'DiagnosisCategory', 'num_procedures', 'first_procedure', 'Gender', 'Race', 'isWeekend', 'ClaimDuration']
features = dataset[feature_columns]
features_scaled = scaler.transform(features)

scaled_dataset = pd.DataFrame(features_scaled, columns=features.columns)
scaled_dataset["ClaimDayOfYear"] = dataset["ClaimDayOfYear"]
scaled_dataset["Age"] = dataset["Age"]
scaled_dataset["first_diagnosis"] = dataset["first_diagnosis"]
scaled_dataset["is_fraudulent"] = dataset["is_fraudulent"]

# shuffled_dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)
shuffled_dataset = scaled_dataset.sample(frac=1, random_state=42).reset_index(drop=True)

print(dataset.isna().sum())


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

train_params = shuffled_dataset.iloc[:train_size, :-1]
train_labels = shuffled_dataset.iloc[:train_size,  -1].reset_index(drop=True)

val_params = shuffled_dataset.iloc[train_size : train_size + val_size, :-1]
val_labels = shuffled_dataset.iloc[train_size : train_size + val_size, -1].reset_index(drop=True)

test_params = shuffled_dataset.iloc[(train_size + val_size):, :-1]
test_labels = shuffled_dataset.iloc[(train_size + val_size):, -1].reset_index(drop=True)

train_tensor = CustomDataset(train_params, train_labels)
val_tensor = CustomDataset(val_params, val_labels)
test_tensor = CustomDataset(test_params, test_labels)

train_loader = torch.utils.data.DataLoader(train_tensor, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_tensor, batch_size=64, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_tensor, batch_size=64, shuffle=False)


# Neural Network Model
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(14, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.layers(x)

    
num_zeros = (train_labels == 0).sum()
num_ones = (train_labels == 1).sum()
pos_weight = num_zeros / num_ones

criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32))

# counts      = test_labels.value_counts()                # raw counts
# proportions = test_labels.value_counts(normalize=True)  # relative frequency

# print("Counts:\n", counts)
# print("\nProportions:\n", proportions)

def train(epochs, train_loader, val_loader):
    neural_network = Model()
    optimizer = optim.Adam(neural_network.parameters(), lr=1e-6, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    for i in range(epochs):
        training_loss = 0.0
        neural_network.train()
        for j, data in enumerate(train_loader):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = neural_network(inputs)

            if torch.isnan(outputs).any():
                print(f"NaN detected in outputs at batch {j}")
                return neural_network

            loss = criterion(outputs, labels.unsqueeze(1))

            if torch.isnan(loss):
                print(f"NaN detected in loss at batch {j}")
                return neural_network

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(neural_network.parameters(), max_norm=1.0)
            optimizer.step()
            training_loss += loss.item()

        valid_loss = 0.0
        neural_network.eval()
        with torch.no_grad():
            for j, data in enumerate(val_loader):
                inputs, labels = data
                outputs = neural_network(inputs)
                loss = criterion(outputs, labels.unsqueeze(1))
                valid_loss += loss.item()

        scheduler.step(valid_loss)

        print(f"Epoch {i + 1} completed.") 
        print(f"Training Loss: {training_loss / len(train_loader)}")
        print(f"Validation Loss: {valid_loss / len(val_loader)}")

    print("Finished Training")
    return neural_network


new_nn = train(20, train_loader, val_loader)
test_loss = 0.0
new_nn.eval()

correct = 0
total = 0
with torch.no_grad():
    for i, data in enumerate(test_loader):
        inputs, labels = data
        outputs = new_nn(inputs)
        loss = criterion(outputs, labels.unsqueeze(1))
        test_loss += loss.item()

        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float().squeeze(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    print(f"Test Loss: {test_loss / len(test_tensor)}")
    print(f"NN Test Accuracy: {correct/total}")

# all_preds = []
# all_labels = []

# new_nn.eval()
# with torch.no_grad():
#     for inputs, labels in test_loader:
#         logits = new_nn(inputs)
#         probs = torch.sigmoid(logits)
#         batch_preds = (probs > 0.5).float().squeeze(1)
#         all_preds.append(batch_preds)
#         all_labels.append(labels)

# all_preds = torch.cat(all_preds)
# all_labels = torch.cat(all_labels)

# # Prediction counts
# uv, uc = torch.unique(all_preds, return_counts=True)
# print("Pred counts:", dict(zip(uv.tolist(), uc.tolist())))

# # True label distribution
# import numpy as np
# y = all_labels.numpy().astype(int)
# dist = np.bincount(y, minlength=2) / len(y)
# print(f"True class distribution: 0 → {dist[0]:.2f}, 1 → {dist[1]:.2f}")


torch.save(new_nn.state_dict(), "model.pth")

# XGBoost Model
bst = XGBClassifier(n_estimators=2, max_depth=2, learning_rate=1e-3, objective='binary:logistic')

# fit model
bst.fit(train_params, train_labels)

# make predictions
preds = bst.predict(test_params)

correct = 0
for i in range(len(preds)):
    if preds[i] == test_labels[i]:
        correct += 1
print(f"XGBoost Accuracy: {correct / len(preds)}")

bst.save_model('xg.json')
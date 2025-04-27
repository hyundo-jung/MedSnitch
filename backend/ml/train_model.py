import torch
from torch import nn, sigmoid
import torch.optim as optim
import pandas as pd
import xgboost
from xgboost import XGBClassifier
import os
import numpy as np
import joblib
import test_ml_model

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

scaler = joblib.load(os.path.join(BASE_DIR, 'data', 'processed', 'scaler.joblib'))

# Data for both models
dataset = pd.read_csv(os.path.join(BASE_DIR, "data/processed/processed_claims.csv"))

dataset["first_procedure"] = dataset["first_procedure"].fillna(0)
dataset["first_diagnosis"] = dataset["first_diagnosis"].astype("category").cat.codes
# print(dataset.dtypes)

# print(scaler.feature_names_in_)

# print(dataset["ClaimDayOfYear"])

feature_columns = ['claimType', 'StayDuration', 'cost', 'num_diagnoses', 'DiagnosisCategory', 'num_procedures', 'first_procedure', 'Gender', 'Race', 'isWeekend', 'ClaimDuration']
features = dataset[feature_columns]
features_scaled = scaler.transform(features)

scaled_dataset = pd.DataFrame(features_scaled, columns=features.columns)
scaled_dataset["ClaimDayOfYear"] = dataset["ClaimDayOfYear"]
scaled_dataset["Age"] = dataset["Age"]
scaled_dataset["first_diagnosis"] = dataset["first_diagnosis"]
scaled_dataset["is_fraudulent"] = dataset["is_fraudulent"]

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

num_zeros = (train_labels == 0).sum()
num_ones = (train_labels == 1).sum()
pos_weight = (num_zeros / num_ones) * 0.8 # 

criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32))

counts      = test_labels.value_counts()                # raw counts
proportions = test_labels.value_counts(normalize=True)  # relative frequency

print("Counts:\n", counts)
print("\nProportions:\n", proportions)

def train_nn_model(epochs=20):
    neural_network = test_ml_model.Model()
    optimizer = optim.Adam(neural_network.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    for i in range(epochs):
        training_loss = 0.0
        neural_network.train()
        for j, data in enumerate(train_loader):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = neural_network(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
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




def train_xgb_model():
    bst = XGBClassifier(n_estimators=800, max_depth=8, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, random_state=42, 
                    reg_alpha=0.1, reg_lambda=1.0, objective='binary:logistic', scale_pos_weight=pos_weight, eval_metric='auc')
    
    bst.fit(train_params, train_labels)
    return bst

if __name__ == "__main__":
    print("Starting NN training...")
    new_nn = train_nn_model()
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

    torch.save(new_nn.state_dict(), os.path.join(BASE_DIR, "model.pth"))



    print("\nStarting XGBoost training...")
    bst = train_xgb_model()
    preds = bst.predict(test_params)

    correct = 0
    for i in range(len(preds)):
        if preds[i] == test_labels[i]:
            correct += 1
    print(f"XGBoost Accuracy: {correct / len(preds)}")

    unique_vals, counts = np.unique(preds, return_counts=True)
    print("XGBoost Prediction Distribution:")
    for val, cnt in zip(unique_vals, counts):
        print(f"  Class {int(val)}: {cnt} samples")

    bst.save_model('xg.json')
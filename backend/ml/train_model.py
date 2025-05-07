import torch
import torch.optim as optim
import pandas as pd
from xgboost import XGBClassifier
import os
import joblib
import test_ml_model
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

scaler = joblib.load(os.path.join(BASE_DIR, 'data', 'processed', 'scaler.joblib'))

# Data for both models
dataset = pd.read_csv(os.path.join(BASE_DIR, "data/processed/processed_claims.csv"))

feature_columns = ['StayDuration', 'cost', 'num_diagnoses', 'num_procedures', 'Gender', 'Race', 'ClaimDay_sin', 'ClaimDay_cos', 'ClaimDuration']
features = dataset[feature_columns]
features_scaled = scaler.transform(features)

scaled_dataset = pd.DataFrame(features_scaled, columns=features.columns)
to_add = ['claimType', 'DiagnosisCategory', 'isWeekend', 'Age', 'first_procedure', 'is_fraudulent']
for col in to_add:
    scaled_dataset[col] = dataset[col]

shuffled_dataset = scaled_dataset.sample(frac=1, random_state=42).reset_index(drop=True)

print(dataset.isna().sum())

class FraudDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe):
        # Separate numeric and categorical columns
        self.x_numeric = dataframe.drop(columns=["DiagnosisCategory", "is_fraudulent"]).values.astype("float32")
        self.x_diag_cat = dataframe["DiagnosisCategory"].astype("int64").values  # Must be long for embedding
        self.labels = dataframe["is_fraudulent"].values.astype("float32")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.x_numeric[idx], dtype=torch.float32),
            torch.tensor(self.x_diag_cat[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )

# SPLIT
shuffled = scaled_dataset.sample(frac=1, random_state=42).reset_index(drop=True)
train_size = int(0.6 * len(shuffled))
val_size = int(0.2 * len(shuffled))

train_df = shuffled.iloc[:train_size]
val_df = shuffled.iloc[train_size:train_size + val_size]
test_df = shuffled.iloc[train_size + val_size:]

# LOADERS
train_tensor = FraudDataset(train_df)
val_tensor = FraudDataset(val_df)
test_tensor = FraudDataset(test_df)

train_loader = DataLoader(train_tensor, batch_size=64, shuffle=True)
val_loader = DataLoader(val_tensor, batch_size=64, shuffle=False)
test_loader = DataLoader(test_tensor, batch_size=64, shuffle=False)

# LOSS FUNCTION
num_zeros = (train_df["is_fraudulent"] == 0).sum()
num_ones = (train_df["is_fraudulent"] == 1).sum()
pos_weight = (num_zeros / num_ones) * 0.4
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32).to(device))

# PRINT DISTRIBUTIONS
print("Fraud Distribution in Test Set:")
print(test_df["is_fraudulent"].value_counts(normalize=True))

# Save for use in Model()
num_diag_categories = dataset['DiagnosisCategory'].max() + 1
print("num_diag_categories:", num_diag_categories)

def train_nn_model(epochs=20, patience=5):
    neural_network = test_ml_model.Model(num_diag_categories).to(device)
    optimizer = optim.Adam(neural_network.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0

    for i in range(epochs):
        training_loss = 0.0
        neural_network.train()
        # === Training Loop ===
        for x_numeric, x_diag_cat, labels in train_loader:
            x_numeric = x_numeric.to(device)
            x_diag_cat = x_diag_cat.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = neural_network(x_numeric, x_diag_cat)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()
            training_loss += loss.item()

        # === Validation Loop ===
        valid_loss = 0.0
        neural_network.eval()
        with torch.no_grad():
            for x_numeric, x_diag_cat, labels in val_loader:
                x_numeric = x_numeric.to(device)
                x_diag_cat = x_diag_cat.to(device)
                labels = labels.to(device)

                outputs = neural_network(x_numeric, x_diag_cat)
                loss = criterion(outputs, labels.unsqueeze(1))
                valid_loss += loss.item()


        avg_train_loss = training_loss / len(train_loader)
        avg_val_loss = valid_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        print(f"Epoch {i + 1} completed.")
        print(f"Training Loss: {avg_train_loss:.4f}")
        print(f"Validation Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss - 1e-4:  # significant improvement
            best_val_loss = avg_val_loss
            best_model_state = neural_network.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {i + 1} epochs.")
                break

    if best_model_state is not None:
        neural_network.load_state_dict(best_model_state)

    print("Finished Training")
    return neural_network

def train_xgb_model():
    x_train = train_df.drop(columns=["is_fraudulent"])
    y_train = train_df["is_fraudulent"]

    bst = XGBClassifier(
        n_estimators=800,
        max_depth=8,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective='binary:logistic',
        scale_pos_weight=pos_weight,
        eval_metric='auc'
    )

    bst.fit(x_train, y_train)
    return bst

if __name__ == "__main__":
    print("\nStarting XGBoost training...")
    bst = train_xgb_model()

    # save xg model
    bst.save_model('xg.json')

    x_test = test_df.drop(columns=["is_fraudulent"])
    y_test = test_df["is_fraudulent"]

    probs = bst.predict_proba(x_test)[:, 1]
    preds = (probs > 0.5).astype(int)

    precision = precision_score(y_test, preds, zero_division=0)
    recall = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    accuracy = (preds == y_test).mean()

    print("\nXGBoost Model Results (Test Set):")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Precision:     {precision:.4f}")
    print(f"Recall:        {recall:.4f}")
    print(f"F1 Score:      {f1:.4f}")

    print("Starting NN training...")
    new_nn = train_nn_model()
    new_nn.eval()

    # save nn model
    torch.save(new_nn.state_dict(), "nn_model.pt")

    # === Step 1: Find best threshold on validation set ===
    val_probs = []
    val_labels = []

    with torch.no_grad():
        for x_numeric, x_diag_cat, labels in val_loader:
            x_numeric = x_numeric.to(device)
            x_diag_cat = x_diag_cat.to(device)

            outputs = new_nn(x_numeric, x_diag_cat)
            probs = torch.sigmoid(outputs).squeeze(1).cpu().numpy()
            val_probs.extend(probs)
            val_labels.extend(labels.numpy())

    precisions, recalls, thresholds = precision_recall_curve(val_labels, val_probs)
    f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = f1s.argmax()
    best_thresh = 0.2464#thresholds[best_idx]

    print(f"\nBest threshold from validation: {best_thresh:.4f}")
    print(f"Val Precision: {precisions[best_idx]:.4f}, Recall: {recalls[best_idx]:.4f}, F1: {f1s[best_idx]:.4f}")

    # === Step 2: Evaluate on test set using best threshold ===
    test_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x_numeric, x_diag_cat, labels in test_loader:
            x_numeric = x_numeric.to(device)
            x_diag_cat = x_diag_cat.to(device)
            labels = labels.to(device)

            outputs = new_nn(x_numeric, x_diag_cat)
            probs = torch.sigmoid(outputs).squeeze(1)
            preds = (probs > best_thresh).float()

            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    print("\nNN Model Results (Test Set):")
    print(f"Test Accuracy: {correct / total:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
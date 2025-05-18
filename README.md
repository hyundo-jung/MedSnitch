# MedSnitch

A supervised learning system for detecting fraudulent medical billing practices such as phantom billing, unbundling, and duplicate claims. Designed for insurers and regulators, the project includes a frontend dashboard, RESTful API, and multiple classification models. The models are trained on a synthesized kaggle dataset, providing insights on how the model would perform when provided real data.


## Features
- Fraud Detection: Identifies anomalies in medical billing data.

- Multiple Models: Implements both XGBoost and Neural Network classifiers.

- Interactive Dashboard: Visualizes predictions and model performance.

- RESTful API: Facilitates integration with other systems.

## Datasets Used
- Kaggle dataset: https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis/
- HCUP Diagnosis Mappings: https://hcup-us.ahrq.gov/toolssoftware/ccs/ccs.jsp

## Data Preprocessing

The preprocessor:
- Combines inpatient/outpatient claims
- Processes diagnosis/procedure codes (keeps first code + counts)
- Converts dates to features
- Adds beneficiary details
- Calculates costs
- Adds fraud labels
- Scales features
- Removes IDs

## Neural Network Model
### Inputs:

- 13 numerical features (e.g., claim amount, patient age, number of diagnoses).

- 1 categorical diagnosis code, mapped to a vector via an embedding layer (dimension = 8).

### Processing Steps:

1. The diagnosis code is passed through an nn.Embedding layer, transforming the categorical input into a dense 8-dimensional vector.

2. This embedded vector is concatenated with the 13 numeric features to form a 21-dimensional input vector.

3. The combined vector is passed through a series of fully connected layers:

   - Layer 1: Linear (21 → 256), followed by Batch Normalization, ReLU activation, and Dropout (0.3).

   - Layer 2: Linear (256 → 128), followed by Batch Normalization and ReLU activation.

   - Output Layer: Linear (128 → 1)
### Output:
- A single scalar prediction, which is the probability of fraud.
- To produce a classification of fraudulent or not, a custom threshold was used. This threshold was found by maximizing the f1 score.

### Results:
After early stopping was triggered after 19 epochs of training, the following results were achieved:
- Precision: 0.3862
- Recall: 1.0000
- F1 Score: 0.5572

## XG Boost Model
### Training
A model with the following paramaters was trained:
```python
XGBClassifier(
    n_estimators=800,        # Number of boosting rounds (trees)
    max_depth=8,             # Maximum tree depth
    learning_rate=0.03,      # Step size shrinkage
    subsample=0.8,           # Fraction of data sampled per tree
    colsample_bytree=0.8,    # Fraction of features sampled per tree
    reg_alpha=0.1,           # L1 regularization term
    reg_lambda=1.0,          # L2 regularization term
    objective='binary:logistic',  # Binary classification with logistic loss
    scale_pos_weight=pos_weight, # Class imbalance handling
    eval_metric='auc',       # Evaluation metric during training
    random_state=42          # Reproducibility
)
```
### Results:
After training was performed, the following results were achieved:
- Precision: 0.5850
- Recall: 0.0607
- F1 Score: 0.1101

## Interpretation of Results
While the F1 scores achieved were not as high as expected, this project still serves as a strong start to combatting medical bill fraud. Our largest limitation was our lack of access to reliable data due to the privacy of medical bills. We were forced to use a synthesized dataset that also was not properly labeled claim-by-claim.

We could share this project with larger organizations such as insurace companies so that we could gain access to a more reliable dataset. With a real dataset, our models would achieve higher F1 scores.



## Interface Demo
<img width="1516" alt="Screenshot 2025-05-11 at 4 56 58 PM" src="https://github.com/user-attachments/assets/e7d82133-39df-4a76-86c0-7acebaffa6f2" />
<img width="1005" alt="Screenshot 2025-05-11 at 4 58 03 PM" src="https://github.com/user-attachments/assets/a2dd9424-fcab-428a-803b-8b5bbaf6950c" />
<img width="927" alt="Screenshot 2025-05-11 at 4 59 48 PM" src="https://github.com/user-attachments/assets/fdc0d23c-65b3-4966-b7f6-fb7a93b9333d" />
<img width="908" alt="Screenshot 2025-05-11 at 4 57 11 PM" src="https://github.com/user-attachments/assets/7c92ea08-2b05-4b0d-bf75-b6782bf42126" />



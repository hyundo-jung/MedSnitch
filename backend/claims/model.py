import torch
import joblib
import os
from xgboost import XGBClassifier
from ml import test_ml_model
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class ModelHandler:
    def __init__(self):
        metadata_path = os.path.join(BASE_DIR, 'ml', 'nn_model_metadata.json')
        with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        self.num_diag_categories = metadata['num_diag_categories']

        mapping_path = os.path.join(BASE_DIR, 'data', 'mappings', 'icd_to_ccs_mapping.json')
        with open(mapping_path, 'r') as f:
            self.icd_to_ccs_mapping = json.load(f)


        self.nn_model = self.load_nn_model(self.num_diag_categories)
        self.xgb_model = self.load_xgb_model()
        self.scaler = joblib.load(os.path.join(BASE_DIR, 'data', 'processed', 'scaler.joblib'))

    def load_nn_model(self, num_diag_categories):
        # Load the neural network model with the number of diagnostic categories
        metadata_path = os.path.join(BASE_DIR, 'ml', 'nn_model_metadata.json')

        model = test_ml_model.Model(num_diag_categories=num_diag_categories, embedding_dim=8)
        model.load_state_dict(torch.load(os.path.join(BASE_DIR, 'ml', 'nn_model.pt'), map_location=torch.device('cpu')))
        model.eval()
        return model
    
    def load_xgb_model(self):
        # Load the XGBoost model
        model = XGBClassifier()
        model.load_model(os.path.join(BASE_DIR, 'ml', 'xg.json'))
        return model
    
    def map_icd_to_ccs(self, icd_code):
        icd_code = str(icd_code).strip()  # Ensure no extra spaces
        mapped_value = self.icd_to_ccs_mapping.get(icd_code, -1)
        if mapped_value == -1:
            print(f"Warning: ICD code '{icd_code}' not found in mapping.")
        return mapped_value

    def predict_nn(self, x_numeric, x_diag_cat):
        # Split the input into numeric features and diagnosis category

        # Make prediction
        with torch.no_grad():
            output = torch.sigmoid(self.nn_model(x_numeric, x_diag_cat))
        
        # Return binary classification result based on threshold
        return (output.item() > 0.2464)


    def predict_xgb(self, input_array):
        # Predict using XGBoost model
        return int(self.xgb_model.predict(input_array.reshape(1, -1))[0])

# Singleton pattern for model handler
model_handler = None

def get_model_handler():
    global model_handler
    if model_handler is None:
        model_handler = ModelHandler()
    return model_handler

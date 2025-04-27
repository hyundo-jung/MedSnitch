import torch
import joblib
import os
from xgboost import XGBClassifier
from ml import test_ml_model

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class ModelHandler:
    def __init__(self):
        self.nn_model = self.load_nn_model()
        self.xgb_model = self.load_xgb_model()
        self.scaler = joblib.load(os.path.join(BASE_DIR, 'data', 'processed', 'scaler.joblib'))

    def load_nn_model(self):
        model = test_ml_model.Model()
        model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
        model.eval()
        return model
    
    def load_xgb_model(self):
        model = XGBClassifier()
        model.load_model('xg.json')
        return model
    
    def predict_nn(self, input_array):
        tensor_input = torch.tensor(input_array, dtype=torch.float32)
        output = torch.sigmoid(self.nn_model(tensor_input))
        return (output.item() > 0.5)

    def predict_xgb(self, input_array):
        return int(self.xgb_model.predict(input_array.reshape(1, -1))[0])
    
model_handler = None

def get_model_handler():
    global model_handler
    if model_handler is None:
        model_handler = ModelHandler()
    return model_handler
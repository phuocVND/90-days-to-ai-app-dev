from joblib import load
import os

class PriceModel:
    def __init__(self):
        model_path = os.path.join(os.path.dirname(__file__), "price_model.joblib")
        self.model = load(model_path)

    def predict(self, features):
        return self.model.predict([features])[0]

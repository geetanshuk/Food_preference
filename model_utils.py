from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

# Load the model ONCE when the file is imported
model = load_model('scripts/food_classifier.h5')

def predict_food_bytes(img):
    probability = model.predict(img)[0][0]
    return float(probability)
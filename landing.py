from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
from model_utils import predict_food_bytes
from io import BytesIO
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np


app = Flask(__name__)

# model = load_model('food_classifier.h5')
IMG_SIZE = (224, 224)

@app.route('/')
def index():
    return render_template('landing_page.html')

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files['image'] 
    img_bytes = file.read()                               # read raw uploaded bytes
    img = tf.keras.utils.load_img(BytesIO(img_bytes),     # wrap in BytesIO
                              target_size=IMG_SIZE)
    img = tf.keras.utils.img_to_array(img)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    probability = predict_food_bytes(img)

    return f"Probability friend likes it: {probability * 100:.2f}%"

if __name__ == '__main__':
    app.run(debug=True)
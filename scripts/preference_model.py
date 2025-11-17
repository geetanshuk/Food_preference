import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.models import load_model
import numpy as np


IMG_SIZE = (224, 224)

def load_image(path):
    img = tf.keras.utils.load_img(path, target_size=IMG_SIZE)
    img = tf.keras.utils.img_to_array(img)
    img = img / 255.0
    return np.expand_dims(img, axis=0)

model1 = load_model("food_classifier.h5")

img = load_image("new_food.jpg")

probs = model1.predict(img)
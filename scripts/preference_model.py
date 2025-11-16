from tensorflow.keras import models
from tensorflow.keras.models import load_model

model1 = load_model("food_classifier.h5")

embedding_model = models.Model(
    inputs=model1.input,
    outputs=model1.layers[-3].output
)
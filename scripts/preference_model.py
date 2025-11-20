import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.models import load_model, Model
import numpy as np
import os



IMG_SIZE = (224, 224)



model1 = load_model("food_classifier.h5")

# Call the model once on a dummy image to define inputs
dummy = np.zeros((1, 224, 224, 3))
model1(dummy)

embedding_model = Model(
    inputs=model1.layers[0].input,
    outputs=model1.layers[-3].output  # Dense(128) layer before Dropout
)



np.save("embeddings.npy", X)
np.save("labels.npy", Y)

model2 = models.Sequential([
    layers.Input(shape=(128,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # probability friend likes it
])

model2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model2.fit(
    X, Y,
    epochs=20,
    batch_size=32,
    validation_split=0.2,  # 20% of data used for validation
    shuffle=True
)

img = load_image("hamburger_test.jpg")

embedding = embedding_model.predict(img)

probs = model2.predict(embedding)[0][0]

print("Probability friend likes it:", probs)


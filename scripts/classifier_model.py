import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers, applications
from tensorflow.keras.applications import EfficientNetB0
import json
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.applications.efficientnet import preprocess_input
import os
import numpy as np

# Parameters
IMG_SIZE = (224, 224)  # standard for CNNs / transfer learning
BATCH_SIZE = 32
NUM_CLASSES = 22

food_ratings = {"caesar_salad": {"rating": 7, "vegetarian": True}, "cheesecake": {"rating": 4, "vegetarian": True},
               "french_fries": {"rating": 9, "vegetarian": True}, "fried_rice": {"rating": 4, "vegetarian": True},
               "garlic_bread": {"rating": 7, "vegetarian": True},
               "grilled_cheese_sandwich": {"rating": 7, "vegetarian": True}, "hamburger": {"rating": 0, "vegetarian": False},
               "hot_dog": {"rating": 0, "vegetarian": False}, "ice_cream": {"rating": 8, "vegetarian": True}, "macaroni_and_cheese": {"rating": 7, "vegetarian": True},
               "miso_soup": {"rating": 6, "vegetarian": True}, "pizza": {"rating": 7, "vegetarian": True}, "pulled_pork_sandwich": {"rating": 0, "vegetarian": False},
               "sushi": {"rating": 0, "vegetarian": False}, "waffles": {"rating": 7, "vegetarian": True},
               "apple_pie": {"rating": 5, "vegetarian": True}, "breakfast_burrito": {"rating": 10, "vegetarian": True}, "guacamole": {"rating": 2, "vegetarian": True}, 
               "ramen": {"rating": 4, "vegetarian": True}, "tiramisu": {"rating": 4, "vegetarian": True}}

def load_image(path):
    img = tf.keras.utils.load_img(path, target_size=IMG_SIZE)
    img = tf.keras.utils.img_to_array(img)
    img = preprocess_input(img)
    return np.expand_dims(img, axis=0)


# train path
train_path =  "C:/Users/geeta/Github Projects/Food_preference/dataset/train"
test_path = "C:/Users/geeta/Github Projects/Food_preference/dataset/test"


def binary_dataset(train_path, food_ratings):
    class_images = []
    labels = []

    # inserting embeddings and labels into a list
    for food_name, info in food_ratings.items():
        
        image_folder = os.path.join(train_path, food_name)
        # for labels
        label = 1 if info["rating"] >= 6 else 0
        print(food_name)

        for image in os.listdir(image_folder):
            image_path = os.path.join(image_folder, image)
            image_array = load_image(image_path)
            class_images.append(image_array[0])
            labels.append(label)

    X = np.array(class_images)
    Y = np.array(labels)
    return X, Y
    


def normalize(image, label):
    image = preprocess_input(image)
    return image, label

def preparing_data():
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "../dataset/train",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='int'  # integer labels
    )

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        "../dataset/test",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='int'
    )

    # Normalize pixel values to be between 0 and 1
    train_ds = train_ds.map(normalize)
    test_ds = test_ds.map(normalize)
    return [train_ds, test_ds]


def model_setup():
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.2),
    ])
    base_model = EfficientNetB0(input_shape=(224,224,3),
                            include_top=False,
                            weights='imagenet')

    base_model.trainable = False  # Freeze all layers

    # design the architecture of the model
    model = models.Sequential([
        data_augmentation,
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

def compile(model):
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def train_and_test(X_train, Y_train, X_test, Y_test, model):
    checkpoint_cb = ModelCheckpoint("best_model.keras", save_best_only=True)
    early_stop_cb = EarlyStopping(patience=3, restore_best_weights=True)
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_test, Y_test),
        epochs=20,
        batch_size=BATCH_SIZE,
        shuffle=True,
        callbacks=[checkpoint_cb, early_stop_cb]
    )
    return history, model

def evaluation(X_test, Y_test, model):
    test_loss, test_acc = model.evaluate(X_test, Y_test, batch_size=32)
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}")

# Predict on a new image
def predict_food(model, img_path):
    img = load_image(img_path)
    prob = model.predict(img)[0][0]
    return prob



def main():
    # Prepare datasets
    # X_train, Y_train = binary_dataset(train_path, food_ratings)
    # X_test, Y_test = binary_dataset(test_path, food_ratings)
    # model = model_setup()
    # model = compile(model)
    # history, model = train_and_test(X_train, Y_train, X_test, Y_test, model)
    # evaluation(X_test, Y_test, model)
    # with open("history.json" , "a") as file:
    #     json.dump(history.history, file)
    # model.save("food_classifier.h5")
    model = load_model('food_classifier.h5')
    probability = predict_food(model, "caesar_salad_test.jpg")
    print("Probability friend likes it:", probability)

if __name__ == "__main__":
    main()

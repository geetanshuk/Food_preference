import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers, applications
from tensorflow.keras.applications import EfficientNetB0
import json
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.applications.efficientnet import preprocess_input

food_ratings = {"ceaser_salad": {"rating": 7, "vegetarian": True}, "cheesecake": {"rating": 4, "vegetarian": True},
               "french_fries": {"rating": 9, "vegetarian": True}, "fried_rice": {"rating": 4, "vegetarian": True},
               "french_toast": {"rating": 6, "vegetarian": True}, "garlic_bread": {"rating": 7, "vegetarian": True},
               "grilled_cheese_sandwich": {"rating": 7, "vegetarian": True}, "hamburger": {"rating": 0, "vegetarian": False},
               "hot_dog": {"rating": 0, "vegetarian": False}, "ice_cream": {"rating": 8, "vegetarian": True}, "macaroni_and_cheese": {"rating": 7, "vegetarian": True},
               "miso_soup": {"rating": 6, "vegetarian": True}, "pizza": {"rating": 7, "vegetarian": True}, "pulled_pork_sandwich": {"rating": 0, "vegetarian": False},
               "sushi": {"rating": 0, "vegetarian": False}, "tacos": {"rating": 8, "vegetarian": True}, "waffles": {"rating": 7, "vegetarian": True},
               "apple_pie": {"rating": 5, "vegetarian": True}, "breakfast_burrito": {"rating": 10, "vegetarian": True}, "guacamole": {"rating": 2, "vegetarian": True}, 
               "ramen": {"rating": 4, "vegetarian": True}, "tiramisu": {"rating": 4, "vegetarian": True}}

# Parameters
IMG_SIZE = (224, 224)  # standard for CNNs / transfer learning
BATCH_SIZE = 32
NUM_CLASSES = 22

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
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    return model

def compile(model):
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def train_and_test(train_ds, test_ds, model):
    checkpoint_cb = ModelCheckpoint("best_model.h5", save_best_only=True)
    early_stop_cb = EarlyStopping(patience=3, restore_best_weights=True)
    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=15,
        callbacks=[checkpoint_cb, early_stop_cb]
    )
    return history, model

def evaluation(test_ds, model):
    test_loss, test_acc = model.evaluate(test_ds)
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}")



def main():
    train_ds, test_ds = preparing_data()
    model = model_setup()
    model = compile(model)
    history, model = train_and_test(train_ds, test_ds, model)
    evaluation(test_ds, model)
    with open("history.json" , "a") as file:
        json.dump(history.history, file)
    model.save("food_classifier.h5")

if __name__ == "__main__":
    main()

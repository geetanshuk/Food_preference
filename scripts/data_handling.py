import shutil
import os

# folder where you download the dataset
source_folder = "C:/Users/geeta/Downloads/archive/food-101/food-101/images"
# folder where the training data goes
train_folder = "C:/Users/geeta/Github Projects/Food_preference/dataset/train"
# folder where the testing data goes
test_folder = "C:/Users/geeta/Github Projects/Food_preference/dataset/test"


for class_folder in os.listdir(source_folder):
    # go through the source folder to find the food classes
    class_path = os.path.join(source_folder, class_folder)
    print(class_path)
    
    if os.path.isdir(class_path):
        images = os.listdir(class_path)
        count = 0

        # adds training and testing folders into the dataset
        train_class = os.path.join(train_folder, class_folder)
        test_class = os.path.join(test_folder, class_folder)
        os.makedirs(train_class, exist_ok=True)
        os.makedirs(test_class, exist_ok=True)

        # for each image, based on the count, it moves the images
        # from source into the dataset/train and dataset/test
        for image in images:
            image_path = os.path.join(class_path, image)
            
            if count < 750:
                shutil.move(image_path, train_class)
                count += 1
            else:
                shutil.move(image_path, test_class)
        

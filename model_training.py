food_likes = {"ceaser_salad", "cheesecake",
               "fried_rice",
               "macaroni_and_cheese", "pizza", "pulled_pork_sandwich", 
               "ramen"}

# need to find the archive dataset
# 7 likes -> 5 in train, 2 in test
# 8 dislikes -> 6 in train, 2 in test

import tensorflow as tf
print("GPUs available:", tf.config.list_physical_devices('GPU'))
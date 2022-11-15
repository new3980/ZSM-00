import tensorflow as tf
import matplotlib.image as img
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import cv2
import pathlib
from PIL import Image
from io import BytesIO
import requests

# Load 54 classes model
model54 = tf.keras.models.load_model('/home/makima/change-this-later/test/best_model_54class.hdf5')

food_list = np.sort(['bibimbap','caesar_salad', 'cheesecake','chicken_curry','chicken_wings','chocolate_cake','club_sandwich','crab_cakes','creme_brulee','cup_cakes',\
             'donuts','dumplings','edamame','eggs_benedict','filet_mignon','fish_and_chips','foie_gras','french_fries','fried_rice','frozen_yogurt','garlic_bread',\
             'grilled_cheese_sandwich','grilled_salmon','gyoza','hamburger','hot_and_sour_soup','hot_dog','ice_cream','lasagna','lobster_bisque','macaroni_and_cheese',\
             'macarons','miso_soup','mussels','omelette','onion_rings','oysters','pad_thai','pancakes','panna_cotta','peking_duck','pho','pizza','ramen','steak',\
             'risotto','sashimi','scallops','spaghetti_bolognese','spaghetti_carbonara','sushi','takoyaki','tiramisu','waffles'])

classes = list(food_list)
#print(classes[0])

# def internet_get(url):
#     response = requests.get(url)
#     img = Image.open(BytesIO(response.content))
#     input = img
#     # Resize img to proper for feed model.
#     img = img.resize((299,299))
#     # Convert img to numpy array,rescale it,expand dims and check vertically.
#     x = tf.keras.preprocessing.image.img_to_array(img)
#     x = x / 255.0 
#     x = np.expand_dims(x,axis = 0)
#     img_tensor = np.vstack([x])
#     return input,img_tensor

def local_get(path):
    img = Image.open(path)
    input = img
    # Resize img to proper for feed model.
    img = img.resize((299,299))
    # Convert img to numpy array,rescale it,expand dims and check vertically.
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = x / 255.0 
    x = np.expand_dims(x,axis = 0)
    img_tensor = np.vstack([x])
    return input, img_tensor

def predict_result(path):
    original_img, img_tensor = local_get(path)
    pred = model54.predict(img_tensor)
    classes = list(food_list)
    class_predicted = classes[np.argmax(pred)]
    percent = np.max(pred)

    # plt.xticks([])
    # plt.yticks([])
    # plt.imshow(original_img)
    # plt.title(class_predicted)

    return percent, class_predicted

confidence, label = predict_result('/home/makima/change-this-later/54check/8.jpg')
print(confidence)
print(label)
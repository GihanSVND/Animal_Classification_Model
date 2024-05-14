import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
#import os
from tensorflow import keras

model = keras.models.load_model('/Users/gihansavinda/Desktop/Projects/Machine_Leraning/Animal_identify/model.keras')

class_names = ['elephant', 'none', 'peacock', 'wildboar']

image = cv2.imread("/Users/gihansavinda/Desktop/ZYK5FN3N8DUS.jpg")

IMAGE_SIZE = (128,128)
resized_image = tf.image.resize(image,IMAGE_SIZE)
scaled_image = resized_image/255

predictions = model.predict(np.expand_dims(scaled_image, axis=0))

predicted_class_index = np.argmax(predictions)

detected_animal = class_names[predicted_class_index]
print("Detected animal:", detected_animal)


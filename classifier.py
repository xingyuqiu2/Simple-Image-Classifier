import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import models
import os

dim = (32, 32)

# find all images in pictures folder and store their path
path_images = []
path = './pictures'
valid_images = ['.jpg', '.gif', '.png', '.tga']
for f in os.listdir(path):
    ext = os.path.splitext(f)[1]
    if ext.lower() not in valid_images:
        continue
    path_images.append(os.path.join(path, f))

# define label names
class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# load model
model = models.load_model('image_classifier.model')

# for all image path, predict the label of image
for i, path in enumerate(path_images):
    img = cv2.imread(path)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    prediction = model.predict(np.array([img]) / 255)
    index = np.argmax(prediction)
    print(f'Prediction is {class_names[index]} with probability = {prediction[0][index]}')

    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)
    plt.title(f'Prediction: {class_names[index]}')
    plt.xlabel(f'probability: {prediction[0][index]}')
    plt.show()

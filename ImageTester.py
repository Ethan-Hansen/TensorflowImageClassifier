import numpy as np
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
import matplotlib.pyplot as plt
import PIL
from PIL import Image
import itertools
from keras.preprocessing import image

def create_model():
    model = Sequential([
        Conv2D(28, (3, 3), activation='relu', input_shape = (28, 28, 3)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        BatchNormalization(),
        Dropout(.1),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', input_shape = (28, 28, 3)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        BatchNormalization(),
        Dropout(.1),
        BatchNormalization(),
        Conv2D(336, (3, 3), activation='relu', input_shape = (28, 28, 3)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        BatchNormalization(),
        Dropout(.1),
        Flatten(),
        Dense(2, activation='sigmoid'),
        ])
    return model

new_model = create_model()

#Replace this path with the one to your h5 file
new_model.load_weights('D:\School\Senior Design\model.h5')

img_path = input("Please Input Path of the Image: ")
img = image.load_img( img_path, target_size = (28, 28))
img = image.img_to_array(img)
img = np.expand_dims(img, axis = 0)

pred = new_model.predict(img)
guess = np.argmax(pred, axis=1)
if guess[0] == 0:
    print("------------")
    print("Is the Image")
    print("------------")
else:
    print("------------")
    print("Is NOT the Image")
    print("------------")
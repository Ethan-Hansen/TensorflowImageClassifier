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
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
import h5py

train_path = 'D:\School\Senior Design\TestImages'
test_path = 'D:\School\Senior Design\ActualTestImage'

train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size=(28, 28), batch_size=10)
test_batches = ImageDataGenerator().flow_from_directory(test_path, target_size=(28, 28), batch_size=10)
#train_x, valid_x = train_test_split(train_batches, shuffle=True) 

#Sets up model, input_shape has the dimensions of image, channel 3 = rgb, 10 number of classifcations 
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

model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit_generator(train_batches, steps_per_epoch=1218, epochs=3, verbose=2, shuffle=True)

model.summary()

model.save_weights('D:\School\Senior Design\model.h5')

print(test_batches.class_indices)
test_imgs = next(test_batches)
pred = model.predict(test_imgs[0], verbose=0)
print(np.argmax(test_imgs[1], axis=1))
print(np.argmax(pred, axis=1))
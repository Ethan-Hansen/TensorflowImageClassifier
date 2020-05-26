import numpy as np
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
import matplotlib.pyplot as plt
import PIL
from PIL import Image
import itertools

train_path = 'C:\\Users\\hansn\\Numbers\\train'
valid_path = 'C:\\Users\\hansn\\Numbers\\valid'
test_path = 'C:\\Users\\hansn\\Numbers\\test'

train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size=(28, 28), batch_size=10)
valid_batches = ImageDataGenerator().flow_from_directory(valid_path, target_size=(28, 28), batch_size=5)
test_batches = ImageDataGenerator().flow_from_directory(test_path, target_size=(28, 28), batch_size=10)

#Sets up model, input_shape has the dimensions of image, channel 3 = rgb, 10 number of classifcations 
model = Sequential([
        Conv2D(140, (3, 3), activation='relu', input_shape=(28, 28, 3)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        BatchNormalization(),
        Dropout(rate=.1),
        Conv2D(280, (3, 3), activation='relu', input_shape=(28, 28, 3)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        BatchNormalization(),
        Dropout(rate=.2),
        Conv2D(560, (3, 3), activation='relu', input_shape=(28, 28, 3)),
        BatchNormalization(),
        Dropout(rate=.3),
        Flatten(),
        Dropout(rate=.4),
        Dense(200, activation='relu'),
        BatchNormalization(),
        Dense(10, activation='softmax'),
        ])

model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit_generator(train_batches, steps_per_epoch=10,
                    validation_data=valid_batches, validation_steps=10, epochs=30, verbose=2, shuffle=True)

model.summary()

print(test_batches.class_indices)
test_imgs = next(test_batches)
pred = model.predict(test_imgs[0], verbose=0)
print(np.argmax(test_imgs[1], axis=1))
print(np.argmax(pred, axis=1))
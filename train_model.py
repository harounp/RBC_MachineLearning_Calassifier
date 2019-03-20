import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import cv2
import random
#import pickle
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard

NAME = "Burr_Normal_Optimization{}".format(int(time.time()))
tensorboard = TensorBoard(log_dir = 'logs/{}'.format(NAME))

directory = ("data")
categories = ["burr", "normal"]

IMSIZE= 100
training_data = []

def create_dataset():

    for category in categories:
        path = os.path.join(directory,category)
        class_num = categories.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                resize_array = cv2.resize(img_array, (IMSIZE,IMSIZE))
                training_data.append([resize_array, class_num])
            except Exception as e:
                pass

create_dataset()
random.shuffle(training_data)

X =[]
y=[]

for features, label in training_data:
    X.append(features)
    y.append(label)

X= np.array(X).reshape(-1, IMSIZE, IMSIZE, 1)

# pickle_out = open('X.pickle' , "wb")
# pickle.dump (X,pickle_out)
# pickle_out.close()
#
# pickle_out = open('y.pickle' , "wb")
# pickle.dump (y,pickle_out)
# pickle_out.close()

X = X/255.0 # normalizing the images

# Normalize pixel values to be between 0 and 1

model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(IMSIZE, IMSIZE, 1)))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())
model.add(Dense(64, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss = "binary_crossentropy",
            optimizer = "adam",
            metrics = ["accuracy"])

model.fit(X , y , batch_size = 64, epochs = 3, validation_split = 0.2, callbacks = [tensorboard]) # batch_size = # of samples at a time (usually 20 - 200)
model.save('Burr_Normal.model')

#IMPORTS
import os
os.environ["CUDA_VISIBLE_DEVICES"]="7"

import tensorflow as tf

from tensorflow.keras import datasets, layers, models
#from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import csv

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0
validate_images, validate_labels= test_images[0:2000], test_labels[0:2000]
test_images, test_labels= test_images[2000:], test_labels[2000:]
#train_images, train_labels = train_images[0:10000], train_labels[0:10000]
#test_images, test_labels = test_images[0:200], test_labels[0:200]

#UNCOMMENT THE FOLLOWING IF GRAYSCALE TEST IMAGES DESIRED
# from skimage.color import rgb2gray, gray2rgb
# test_images = rgb2gray(test_images) 
##This converts to grayscale
# test_images= gray2rgb(test_images)
##This converts the grayscale image to have 3 channels so it can work with our CNN

#UNCOMMENT THE FOLLOWING IF BLURRED TEST IMAGES DESIRED
#for i in range(len(test_images)):
#    test_images[i]=cv2.GaussianBlur(test_images[i], (11, 11), 7)
#print(test_images[0][0].size)
#plt.imshow(test_images[0],cmap="gray")

#Testing on diff datasets
result={}
paths=['/om/user/aunell/data/0.05biomodel','/om/user/aunell/data/0.05model', '/om/user/aunell/data/0biomodel','/om/user/aunell/data/0model']
for path in paths:
    model = tf.keras.models.load_model(path)
    weights= model.get_weights()
    model=None
    visible = layers.Input(shape=(32,32,3))
    conv1 = layers.Conv2D(32, kernel_size=(3,3), activation='relu')(visible)
    noise1 = layers.GaussianNoise(0)(conv1, training=True)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(noise1)

    conv2 = layers.Conv2D(64, kernel_size=(3,3), activation='relu')(pool1)
    noise2 = layers.GaussianNoise(0)(conv2, training=True)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(noise2)

    conv3 = layers.Conv2D(64, kernel_size=(3,3), activation='relu')(pool2)
    noise1 = layers.GaussianNoise(0)(conv3, training=True)
    flat = layers.Flatten()(noise1)
    hidden1 = layers.Dense(64, activation='relu')(flat)
    output = layers.Dense(10)(hidden1)

    model1 = Model(inputs=visible, outputs=output)
    model1.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
    model1.set_weights(weights)
    test_loss, test_acc = model1.evaluate(test_images, test_labels, verbose=0)
    result[path[21:]]=test_acc
print(result)
  
#PLOT
x=range(0,51)
print(len(parameters['0bio']))
plt.plot(x, parameters[0],label='0 Noise in Training')
plt.plot(x, parameters[.05], label='.05 Noise in Training')
plt.plot(x, parameters['0bio'], label='Clear to Noise')
plt.plot(x, parameters['0.05bio'], label='Noise to Clear')
plt.xlabel('Noise (times a factor of 1000)')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('/om/user/aunell/data/TestNoise/results/biomimeticTesting.png')
plt.show()

import os
os.environ["CUDA_VISIBLE_DEVICES"]="7"

import tensorflow as tf

from tensorflow.keras import datasets, layers, models, backend, Model, callbacks
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

#CHANGING TRAINING IMAGES
from skimage.color import rgb2gray, gray2rgb
from skimage.filters import difference_of_gaussians
import cv2
from sklearn.utils import shuffle
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']


train_imagesClear=np.copy(train_images)
for i in range(len(train_images)):
#     gauss = np.random.normal(0,.1,(32,32,3))
#     gauss = gauss.reshape(32,32,3)
#     image=(train_imagesBlur[i]+gauss)
#     image=np.clip(image, 0, 1)
    image=train_imagesClear[i]
    image= cv2.GaussianBlur(image,(3,3),0)
    image= rgb2gray(image)
    image= gray2rgb(image)
    train_imagesClear[i]=image

train_imagesNoisy=np.copy(train_images)
# for i in range(len(train_images)):
#     gauss = np.random.normal(0,.1,(32,32,3))
#     gauss = gauss.reshape(32,32,3)
#     image=(train_imagesNoisy[i]+gauss)
#     image=np.clip(image, 0, 1)
#     train_imagesNoisy[i]=image

train_imagesGray=np.copy(train_images)
for i in range(len(train_images)):
    image=train_imagesGray[i]
    image= rgb2gray(image)
    image= gray2rgb(image)
    train_imagesGray[i]=image

train_imagesBlur=np.copy(train_images)
for i in range(len(train_images)):
    image=train_imagesBlur[i]
    image= cv2.GaussianBlur(image,(3,3),0)
    train_imagesBlur[i]=image
    

train_imagesCV=np.copy(train_images)
train_labelsCV=np.copy(train_labels)
train_imagesCV=np.concatenate((train_imagesCV, train_imagesClear))
train_labelsCV=np.concatenate((train_labelsCV, train_labels))
train_imagesCV=np.concatenate((train_imagesCV, train_imagesBlur))
train_labelsCV=np.concatenate((train_labelsCV, train_labels))
train_imagesCV=np.concatenate((train_imagesCV, train_imagesGray))
train_labelsCV=np.concatenate((train_labelsCV, train_labels))


test_imagesNoise=np.copy(test_images)
for i in range(len(test_images)):
    gauss = np.random.normal(0,.1,(32,32,3))
    gauss = gauss.reshape(32,32,3)
    image=(test_imagesNoise[i]+gauss)
    image=np.clip(image, 0, 1)
    test_imagesNoise[i]=image
    
test_imagesGray=np.copy(test_images)
for i in range(len(test_images)):
    image=test_imagesGray[i]
    image= rgb2gray(image)
    image= gray2rgb(image)
    test_imagesGray[i]=image

test_imagesBlur=np.copy(test_images)
for i in range(len(test_images)):
    image=test_imagesBlur[i]
    image= cv2.GaussianBlur(image,(3,3),0)
    test_imagesBlur[i]=image

test_imagesNoiseAndGray=np.copy(test_images)
for i in range(len(test_images)):
    gauss = np.random.normal(0,.1,(32,32,3))
    gauss = gauss.reshape(32,32,3)
    image=(test_imagesNoiseAndGray[i]+gauss)
    image=np.clip(image, 0, 1)
    image= rgb2gray(image)
    image= gray2rgb(image)
    test_imagesNoiseAndGray[i]=image
    
test_imagesNoiseAndBlur=np.copy(test_images)
for i in range(len(test_images)):
    gauss = np.random.normal(0,.1,(32,32,3))
    gauss = gauss.reshape(32,32,3)
    image=(test_imagesNoiseAndBlur[i]+gauss)
    image=np.clip(image, 0, 1)
    image= cv2.GaussianBlur(image,(3,3),0)
    test_imagesNoiseAndBlur[i]=image

test_imagesBlurAndGray=np.copy(test_images)
for i in range(len(test_images)):
    image=test_imagesBlurAndGray[i]
    image= rgb2gray(image)
    image= gray2rgb(image)
    image= cv2.GaussianBlur(image,(3,3),0)
    test_imagesBlurAndGray[i]=image

test_imagesBlurAndGrayAndNoise=np.copy(test_images)
for i in range(len(test_images)):
    gauss = np.random.normal(0,.1,(32,32,3))
    gauss = gauss.reshape(32,32,3)
    image=(test_imagesBlurAndGrayAndNoise[i]+gauss)
    image=np.clip(image, 0, 1)
    image= cv2.GaussianBlur(image,(3,3),0)
    image= rgb2gray(image)
    image= gray2rgb(image)
    test_imagesBlurAndGrayAndNoise[i]=image
print('3')  
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_imagesCV[-1-i])
    # The CIFAR labels happen to be arrays, 
    # which is why you need the extra index
    plt.xlabel(class_names[train_labelsCV[-1-i][0]])
plt.show()


#Activations
parameters={}
#l==0 when using clear images in testing
for l in range(1):
    #NoisyImages
    z=0
    training=[0, .1]
    for m in training:
      model=None
      #parameters[m]=None
      fintest=[]
      for i in range(1,4):
          print('i', i)
          noise_dict={1: 0, 2: m, 3: 0}
          if i !=1:
            weights0=model.get_weights()
            del(model)
            tf.compat.v1.reset_default_graph()
          print('starting model')
          model = models.Sequential()
          model.add(layers.Conv2D(32, (3, 3), input_shape=(32,32,3)))
          model.add(layers.Activation('relu'))
          model.add(layers.GaussianNoise(noise_dict[1]))
          model.add(layers.MaxPooling2D((2, 2)))

          model.add(layers.Conv2D(64, (3, 3)))
          model.add(layers.Activation('relu'))
          model.add(layers.GaussianNoise(noise_dict[2]))
          model.add(layers.MaxPooling2D((2, 2)))

          model.add(layers.Conv2D(64, (3, 3)))
          model.add(layers.Activation('relu'))
          model.add(layers.GaussianNoise(noise_dict[3]))
          model.add(layers.Flatten())
          model.add(layers.Dense(64, activation='relu'))
          model.add(layers.Dense(10))
          model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
          if i != 1:
              model.set_weights(weights0)
          history = model.fit(train_images, train_labels, epochs=3, 
                            validation_data=(validate_images, validate_labels))
          print('ending model')

      print('test model')
      model1=model
      if m==0:
          name='clearmodel'
      else:
        name='noisemodel'
      path='/om/user/aunell/data/ActivationTesting/'+name
      model.save(path)
      weights1=model1.get_weights()
      fintest_trial=[]

       #biomimetic training
      if m==0:
            #CleartoNoise
            weights0=model.get_weights()
            model=None
            noise_dict={1: 0, 2: .1, 3: 0}
            print('starting biomodel')
            model = models.Sequential()
            model.add(layers.Conv2D(32, (3, 3), input_shape=(32,32,3)))
            model.add(layers.Activation('relu'))
            model.add(layers.GaussianNoise(noise_dict[1]))
            model.add(layers.MaxPooling2D((2, 2)))

            model.add(layers.Conv2D(64, (3, 3)))
            model.add(layers.Activation('relu'))
            model.add(layers.GaussianNoise(noise_dict[2]))
            model.add(layers.MaxPooling2D((2, 2)))

            model.add(layers.Conv2D(64, (3, 3)))
            model.add(layers.Activation('relu'))
            model.add(layers.GaussianNoise(noise_dict[3]))
            model.add(layers.Flatten())
            model.add(layers.Dense(64, activation='relu'))
            model.add(layers.Dense(10))
            model.compile(optimizer='adam',
                          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                          metrics=['accuracy'])
            model.set_weights(weights0)

            callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, mode= 'min')
            history = model.fit(train_images, train_labels,
                                validation_data=(validate_images, validate_labels), epochs=50,
                                 callbacks=[callback])
            modelName='CleartoNoisebio'
            print('ending biomodel')
      else:
        #NoisetoClear
            weights0=model.get_weights()
            model=None
            noise_dict={1: 0, 2: 0, 3: 0}
            print('starting biomodel')
            model = models.Sequential()
            model.add(layers.Conv2D(32, (3, 3), input_shape=(32,32,3)))
            model.add(layers.Activation('relu'))
            model.add(layers.GaussianNoise(noise_dict[1]))
            model.add(layers.MaxPooling2D((2, 2)))

            model.add(layers.Conv2D(64, (3, 3)))
            model.add(layers.Activation('relu'))
            model.add(layers.GaussianNoise(noise_dict[2]))
            model.add(layers.MaxPooling2D((2, 2)))

            model.add(layers.Conv2D(64, (3, 3)))
            model.add(layers.Activation('relu'))
            model.add(layers.GaussianNoise(noise_dict[3]))
            model.add(layers.Flatten())
            model.add(layers.Dense(64, activation='relu'))
            model.add(layers.Dense(10))
            model.compile(optimizer='adam',
                          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                          metrics=['accuracy'])
            model.set_weights(weights0)

            callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, mode= 'min')
            history = model.fit(train_images, train_labels,
                                validation_data=(validate_images, validate_labels), epochs=50,
                                 callbacks=[callback])
            modelName='NoisetoClearbio'
            print('ending biomodel')
      print('test model')
      fintest_trial=[]
      weights=model.get_weights()
      path='/om/user/aunell/data/ActivationTesting/'+modelName
      model.save(path)
#Testing on diff regimines (generated by noisy inputs) and diff datasets (noisy inputs)
# def display_activation(activations, col_size, row_size, act_index, title): 
#     activation = activations[act_index]
#     activation_index=0
#     fig, ax = plt.subplots(row_size, col_size, figsize=(row_size*4,col_size*1))
#     plt.title(title)
#     for row in range(0,row_size):
#         for col in range(0,col_size):
#             ax[row][col].imshow(activation[:,activation_index], cmap='gray')
#             activation_index += 1
            
testImages= [test_images, test_imagesBlur]
noise=[0,.1]
paths=['/om/user/aunell/data/ActivationTesting/clearmodel','/om/user/aunell/data/ActivationTesting/noisemodel', 
       '/om/user/aunell/data/ActivationTesting/CleartoNoisebio','/om/user/aunell/data/ActivationTesting/NoisetoClearbio']
for path in paths:
    for i in noise:
        model = tf.keras.models.load_model(path)
        weights= model.get_weights()
        model=None
        visible = layers.Input(shape=(32,32,3))
        conv1 = layers.Conv2D(32, kernel_size=(3,3), activation='relu')(visible)
        noise1 = layers.GaussianNoise(0)(conv1, training=True)
        pool1 = layers.MaxPooling2D(pool_size=(2, 2))(noise1)

        conv2 = layers.Conv2D(64, kernel_size=(3,3), activation='relu')(pool1)
        noise2 = layers.GaussianNoise(i)(conv2, training=True)
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


        img_tensor = np.expand_dims(test_imagesBlur[0], axis=0)
        layer_outputs = [layer.output for layer in model1.layers]
        activation_model = Model(inputs=model1.input, outputs=layer_outputs)
        activations = activation_model.predict(test_images)
        print(activations[4].shape)
        print(activations[3].shape)
        #SAVE ACTIVATION HERE
        title= path[39:]
        if i!=0:
            path2='/om/user/aunell/data/ActivationTesting/pickledFiles/NoisyTest/'+str(title)+'.pkl'
        else:
            path2='/om/user/aunell/data/ActivationTesting/pickledFiles/ClearTest/'+str(title)+'.pkl'
        pickle.dump(activations, open(path2, 'wb'))
    

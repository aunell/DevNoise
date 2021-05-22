#In Network Basic CIFAR
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

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0
validate_images, validate_labels= test_images[0:2000], test_labels[0:2000]
test_images, test_labels= test_images[2000:], test_labels[2000:]
#train_images, train_labels = train_images[0:10000], train_labels[0:10000]
#test_images, test_labels = test_images[0:200], test_labels[0:200]

#CHANGING TEST IMAGES
from skimage.color import rgb2gray, gray2rgb
from skimage.filters import difference_of_gaussians
import cv2
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']


test_imagesNoise=np.copy(test_images)
for i in range(len(test_images)):
    gauss = np.random.normal(0,.2,(32,32,3))
    gauss = gauss.reshape(32,32,3)
    image=(test_imagesNoise[i]+gauss)
    image=np.clip(image, 0, 1)
    test_imagesNoise[i]=image
    
train_imagesClear=np.copy(train_images)
for i in range(len(train_images)):
    image=train_imagesClear[i]
    image= cv2.GaussianBlur(image,(3,3),0)
    image= rgb2gray(image)
    image= gray2rgb(image)
    train_imagesClear[i]=image

plt.figure(figsize=(10,10))
print(type(test_images[0]))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[i], cmap='gray')
    # The CIFAR labels happen to be arrays, 
    # which is why you need the extra index
    plt.xlabel(class_names[test_labels[i][0]])
plt.show()


noise=[0, 0.1]
parameters={}
for m in noise:
  model=None
  parameters[m]=None
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
      if m==0:
        history = model.fit(train_imagesClear, train_labels, epochs=3, 
                        validation_data=(validate_images, validate_labels))
      else:
        history = model.fit(train_images, train_labels, epochs=3, 
                        validation_data=(validate_images, validate_labels))
      print('ending model')
        
  print('test model')
  model1=model
  name=str(m)+'model'
  path='/om/user/aunell/data/'+name
  model.save(path)
  weights1=model1.get_weights()
  fintest_trial=[]
  for n in range(0,1):
            del(model1)
            tf.compat.v1.reset_default_graph()
            n=n/1000
            
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
            model1.set_weights(weights1)
            test_loss, test_acc = model1.evaluate(test_imagesNoise,  test_labels, verbose=0)
            fintest_trial.append(test_acc)
  parameters[m]=fintest_trial
            
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
        
        print('ending biomodel')
        name=str(m)+'C2N'
        modelName=str(m)+'C2N'
        path='/om/user/aunell/data/'+modelName
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
        history = model.fit(train_imagesClear, train_labels,
                            validation_data=(validate_images, validate_labels), epochs=50,
                             callbacks=[callback])
        print('ending biomodel')
        name=str(m)+'N2C'
        modelName=str(m)+'N2C'
        path='/om/user/aunell/data/'+modelName
          
  print('test model')
  fintest_trial=[]
  weights=model.get_weights()
  model.save(path)
  for n in range(0,1):
            print('n', n)
            del(model)
            tf.compat.v1.reset_default_graph()
            n=n/1000
            
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

            model = Model(inputs=visible, outputs=output)
            model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
            model.set_weights(weights)
            test_loss, test_acc = model.evaluate(test_imagesNoise,  test_labels, verbose=0)
            fintest_trial.append(test_acc)
  parameters[name]=fintest_trial
  
  
#   BAR PLOT

# #BAR PLOT RESULTS
print(parameters)
key_list = list(parameters.keys())
#x=[noisemodel, NoisetoClearbio, clearmodel, CleartoNoisebio]
cleartest=[]
noisetest=[]
data=[]
for i in range(4):
    cleartest.append(parameters[key_list[i]][0])
data.append(cleartest)

ind = np.arange(4) 
width = 0.35       
plt.bar(ind, cleartest, width, label='Noisy Test Images')

plt.ylabel('accuracy')
plt.title('.2 STD for Network Noise, .3 for Input, FashionMNIST')

plt.xticks(ind + width / 2, ('Clear', 'C2N', 'Noise', 'N2C'))
plt.legend(loc='lower right')
plt.show()

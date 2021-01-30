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

#SHOW CLASS EXAMPLES
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    # The CIFAR labels happen to be arrays, 
    # which is why you need the extra index
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()

#TRAINING
noise=[0, 0.05]
parameters={}
for m in noise:
  model=None
  parameters[m]=None
  fintest=[]
  val_acc_vals=[]
  for i in range(1,4):
      print('i', i)
      noise_dict={1: m, 2: m, 3: m}
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
  name=str(m)+'model'
  path='/om/user/aunell/data/'+name
  model.save(path)
  weights1=model1.get_weights()
  fintest_trial=[]
  for n in range(0,51):
            print('n', n)
            del(model1)
            tf.compat.v1.reset_default_graph()
            n=n/1000
            
            visible = layers.Input(shape=(32,32,3))
            conv1 = layers.Conv2D(32, kernel_size=(3,3), activation='relu')(visible)
            noise1 = layers.GaussianNoise(n)(conv1, training=True)
            pool1 = layers.MaxPooling2D(pool_size=(2, 2))(noise1)

            conv2 = layers.Conv2D(64, kernel_size=(3,3), activation='relu')(pool1)
            noise2 = layers.GaussianNoise(n)(conv2, training=True)
            pool2 = layers.MaxPooling2D(pool_size=(2, 2))(noise2)

            conv3 = layers.Conv2D(64, kernel_size=(3,3), activation='relu')(pool2)
            noise1 = layers.GaussianNoise(n)(conv3, training=True)
            flat = layers.Flatten()(noise1)
            hidden1 = layers.Dense(64, activation='relu')(flat)
            output = layers.Dense(10)(hidden1)

            model1 = Model(inputs=visible, outputs=output)
            model1.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
            model1.set_weights(weights1)
            test_loss, test_acc = model1.evaluate(test_images,  test_labels, verbose=0)
            fintest_trial.append(test_acc)
  parameters[m]=fintest_trial
            
   #biomimetic training
  if m==0:
        #CleartoNoise
        weights0=model.get_weights()
        model=None
        noise_dict={1: .05, 2: .05, 3: .05}
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
        print('ending biomodel')
          
  print('test model')
  fintest_trial=[]
  weights=model.get_weights()
  name=str(m)+'bio'
  modelName=str(m)+'biomodel'
  path='/om/user/aunell/data/'+modelName
  model.save(path)
  for n in range(0,51):
            print(fintest_trial)
            print('n', n)
            del(model)
            tf.compat.v1.reset_default_graph()
            n=n/1000
            
            visible = layers.Input(shape=(32,32,3))
            conv1 = layers.Conv2D(32, kernel_size=(3,3), activation='relu')(visible)
            noise1 = layers.GaussianNoise(n)(conv1, training=True)
            pool1 = layers.MaxPooling2D(pool_size=(2, 2))(noise1)

            conv2 = layers.Conv2D(64, kernel_size=(3,3), activation='relu')(pool1)
            noise2 = layers.GaussianNoise(n)(conv2, training=True)
            pool2 = layers.MaxPooling2D(pool_size=(2, 2))(noise2)

            conv3 = layers.Conv2D(64, kernel_size=(3,3), activation='relu')(pool2)
            noise1 = layers.GaussianNoise(n)(conv3, training=True)
            flat = layers.Flatten()(noise1)
            hidden1 = layers.Dense(64, activation='relu')(flat)
            output = layers.Dense(10)(hidden1)

            model = Model(inputs=visible, outputs=output)
            model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
            model.set_weights(weights)
            test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=0)
            fintest_trial.append(test_acc)
  parameters[name]=fintest_trial
  
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

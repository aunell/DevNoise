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

#CHANGING TEST IMAGES
from skimage.color import rgb2gray, gray2rgb
from skimage.filters import difference_of_gaussians
import cv2
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
# test_images = rgb2gray(test_imagesPRE)
# test_images= gray2rgb(test_images)
# #test_images=difference_of_gaussians(test_imagesPRE, 10)
# for i in range(len(test_images)):
#     test_images[i]=cv2.GaussianBlur(test_images[i], (11, 11), 7)
# print(test_images[0][0].size)
# plt.imshow(test_images[0],cmap="gray")

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

#Noise in certain layers-- large loop/test

totalNoiseValues=[0, .3, .6, 0.9]
parameters={}
for l in range(1,4):
    for m in totalNoiseValues:
        #noisy
      if l==1:
        noise_dict={1: m, 2: 0, 3: 0}
      elif l==2:
        noise_dict={1: 0, 2: m, 3: 0}
      else:
        noise_dict={1: 0, 2: 0, 3: m}
      model=None
      parameters[m]=None
      fintest=[]
      for i in range(1,4):
          print('i', i)
          #noise_dict={1: 0, 2: 0, 3: m}
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
      noiseModel=model
      if m==0:
        zeroModel=model
        zeroModelWeights=zeroModel.get_weights()
      name=str(m)+'noisymodelLayer'+str(l)
      path='/om/user/aunell/data/'+name
      model.save(path)
      weights1=model1.get_weights()
      fintest_trial=[]
    #testing noisy model
      for n in range(0, 2):
                print('n', n)
                del(model1)
                tf.compat.v1.reset_default_graph()
                if n==0:
                    test_dict={1: 0, 2: 0, 3: 0}
                else:
                    test_dict=noise_dict

                visible = layers.Input(shape=(32,32,3))
                conv1 = layers.Conv2D(32, kernel_size=(3,3), activation='relu')(visible)
                noise1 = layers.GaussianNoise(test_dict[1])(conv1, training=True)
                pool1 = layers.MaxPooling2D(pool_size=(2, 2))(noise1)

                conv2 = layers.Conv2D(64, kernel_size=(3,3), activation='relu')(pool1)
                noise2 = layers.GaussianNoise(test_dict[2])(conv2, training=True)
                pool2 = layers.MaxPooling2D(pool_size=(2, 2))(noise2)

                conv3 = layers.Conv2D(64, kernel_size=(3,3), activation='relu')(pool2)
                noise1 = layers.GaussianNoise(test_dict[3])(conv3, training=True)
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
                parameters[name]=fintest_trial
      #testing clear model
      name=str(m)+'clearmodelLayer'+str(l)
      fintest_trial=[]
      for n in range(0, 2):
                print('n', n)
                del(model1)
                tf.compat.v1.reset_default_graph()
                if n==0:
                    test_dict={1: 0, 2: 0, 3: 0}
                else:
                    test_dict=noise_dict

                visible = layers.Input(shape=(32,32,3))
                conv1 = layers.Conv2D(32, kernel_size=(3,3), activation='relu')(visible)
                noise1 = layers.GaussianNoise(test_dict[1])(conv1, training=True)
                pool1 = layers.MaxPooling2D(pool_size=(2, 2))(noise1)

                conv2 = layers.Conv2D(64, kernel_size=(3,3), activation='relu')(pool1)
                noise2 = layers.GaussianNoise(test_dict[2])(conv2, training=True)
                pool2 = layers.MaxPooling2D(pool_size=(2, 2))(noise2)

                conv3 = layers.Conv2D(64, kernel_size=(3,3), activation='relu')(pool2)
                noise1 = layers.GaussianNoise(test_dict[3])(conv3, training=True)
                flat = layers.Flatten()(noise1)
                hidden1 = layers.Dense(64, activation='relu')(flat)
                output = layers.Dense(10)(hidden1)

                model1 = Model(inputs=visible, outputs=output)
                model1.compile(optimizer='adam',
                          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                          metrics=['accuracy'])
                model1.set_weights(zeroModelWeights)
                test_loss, test_acc = model1.evaluate(test_images,  test_labels, verbose=0)
                fintest_trial.append(test_acc)
                parameters[name]=fintest_trial
      #biomimetic training
      #CleartoNoise
      weights0=zeroModel.get_weights()
      model=None
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
      name=str(m)+'bioC2NLayer'+str(l)
      path='/om/user/aunell/data/'+name
      model.save(path)
      for n in range(0,2):
                print(fintest_trial)
                del(model)
                tf.compat.v1.reset_default_graph()
                
                if n==0:
                    test_dict={1: 0, 2: 0, 3: 0}
                else:
                    test_dict=noise_dict
                
                visible = layers.Input(shape=(32,32,3))
                conv1 = layers.Conv2D(32, kernel_size=(3,3), activation='relu')(visible)
                noise1 = layers.GaussianNoise(test_dict[1])(conv1, training=True)
                pool1 = layers.MaxPooling2D(pool_size=(2, 2))(noise1)

                conv2 = layers.Conv2D(64, kernel_size=(3,3), activation='relu')(pool1)
                noise2 = layers.GaussianNoise(test_dict[2])(conv2, training=True)
                pool2 = layers.MaxPooling2D(pool_size=(2, 2))(noise2)

                conv3 = layers.Conv2D(64, kernel_size=(3,3), activation='relu')(pool2)
                noise1 = layers.GaussianNoise(test_dict[3])(conv3, training=True)
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
      #NoisetoClear
      weights0=noiseModel.get_weights()
      model=None
      print('starting biomodel')
      model = models.Sequential()
      model.add(layers.Conv2D(32, (3, 3), input_shape=(32,32,3)))
      model.add(layers.Activation('relu'))
      model.add(layers.GaussianNoise(0))
      model.add(layers.MaxPooling2D((2, 2)))

      model.add(layers.Conv2D(64, (3, 3)))
      model.add(layers.Activation('relu'))
      model.add(layers.GaussianNoise(0))
      model.add(layers.MaxPooling2D((2, 2)))

      model.add(layers.Conv2D(64, (3, 3)))
      model.add(layers.Activation('relu'))
      model.add(layers.GaussianNoise(0))
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
      name=str(m)+'bioN2CLayer'+str(l)
      path='/om/user/aunell/data/'+name
      model.save(path)
      for n in range(0,2):
                print(fintest_trial)
                del(model)
                tf.compat.v1.reset_default_graph()
                
                if n==0:
                    test_dict={1: 0, 2: 0, 3: 0}
                else:
                    test_dict=noise_dict

                visible = layers.Input(shape=(32,32,3))
                conv1 = layers.Conv2D(32, kernel_size=(3,3), activation='relu')(visible)
                noise1 = layers.GaussianNoise(test_dict[1])(conv1, training=True)
                pool1 = layers.MaxPooling2D(pool_size=(2, 2))(noise1)

                conv2 = layers.Conv2D(64, kernel_size=(3,3), activation='relu')(pool1)
                noise2 = layers.GaussianNoise(test_dict[2])(conv2, training=True)
                pool2 = layers.MaxPooling2D(pool_size=(2, 2))(noise2)

                conv3 = layers.Conv2D(64, kernel_size=(3,3), activation='relu')(pool2)
                noise1 = layers.GaussianNoise(test_dict[3])(conv3, training=True)
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
      
      #Line PLOT
x=[0, .05, .2, .4]
x=totalNoiseValues
key_list = list(parameters.keys())
for key in key_list:
    if parameters[key]==None:
        del parameters[key]
print(parameters)
clearTest={}
noiseTest={}
for key in key_list:
    print(key)
    clearTest[key]=parameters[key][0]
    noiseTest[key]=parameters[key][1]
print(clearTest)
#print(noiseTest)
#convention-- testNetwork
clearClear=[]
clearNoisy=[]
clearC2N=[]
clearN2C=[]
noiseClear=[]
noiseNoisy=[]
noiseC2N=[]
noiseN2C=[]
clearKey=list(clearTest.keys())
noiseKey=list(noiseTest.keys())
for i in range(0,len(clearKey),4):
    noiseNoisy.append(noiseTest[noiseKey[i]])
    clearNoisy.append(clearTest[clearKey[i]])
for i in range(1,len(clearKey),4):
    noiseClear.append(noiseTest[noiseKey[i]])
    clearClear.append(clearTest[clearKey[i]])
for i in range(2,len(clearKey),4):
    noiseC2N.append(noiseTest[noiseKey[i]])
    clearC2N.append(clearTest[clearKey[i]])
for i in range(3,len(clearKey),4):
    noiseN2C.append(noiseTest[noiseKey[i]])
    clearN2C.append(clearTest[clearKey[i]])
#[0:4] for layer 1, [4:8] for layer 2, [8:12] for layer 3
layer=3
if layer==1:
    indice1=0
    indice2=4
elif layer==2:
    indice1=4
    indice2=8
elif layer==3:
    indice1=8
    indice2=12
plt.plot(x, noiseNoisy[indice1:indice2], 'g-', label='noise in training, noise in testing')
plt.plot(x, clearNoisy[indice1:indice2], 'g--', label='noise in training, clear in testing')
plt.plot(x, noiseClear[indice1:indice2],'m-', label='clear in training, noise in testing')
plt.plot(x, clearClear[indice1:indice2],'m--',label='clear in training, clear in testing')
plt.plot(x, noiseC2N[indice1:indice2],'b-', label='C2N training, noise in testing')
plt.plot(x, clearC2N[indice1:indice2],'b--',label='C2N training, clear in testing')
plt.plot(x, noiseN2C[indice1:indice2],'r-', label='N2C training, noise in testing')
plt.plot(x, clearN2C[indice1:indice2],'r--', label='N2C training, clear in testing')
plt.legend(loc='upper center', bbox_to_anchor=(1.4, 1.05),
          ncol=1, fancybox=True, shadow=True)
plt.title('Effect of Noise in 3rd Layer')
plt.xlabel('Gaussian Noise')
plt.xticks(np.arange(0, 1, step=0.3))
plt.ylabel('Accuracy')
plt.savefig('/om/user/aunell/data/Results02.19/ThirdLayerNoise.png')

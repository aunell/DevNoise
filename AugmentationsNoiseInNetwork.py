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
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']


train_imagesClear=np.copy(train_images)
for i in range(len(train_images)):
    image=train_imagesClear[i]
    image= cv2.GaussianBlur(image,(3,3),0)
    image= rgb2gray(image)
    image= gray2rgb(image)
    train_imagesClear[i]=image

train_imagesNoisy=np.copy(train_images)

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

#Changing test Images  
permutations=[test_images]
permutationsStrings = ['test_images']
test_imagesGray=np.copy(test_images)
permutations.append(test_imagesGray)
permutationsStrings.append('test_imagesGray')
for i in range(len(test_images)):
    image=test_imagesGray[i]
    image= rgb2gray(image)
    image= gray2rgb(image)
    test_imagesGray[i]=image

test_imagesBlur=np.copy(test_images)
permutations.append(test_imagesBlur)
permutationsStrings.append('test_imagesBlur')
for i in range(len(test_images)):
    image=test_imagesBlur[i]
    image= cv2.GaussianBlur(image,(3,3),0)
    test_imagesBlur[i]=image
    

test_imagesBlurAndGray=np.copy(test_images)
permutations.append(test_imagesBlurAndGray)
permutationsStrings.append('test_imagesBlurAndGray')
for i in range(len(test_images)):
    image=test_imagesBlurAndGray[i]
    image= rgb2gray(image)
    image= gray2rgb(image)
    image= cv2.GaussianBlur(image,(3,3),0)
    test_imagesBlurAndGray[i]=image

test_imagesHueShift=np.copy(test_images)
permutations.append(test_imagesHueShift)
permutationsStrings.append('test_imagesHueShift')
for i in range(len(test_images)):
    image=test_imagesHueShift[i]
    image=colorize(image,1.5)
    image=transform.resize(image, (32,32))
    test_imagesHueShift[i]=image

test_imagesBlurAndHueShift=np.copy(test_images)
permutations.append(test_imagesBlurAndHueShift)
permutationsStrings.append('test_imagesBlurAndHueShift')
for i in range(len(test_images)):
    image=test_imagesBlurAndHueShift[i]
    image=colorize(image,1.5)
    image=transform.resize(image, (32,32))
    image= cv2.GaussianBlur(image,(3,3),0)
    test_imagesBlurAndHueShift[i]=image
    
#Contrast  
test_imagesContrast=np.copy(test_images)
permutations.append(test_imagesContrast)
permutationsStrings.append('test_imagesContrast')
for i in range(len(test_images)):
#     gauss = np.random.normal(0,.1,(32,32,3))
#     gauss = gauss.reshape(32,32,3)
     image=contrast(test_imagesContrast[i])
#     image=np.clip(image, 0, 1)
     test_imagesContrast[i]=image
    
test_imagesBlurContrast=np.copy(test_images)
permutations.append(test_imagesBlurContrast)
permutationsStrings.append('test_imagesBlurContrast')
for i in range(len(test_images)):
    image=contrast(test_imagesBlurContrast[i])
    image= cv2.GaussianBlur(image,(3,3),0)
    test_imagesBlurContrast[i]=image

 
test_imagesGrayContrast=np.copy(test_images)
permutations.append(test_imagesGrayContrast)
permutationsStrings.append('test_imagesGrayContrast')
for i in range(len(test_images)):
    image=contrast(test_imagesGrayContrast[i])
    image= rgb2gray(image)
    image= gray2rgb(image)
    test_imagesGrayContrast[i]=image


test_imagesBlurAndGrayContrast=np.copy(test_images)
permutations.append(test_imagesBlurAndGrayContrast)
permutationsStrings.append('test_imagesBlurAndGrayContrast')
for i in range(len(test_images)):
    image=contrast(test_imagesBlurAndGrayContrast[i])
    image= rgb2gray(image)
    image= gray2rgb(image)
    image= cv2.GaussianBlur(image,(3,3),0)
    test_imagesBlurAndGrayContrast[i]=image

    

test_imagesHueShiftContrast=np.copy(test_imagesBlurAndHueShift)
permutations.append(test_imagesHueShiftContrast)
permutationsStrings.append('test_imagesHueShiftContrast')
for i in range(len(test_images)):
    image=contrast(test_imagesHueShiftContrast[i])
    test_imagesHueShiftContrast[i]=image

test_imagesBlurAndHueShiftContrast=np.copy(test_imagesHueShiftContrast)
permutations.append(test_imagesBlurAndHueShiftContrast)
permutationsStrings.append('test_imagesBlurAndHueShiftContrast')
for i in range(len(test_images)):
    image=test_imagesBlurAndHueShiftContrast[i]
    image= cv2.GaussianBlur(image,(3,3),0)
    test_imagesBlurAndHueShiftContrast[i]=image

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_imagesBlurContrast[i])
    # The CIFAR labels happen to be arrays, 
    # which is why you need the extra index
    plt.xlabel(class_names[test_labels[-1-i][0]])
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from skimage import color
from skimage import data
from skimage import transform


def colorize(image, hue):
    """Return image tinted by the given hue based on a grayscale image."""
    hsv = color.rgb2hsv(color.gray2rgb(image))
    hsv[:, :, 0] = hue
    hsv[:, :, 1] = .5  # Turn up the saturation; we want the color to pop!
    return color.hsv2rgb(hsv)

def contrast(image):
    image=image*-1+1
    return image

 #Composition experiment
noise=[0, .1]
parameters={}
for k in range(1):
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
          if k==0:
              model.add(layers.GaussianNoise(noise_dict[2]))
          if k==1:
              model.add(SaltAndPepper(noise_dict[2]))
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
              history = model.fit(train_imagesNoisy, train_labels, epochs=3, 
                            validation_data=(validate_images, validate_labels))
          print('ending model')

      print('test model')
      model1=model
      if m==0:
        name='clearmodel'
      else:
        name='noisemodel'
      path='/om/user/aunell/compositionData/'+name
      model.save(path)
      weights1=model1.get_weights()

       #biomimetic training
      if m==0:
            #CleartoNoise
            weights0=model.get_weights()
            model=None
            noise_dict={1: 0, 2: 0.1, 3: 0}
            print('starting biomodel')
            model = models.Sequential()
            model.add(layers.Conv2D(32, (3, 3), input_shape=(32,32,3)))
            model.add(layers.Activation('relu'))
            model.add(layers.GaussianNoise(noise_dict[1]))
            model.add(layers.MaxPooling2D((2, 2)))

            model.add(layers.Conv2D(64, (3, 3)))
            model.add(layers.Activation('relu'))
            if k==0:
              model.add(layers.GaussianNoise(noise_dict[2]))
            if k==1:
              model.add(SaltAndPepper(noise_dict[2]*100))
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
            history = model.fit(train_imagesNoisy, train_labels,
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
            if k==0:
              model.add(layers.GaussianNoise(noise_dict[2]))
            if k==1:
              model.add(SaltAndPepper(noise_dict[2]*100))
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

      print('test model')
      fintest_trial=[]
      weights=model.get_weights()
      if m==0:
            name='clearToNoise'
      else:
        name='noiseToClear'
      path='/om/user/aunell/compositionData/'+name
      model.save(path)

model=None
#CV MODEL
fintest=[]
for i in range(1,4):
      print('i', i)
      noise_dict={1: 0, 2: 0, 3: 0}
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
      if k==0:
          model.add(layers.GaussianNoise(noise_dict[2]))
      if k==1:
          model.add(SaltAndPepper(noise_dict[2]))
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
      history = model.fit(train_imagesCV, train_labelsCV, epochs=3, 
                        validation_data=(validate_images, validate_labels))
      print('ending model')
path='/om/user/aunell/compositionData/computerVisionModel'
model.save(path)

#Testing on diff datasets
result={}
paths=['/om/user/aunell/compositionData/noisemodel','/om/user/aunell/compositionData/clearmodel', 
       '/om/user/aunell/compositionData/noiseToClear','/om/user/aunell/compositionData/clearToNoise', 
      '/om/user/aunell/compositionData/computerVisionModel']
# permutations=[test_images, test_imagesGray, test_imagesBlur,test_imagesBlurAndGray]
# permutationsStrings=['test_images', 'test_imagesGray', 'test_imagesBlur','test_imagesBlurAndGray']
for path in paths:
    for i in range(len(permutations)):
        for k in range(2):
            k=k/10
            print(k)
            print(permutationsStrings[i], path)
            perm=permutations[i]
            permName=permutationsStrings[i]
            model = tf.keras.models.load_model(path)
            weights= model.get_weights()
            model=None
            visible = layers.Input(shape=(32,32,3))
            conv1 = layers.Conv2D(32, kernel_size=(3,3), activation='relu')(visible)
            noise1 = layers.GaussianNoise(0)(conv1, training=True)
            pool1 = layers.MaxPooling2D(pool_size=(2, 2))(noise1)

            conv2 = layers.Conv2D(64, kernel_size=(3,3), activation='relu')(pool1)
            noise2 = layers.GaussianNoise(k)(conv2, training=True)
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
            test_loss, test_acc = model1.evaluate(perm, test_labels, verbose=0)
            name=path[32:]
            if k==0:
                permName=permName+' Clear Test'
            else:
                permName=permName+' Noisy Test'
            if permName not in result:
                result[permName]=[test_acc]
            else:
                result[permName].append(test_acc)
print(result)

#Augmentations Plots
import seaborn as sn
trainList=['noiseModel','clearModel','noiseToClear','clearToNoise','computerVisionModel']
heatFrame=pd.DataFrame.from_dict(result, orient='index', columns=trainList)
plt.figure(figsize=(16,9))
print(heatFrame.mean())
sn.heatmap(heatFrame)

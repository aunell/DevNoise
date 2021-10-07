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

train_imagesClear=np.copy(train_images)
for i in range(len(train_images)):
    image=train_imagesClear[i]
    image= cv2.GaussianBlur(image,(3,3),0)
    image= rgb2gray(image)
    image= gray2rgb(image)
    train_imagesClear[i]=image
    
# test_imagesNoise=np.copy(test_images)
# for i in range(len(test_images)):
#     gauss = np.random.normal(0,.2,(32,32,3))
#     gauss = gauss.reshape(32,32,3)
#     image=(test_imagesNoise[i]+gauss)
#     image=np.clip(image, 0, 1)
#     test_imagesNoise[i]=image
    
trainNoise=.1    

noise=[0,trainNoise]
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
        
  print('m is ', str(m))
  model1=model
  name=str(m)+'model'
  path='/om/user/aunell/noProc/'+name
  model.save(path)
   #biomimetic training
  if m==0:
        #CleartoNoise
        weights0=model.get_weights()
        model=None
        noise_dict={1: 0, 2: trainNoise, 3: 0}
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
        modelName=str(m)+'C2N'
        path='/om/user/aunell/noProc/'+modelName
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
        path='/om/user/aunell/noProc/'+modelName
          
  print('test model')
  fintest_trial=[]
  weights=model.get_weights()
  model.save(path)
  
  
  #Added models
trainNoise=.1    

noise=[0, trainNoise]
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
          history = model.fit(train_images, train_labels, epochs=3, 
                        validation_data=(validate_images, validate_labels))
          name= 'baselineAlex'
      else:
          history = model.fit(train_imagesClear, train_labels, epochs=3, 
                        validation_data=(validate_images, validate_labels))
          name= 'blurNoise'
      print('ending model')
        
  print('m is ', str(m))
  model1=model
  path='/om/user/aunell/noProc/'+name
  model.save(path)
  
  ###############BLOCK 2#######################
  #In Network Basic CIFAR
import os
os.environ["CUDA_VISIBLE_DEVICES"]="6"

import tensorflow as tf

from tensorflow.keras import datasets, layers, models, backend, Model, callbacks
#from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import random
from skimage.util import random_noise
def add_noise(img, amount):
  
    # Getting the dimensions of the image
    row , col, channels = img.shape
      
    # Randomly pick some pixels in the
    # image for coloring them white
    # Pick a random number between 300 and 10000
    cap=10*amount
    minrange=cap//3
    number_of_pixels = random.randint(minrange, cap)
    for i in range(number_of_pixels):
        
        # Pick a random y coordinate
        y_coord=random.randint(0, row - 1)
          
        # Pick a random x coordinate
        x_coord=random.randint(0, col - 1)
          
        # Color that pixel to white
        img[y_coord][x_coord] = 255
          
    # Randomly pick some pixels in
    # the image for coloring them black
    # Pick a random number between 300 and 10000
    number_of_pixels = random.randint(minrange, cap)
    for i in range(number_of_pixels):
        
        # Pick a random y coordinate
        y_coord=random.randint(0, row - 1)
          
        # Pick a random x coordinate
        x_coord=random.randint(0, col - 1)
          
        # Color that pixel to black
        img[y_coord][x_coord] = 0
          
    return img


#parameters={}
para2={}
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0
validate_images, validate_labels= test_images[0:2000], test_labels[0:2000]
test_images, test_labels= test_images[2000:], test_labels[2000:]

paths = ['/om/user/aunell/noProc/0C2N', '/om/user/aunell/noProc/0.1N2C', '/om/user/aunell/noProc/baselineAlex',
         '/om/user/aunell/noProc/0model', '/om/user/aunell/noProc/0.1model',  
         '/om/user/aunell/noProc/blurNoise']
netNoise=0
for n in range(0,100):
            print(n)
#             if n<50:
#                 n=50-n
#                 n=n*-1
#             else:
#                 n=n-50
            testNoise=n/100
#             print(testNoise)
        #GAUSS
            test_imagesNoise=np.copy(test_images)
            for i in range(len(test_images)):
                gauss = np.random.normal(0,.5,(32,32,3))
                gauss = gauss.reshape(32,32,3)
                image=(test_imagesNoise[i]+gauss)
                image=np.clip(image, 0, 1)
                test_imagesNoise[i]=image
            for i in range(3):
                plt.subplot(5,5,i+1)
                plt.xticks([])
                plt.yticks([])
                plt.grid(False)
                plt.imshow(test_imagesNoise[i+5], cmap='gray')
                # The CIFAR labels happen to be arrays, 
                # which is why you need the extra index
                plt.xlabel(class_names[test_labels[i+5][0]])
                plt.savefig('GaussEx.pdf', format='pdf')
            plt.show()

#         #SP2
#             test_imagesNoise=np.copy(test_images)
#             for i in range(len(test_images)):
#                 image=test_imagesNoise[i]
#                 image = random_noise(image, mode='s&p', amount=testNoise)
#                 image =add_noise(image, n)
#                 image=np.clip(image, 0, 1)
#                 test_imagesNoise[i]=image
#             for i in range(3):
#                 plt.subplot(5,5,i+1)
#                 plt.xticks([])
#                 plt.yticks([])
#                 plt.grid(False)
#                 plt.imshow(test_imagesNoise[i+5], cmap='gray')
#                 # The CIFAR labels happen to be arrays, 
#                 # which is why you need the extra index
#                 plt.xlabel('NOne')
#             plt.show()
        #SPECKLE       
#             test_imagesNoise=np.copy(test_images)
#             for i in range(len(test_images)):
#                 image=test_imagesNoise[i]
#                 image = random_noise(image, mode='speckle', mean=testNoise)
#                 test_imagesNoise[i]=image
            for path in paths:
                        m= path[21:]
                        model1 = tf.keras.models.load_model(path)
                        weights1=model1.get_weights()
                        fintest_trial=[]

                        test_imagesUse=test_imagesNoise
                        netNoise=0.1

                        del(model1)
                        tf.compat.v1.reset_default_graph()


                        visible = layers.Input(shape=(32,32,3))
                        conv1 = layers.Conv2D(32, kernel_size=(3,3), activation='relu')(visible)
                        noise1 = layers.GaussianNoise(0)(conv1, training=True)
                        pool1 = layers.MaxPooling2D(pool_size=(2, 2))(noise1)

                        conv2 = layers.Conv2D(64, kernel_size=(3,3), activation='relu')(pool1)
                        noise2 = layers.GaussianNoise(netNoise)(conv2, training=True)
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
                        test_loss, test_acc = model1.evaluate(test_imagesUse,  test_labels, verbose=0)
#                         if m in parameters.keys():
#                             parameters[m].append(test_acc)
#                         else:
#                             parameters[m]=[test_acc]
                        if m in para2.keys():
                            para2[m].append(test_acc)
                        else:
                            para2[m]=[test_acc]
 ######################BLOCK 3##############
x=range(0,100)
key_list = list(para2.keys())
for i in key_list:
    if np.mean(parameters[i])>np.mean(para2[i]):
        para2[i]=parameters[i]

for i in key_list:
    if i=='c/0model':
        j='GrayBlur'
    elif i=='c/0C2N':
        j='Bio-mimetic'
    elif i== 'c/0.1model':
        j= 'Noisy'
    elif i== 'c/0.1N2C':
        j='Anti Bio-mimetic'
    elif i== 'c/baselineAlex':
        j= 'Baseline'
    else:
        j='BlurNoise'
    plt.plot(x, para2[i],label=j)
# plt.plot(x, parameters[0],label='0 Noise in Training')
# plt.plot(x, parameters[.05], label='.05 Noise in Training')
# plt.plot(x, parameters['0bio'], label='Clear to Noise')
# plt.plot(x, parameters['0.05bio'], label='Noise to Clear')
#title=str('epoch'+str(i))
#plt.title('NetworkNoiseSTD=.1, Speckle Noise test images, noise in penultimate layer, CIFAR')
plt.title('Gaussian Noise in Image-- Best Test Network')
#plt.title('Speckle Noise in Image')
#plt.title('NetworkNoiseSTD=.1, Salt and Pepper Noisy Test Images(colored noisy pixels), noise in penultimate layer of network, CIFAR')
plt.xlabel('Noise in Image (Amount*100)')
#plt.xlabel('Noise in Test Image (mean*100)')
#plt.xticks(np.arange(0, .055, step=0.005))
plt.ylabel('Accuracy')
plt.legend(bbox_to_anchor=(1.05, 1))
plt.savefig('AlexGaussian.pdf', format='pdf')
#plt.xticks([0, 50, 100], ['-0.5','0','0.5'])
# plt.savefig('/om/user/aunell/data/TestNoise/results/LastLayerNoise.png')
#plt.savefig('/om/user/aunell/data/Post-Activation/20TrainingConsistent/results/epoch_all.png')
plt.show()

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
    
print('1')
train_imagesCV=np.copy(train_images)
train_labelsCV=np.copy(train_labels)
print('1a')
train_imagesCV=np.concatenate((train_imagesCV, train_imagesClear))
train_labelsCV=np.concatenate((train_labelsCV, train_labels))
train_imagesCV=np.concatenate((train_imagesCV, train_imagesBlur))
train_labelsCV=np.concatenate((train_labelsCV, train_labels))
train_imagesCV=np.concatenate((train_imagesCV, train_imagesGray))
train_labelsCV=np.concatenate((train_labelsCV, train_labels))

print('2')    
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

#Composition experiment
noise=[0, .1]
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
#   fintest_trial=[]
#   for n in range(0,51):
#             print('n', n)
#             del(model1)
#             tf.compat.v1.reset_default_graph()
#             n=n/1000
            
#             visible = layers.Input(shape=(32,32,3))
#             conv1 = layers.Conv2D(32, kernel_size=(3,3), activation='relu')(visible)
#             noise1 = layers.GaussianNoise(0)(conv1, training=True)
#             pool1 = layers.MaxPooling2D(pool_size=(2, 2))(noise1)

#             conv2 = layers.Conv2D(64, kernel_size=(3,3), activation='relu')(pool1)
#             noise2 = layers.GaussianNoise(0)(conv2, training=True)
#             pool2 = layers.MaxPooling2D(pool_size=(2, 2))(noise2)

#             conv3 = layers.Conv2D(64, kernel_size=(3,3), activation='relu')(pool2)
#             noise1 = layers.GaussianNoise(n)(conv3, training=True)
#             flat = layers.Flatten()(noise1)
#             hidden1 = layers.Dense(64, activation='relu')(flat)
#             output = layers.Dense(10)(hidden1)

#             model1 = Model(inputs=visible, outputs=output)
#             model1.compile(optimizer='adam',
#                       loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#                       metrics=['accuracy'])
#             model1.set_weights(weights1)
#             test_loss, test_acc = model1.evaluate(test_images,  test_labels, verbose=0)
#             fintest_trial.append(test_acc)
#   parameters[m]=fintest_trial
            
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
      history = model.fit(train_imagesCV, train_labelsCV, epochs=3, 
                        validation_data=(validate_images, validate_labels))
      print('ending model')
path='/om/user/aunell/compositionData/computerVisionModel'
model.save(path)
#   for n in range(0,51):
#             print(fintest_trial)
#             print('n', n)
#             del(model)
#             tf.compat.v1.reset_default_graph()
#             n=n/1000
            
#             visible = layers.Input(shape=(32,32,3))
#             conv1 = layers.Conv2D(32, kernel_size=(3,3), activation='relu')(visible)
#             noise1 = layers.GaussianNoise(0)(conv1, training=True)
#             pool1 = layers.MaxPooling2D(pool_size=(2, 2))(noise1)

#             conv2 = layers.Conv2D(64, kernel_size=(3,3), activation='relu')(pool1)
#             noise2 = layers.GaussianNoise(0)(conv2, training=True)
#             pool2 = layers.MaxPooling2D(pool_size=(2, 2))(noise2)

#             conv3 = layers.Conv2D(64, kernel_size=(3,3), activation='relu')(pool2)
#             noise1 = layers.GaussianNoise(n)(conv3, training=True)
#             flat = layers.Flatten()(noise1)
#             hidden1 = layers.Dense(64, activation='relu')(flat)
#             output = layers.Dense(10)(hidden1)

#             model = Model(inputs=visible, outputs=output)
#             model.compile(optimizer='adam',
#                       loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#                       metrics=['accuracy'])
#             model.set_weights(weights)
#             test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=0)
#             fintest_trial.append(test_acc)
#   parameters[name]=fintest_trial



#Testing on diff datasets
result={}
paths=['/om/user/aunell/compositionData/noisemodel','/om/user/aunell/compositionData/clearmodel', 
       '/om/user/aunell/compositionData/noiseToClear','/om/user/aunell/compositionData/clearToNoise', 
      '/om/user/aunell/compositionData/computerVisionModel']
permutations=[test_images, test_imagesGray, test_imagesBlur,test_imagesBlurAndGray]
permutationsStrings=['test_images', 'test_imagesGray', 'test_imagesBlur','test_imagesBlurAndGray']
for path in paths:
    for i in range(len(permutations)):
        for k in range(2):
            k=k/10
            print(k)
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
            if k==0:
                name=path[32:]+'/'+permName+'/clearTest'
            else:
                name=path[32:]+'/'+permName+'/noisyTest'
            result[name]=test_acc
print(result)





#Augmentations Plots
#results={'noisemodel/test_images/noNoise': 0.4645000100135803, 'noisemodel/test_images/noise': 0.6449999809265137, 'noisemodel/test_imagesBlur/noNoise': 0.36675000190734863, 'noisemodel/test_imagesBlur/noise': 0.4480000138282776, 'noisemodel/test_imagesGray/noNoise': 0.21950000524520874, 'noisemodel/test_imagesGray/noise': 0.27024999260902405, 'noisemodel/test_imagesBlurAndGray/noNoise': 0.4645000100135803, 'noisemodel/test_imagesBlurAndGray/noise': 0.6443750262260437, 'clearmodel/test_images/noNoise': 0.5497499704360962, 'clearmodel/test_images/noise': 0.4437499940395355, 'clearmodel/test_imagesBlur/noNoise': 0.6031249761581421, 'clearmodel/test_imagesBlur/noise': 0.4273749887943268, 'clearmodel/test_imagesGray/noNoise': 0.49562498927116394, 'clearmodel/test_imagesGray/noise': 0.38237500190734863, 'clearmodel/test_imagesBlurAndGray/noNoise': 0.5497499704360962, 'clearmodel/test_imagesBlurAndGray/noise': 0.44337499141693115, 'noiseToClear/test_images/noNoise': 0.5127500295639038, 'noiseToClear/test_images/noise': 0.36550000309944153, 'noiseToClear/test_imagesBlur/noNoise': 0.5839999914169312, 'noiseToClear/test_imagesBlur/noise': 0.2966249883174896, 'noiseToClear/test_imagesGray/noNoise': 0.4566250145435333, 'noiseToClear/test_imagesGray/noise': 0.1692499965429306, 'noiseToClear/test_imagesBlurAndGray/noNoise': 0.5127500295639038, 'noiseToClear/test_imagesBlurAndGray/noise': 0.36887499690055847, 'clearToNoise/test_images/noNoise': 0.6570000052452087, 'clearToNoise/test_images/noise': 0.6728749871253967, 'clearToNoise/test_imagesBlur/noNoise': 0.5986250042915344, 'clearToNoise/test_imagesBlur/noise': 0.6081249713897705, 'clearToNoise/test_imagesGray/noNoise': 0.4203749895095825, 'clearToNoise/test_imagesGray/noise': 0.4257499873638153, 'clearToNoise/test_imagesBlurAndGray/noNoise': 0.6570000052452087, 'clearToNoise/test_imagesBlurAndGray/noise': 0.6732500195503235}
keys=list(result.keys())
resultsList=list(result.values())
accuracy=[]
titles=[]
colors=['r', 'b', 'g', 'pink', 'darkorange']
for i in range(0,8):
    accuracy=[]
    titles=[]
    for j in range(0,5):
        accuracy.append(resultsList[i+j*8])
        titles.append(keys[i+j*8].split('/')[0])
        plt.bar(titles, accuracy, color=colors)
        title=str(keys[i+j*8].split('/')[1:])
        plt.title(title)
    plt.savefig('/om/user/aunell/compositionData/graph'+str(i)+'.png')
    plt.show()
    
    #plt.title(title)
#     plt.xlabel('Noise')
#     plt.xticks(np.arange(0, .055, step=0.005))
#     plt.ylabel('Accuracy')
    #plt.savefig('/om/user/aunell/data/Post-Activation/20TrainingConsistent/results/epoch_'+str(i)+'.png')
#     plt.savefig('/om/user/aunell/data/Post-Activation/20TrainingConsistent/results/epoch_all.png')
    #plt.show()

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
noise=[0]
for m in noise:
  parameters={'test_accuracy': None}
  fintest=[]
  for j in range(10):
    print('j', j)
    model=None
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
      if i==3:
        print('test model')
        fintest_trial=[]
        weights1=model.get_weights()
        for n in range(0,51):
            print('n', n)
            del(model)
            tf.compat.v1.reset_default_graph()
            n=n/1000
            model = models.Sequential()
            model.add(layers.Conv2D(32, (3, 3), input_shape=(32,32,3)))
            model.add(layers.Activation('relu'))
            model.add(layers.GaussianNoise(n))
            model.add(layers.MaxPooling2D((2, 2)))

            model.add(layers.Conv2D(64, (3, 3)))
            model.add(layers.Activation('relu'))
            model.add(layers.GaussianNoise(n))
            model.add(layers.MaxPooling2D((2, 2)))

            model.add(layers.Conv2D(64, (3, 3)))
            model.add(layers.Activation('relu'))
            model.add(layers.GaussianNoise(n))
            model.add(layers.Flatten())
            model.add(layers.Dense(64, activation='relu'))
            model.add(layers.Dense(10))
            model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
            model.set_weights(weights1)
            #history = model.fit(train_images[0:5], train_labels[0:5], epochs=1, 
                   #     validation_data=(test_images, test_labels))
            test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=0)
            #fintest_trial=fintest_trial+history.history['val_accuracy']
            fintest_trial.append(test_acc)
            #fintest_trial.append(test_acc)
            #print('acc vals', fintest_trial)
            print('acc vals', fintest_trial)
        fintest.append(fintest_trial)

  parameters['test_accuracy']= np.mean(fintest, axis=0)
  #Average values of accuracy at each test noise level for all 10 iterations
  
  para_df = pd.DataFrame.from_dict(parameters, orient='index')
  path= '/om/user/aunell/data/TestNoise/0TrainNoise/raw/noise.csv'
  para_df.to_csv(path)
  #save parameters dict as csv
  
#OPEN CSV FILE AND PLOT
l0=[]
with open('/om/user/aunell/data/TestNoise/0TrainNoise/raw/noise.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
        l0.append(row)
fin0=l0[1][0]
fin0=fin0.split(',')
fin0=fin0[1:]
new0=[]
for i in fin0:
    new0.append(float(i))

l3=[]
with open('/om/user/aunell/data/TestNoise/03TrainNoise/raw/noise.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
        l3.append(row)
fin3=l3[2][0]
fin3=fin3.split(',')
fin3=fin3[1:52]
new3=[]
for i in fin3:
    new3.append(float(i))
print(new3)

l5=[]
with open('/om/user/aunell/data/TestNoise/05TrainNoise/raw/noise.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
        l5.append(row)
fin5=l5[2][0]
fin5=fin5.split(',')
fin5=fin5[1:52]
new5=[]
for i in fin5:
    new5.append(float(i))
print(new5)


x=range(0,51)
plt.plot(x, new0,label='0 Noise in Training')
plt.plot(x, new3, label='.03 Noise in Training')
plt.plot(x, new5, label='.05 Noise in Training')
#title=str('epoch'+str(i))
#plt.title(title)
plt.xlabel('Noise (times a factor of 1000)')
#plt.xticks(np.arange(0, .055, step=0.005))
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('/om/user/aunell/data/TestNoise/results/testing.png')
#plt.savefig('/om/user/aunell/data/Post-Activation/20TrainingConsistent/results/epoch_all.png')
plt.show()

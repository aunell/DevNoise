#Baseline
from __future__ import print_function
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation, GaussianNoise
from tensorflow.keras.layers import AveragePooling2D, Input, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from skimage.color import rgb2gray, gray2rgb
from tensorflow.keras import datasets, layers, models, backend, Model, callbacks
#from keras.models import load_model
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import random
from skimage.util import random_noise
os.environ["CUDA_VISIBLE_DEVICES"]="7"

# Training parameters
batch_size = 32  # orig paper trained all networks with batch_size=128
epochs = 200 #200 gives optimal acc
data_augmentation = False
num_classes = 10

# Subtracting pixel mean improves accuracy
subtract_pixel_mean = True

# Model parameter
# ----------------------------------------------------------------------------
#           |      | 200-epoch | Orig Paper| 200-epoch | Orig Paper| sec/epoch
# Model     |  n   | ResNet v1 | ResNet v1 | ResNet v2 | ResNet v2 | GTX1080Ti
#           |v1(v2)| %Accuracy | %Accuracy | %Accuracy | %Accuracy | v1 (v2)
# ----------------------------------------------------------------------------
# ResNet20  | 3 (2)| 92.16     | 91.25     | -----     | -----     | 35 (---)
# ResNet32  | 5(NA)| 92.46     | 92.49     | NA        | NA        | 50 ( NA)
# ResNet44  | 7(NA)| 92.50     | 92.83     | NA        | NA        | 70 ( NA)
# ResNet56  | 9 (6)| 92.71     | 93.03     | 93.01     | NA        | 90 (100)
# ResNet110 |18(12)| 92.65     | 93.39+-.16| 93.15     | 93.63     | 165(180)
# ResNet164 |27(18)| -----     | 94.07     | -----     | 94.54     | ---(---)
# ResNet1001| (111)| -----     | 92.39     | -----     | 95.08+-.14| ---(---)
# ---------------------------------------------------------------------------
n = 6

# Model version
# Orig paper: version = 1 (ResNet v1), Improved ResNet: version = 2 (ResNet v2)
version = 2

# Computed depth from supplied model parameter n
if version == 1:
    depth = n * 6 + 2
elif version == 2:
    depth = n * 9 + 2

# Model name, depth and version
model_type = 'ResNet%dv%d' % (depth, version)

def testData(data, imNoise=None, resnet=True, cifarIndex=0):
    if data=='cifar10':
        (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
        test_images = test_images / 1.0
        if imNoise!=None:
            testNoise=n/100*255
            print(testNoise)
            test_images=np.copy(test_images)
            for i in range(len(test_images)):
                gauss = np.random.normal(0,testNoise,(32,32,3))
                gauss = gauss.reshape(32,32,3)
                image=(test_images[i]+gauss)
                image=np.clip(image, 0, 255)
                test_images[i]=image
    else:
        dirs=list_files('/om/user/aunell/Optimized/Cifar10C')
        test_labels=np.load('/om/user/aunell/Optimized/labels.npy')
        test_labels=np.array([test_labels])
        test_labels=test_labels.transpose()
   #     test_labels = keras.utils.to_categorical(test_labels, 10)
        test_labels=test_labels[-10000:]
        test_images=np.load(dirs[cifarIndex])
        test_images=test_images[-10000:]
        input_shape = test_images.shape[1:]
    # Normalize pixel values to be between 0 and 1
    test_images =  test_images / 255.0
    if resnet:
       # train_labels = keras.utils.to_categorical(train_labels, num_classes)
        test_labels = keras.utils.to_categorical(test_labels, num_classes)
        if subtract_pixel_mean:
            x_test_mean = np.mean(test_images, axis=0)
            test_images -= x_test_mean
    for i in range(3):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(test_images[i+5], cmap='gray')
        # The CIFAR labels happen to be arrays, 
        # which is why you need the extra index
        plt.xlabel('reg')
        plt.show()
    return [test_images, test_labels]

def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x



def resnet_v2(input_shape, depth, num_classes=10, noise=False):
    """ResNet Version 2 Model builder [b]

    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
    bottleneck layer
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filter maps is
    doubled. Within each stage, the layers have the same number filters and the
    same filter map sizes.
    Features maps sizes:
    conv1  : 32x32,  16
    stage 0: 32x32,  64
    stage 1: 16x16, 128
    stage 2:  8x8,  256

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    inputs = Input(shape=input_shape)
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = resnet_layer(inputs=inputs,
                     num_filters=num_filters_in,
                     conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # first layer but not first stage
                    strides = 2    # downsample

            # bottleneck residual unit
            y = resnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_in,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])

        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    if noise:
        x = GaussianNoise(.3)(x)
        #loc 1
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=8)(x)
#     if noise:
#         x = GaussianNoise(.1)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Input image dimensions.
# input_shape = testData('cifar10', imNoise=0, resnet=True, cifarIndex=0)[0]
# print(len(input_shape))
# input_shape=np.array(input_shape)
(train_images, train_labels2), (test_images2, test_labels2) = cifar10.load_data()
input_shape = train_images.shape[1:]

if version == 2:
    model = resnet_v2(input_shape=input_shape, depth=depth, noise=True)
else:
    model = resnet_v1(input_shape=input_shape, depth=depth)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=lr_schedule(0)),
              metrics=['accuracy'])
model.summary()
print(model_type)


lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

callbacks = [lr_reducer, lr_scheduler]

import tensorflow as tf

parameters={}

pathsRes= ['/om/user/aunell/data/Biomimeticloc1a', '/om/user/aunell/data/noiseloc1a',
         '/om/user/aunell/data/antiBioloc1a', '/om/user/aunell/data/baseline',
        '/om/user/aunell/data/grayBlurloc1', '/om/user/aunell/data/grayblurColor']

pathsAlex= ['/om/user/aunell/noProc/0C2N', '/om/user/aunell/noProc/blurNoise', '/om/user/aunell/noProc/0.1model',
         '/om/user/aunell/noProc/0.1N2C', '/om/user/aunell/noProc/baselineAlex',
         '/om/user/aunell/noProc/0model',  
         '/om/user/aunell/noProc/grayblurColor']

dataset='cifar10c'
resnet=True

if dataset=='cifar10':
    for n in range(0,10):
        data=testData('cifar10', imNoise=n, resnet=resnet, cifarIndex=0)
        test_images=data[0]
        test_labels=data[1]
        if resnet:
            for p in range(len(pathsRes)):
                    m= pathsRes[p][21:]
                    model = tf.keras.models.load_model(pathsRes[p])
                    weights1=model.get_weights()
                    fintest_trial=[]
                    if p==0 or p==1:
                        netNoise=0.1
                    else:
                        netNoise=0

                    del(model)
                    tf.compat.v1.reset_default_graph()
                    if p==0 or p==1:
                        model = resnet_v2(input_shape=input_shape, depth=depth, noise=True)
                    else:
                        model = resnet_v2(input_shape=input_shape, depth=depth, noise=False)
                    model.compile(loss='categorical_crossentropy',
                           optimizer=Adam(lr=lr_schedule(0)),
                           metrics=['accuracy'])
                    model.set_weights(weights1)
                    print(len(test_labels))
                    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=0)
                    print(m,  test_acc)
                    if m in parameters.keys():
                        parameters[m].append(test_acc)
                    else:
                        parameters[m]=[test_acc]
        else:
            for p in range(len(pathsAlex)):
                    m= pathsAlex[p][23:]
                    print(m)
                    model = tf.keras.models.load_model(pathsAlex[p])
                    weights1=model.get_weights()
                    fintest_trial=[]
                    if p==0 or p==1 or p==2:
                        netNoise=0.1
                    else:
                        netNoise=0

                    del(model)
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
                                  # loss='categorical_crossentropy',
                                          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                                          metrics=['accuracy'])
                    model1.set_weights(weights1)
                    test_loss, test_acc = model1.evaluate(test_images, test_labels, verbose=0)
                    if m in parameters.keys():
                        parameters[m].append(test_acc)
                    else:
                        parameters[m]=[test_acc]
    x=range(0,10)
    key_list = list(parameters.keys())
    for i in key_list:
        plt.plot(x, parameters[i],label=i)
    plt.title('Noise Robustness Alexnet-- Best Results With or Without Noise in Test')
    plt.xlabel('Noise in Test Image (% of .1 STD)')
    plt.ylabel('Accuracy')
    plt.legend(bbox_to_anchor=(1.05, 1))
                #plt.savefig('/om/user/aunell/data/Post-Activation/20TrainingConsistent/results/epoch_all.png')
    plt.show()
else:
    for j in range(len(list_files('/om/user/aunell/Optimized/Cifar10C'))):
        data=testData('cifar10c', imNoise=0, resnet=resnet, cifarIndex=j)
        test_images=data[0]
        test_labels=data[1]
        if resnet:
            for p in range(len(pathsRes)):
                    m= pathsRes[p][21:]
                    model = tf.keras.models.load_model(pathsRes[p])
                    weights1=model.get_weights()
                    fintest_trial=[]
                    if p==0 or p==1:
                        netNoise=0.1
                    else:
                        netNoise=0

                    del(model)
                    tf.compat.v1.reset_default_graph()
                    if p==0 or p==1:
                        model = resnet_v2(input_shape=input_shape, depth=depth, noise=True)
                    else:
                        model = resnet_v2(input_shape=input_shape, depth=depth, noise=False)
                    model.compile(loss='categorical_crossentropy',
                           optimizer=Adam(lr=lr_schedule(0)),
                           metrics=['accuracy'])
                    model.set_weights(weights1)
                    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=0)
                    if m in parameters.keys():
                        parameters[m].append(test_acc)
                    else:
                        parameters[m]=[test_acc]
        else:
            for p in range(len(pathsAlex)):
                    m= pathsAlex[p][23:]
                    print(m)
                    model = tf.keras.models.load_model(pathsAlex[p])
                    weights1=model.get_weights()
                    fintest_trial=[]
                    if p==0 or p==1 or p==2:
                        netNoise=0.1
                    else:
                        netNoise=0

                    del(model)
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
                    test_loss, test_acc = model1.evaluate(test_images, test_labels, verbose=0)
                    if m in parameters.keys():
                        parameters[m].append(test_acc)
                    else:
                        parameters[m]=[test_acc]
    x=range(0,10)
    key_list = list(parameters.keys())
    for i in key_list:
                plt.plot(x, parameters[i],label=i)
    plt.title('Noise Robustness Alexnet-- Best Results With or Without Noise in Test')
    plt.xlabel('Noise in Test Image (% of .1 STD)')
    plt.ylabel('Accuracy')
    plt.legend(bbox_to_anchor=(1.05, 1))
            #plt.savefig('/om/user/aunell/data/Post-Activation/20TrainingConsistent/results/epoch_all.png')
    plt.show()

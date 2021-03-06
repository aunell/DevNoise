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

os.environ["CUDA_VISIBLE_DEVICES"]="7"

#####################################

# Training parameters
batch_size = 32  # orig paper trained all networks with batch_size=128
epochs = 200
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

# Load the CIFAR10 data.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0

x_trainClear=np.copy(x_train)
for i in range(len(x_train)):
    image=x_train[i]
    image= cv2.GaussianBlur(image,(3,3),0)
    image= rgb2gray(image)
    image= gray2rgb(image)
    x_trainClear[i]=image


# Input image dimensions.
input_shape = x_train.shape[1:]


# If subtract pixel mean is enabled
if subtract_pixel_mean:
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean
    x_trainC_mean= np.mean(x_trainClear, axis=0)
    x_trainClear-=x_trainC_mean

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print('y_train shape:', y_train.shape)

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

for i in range(3):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i+5], cmap='gray')
    # The CIFAR labels happen to be arrays, 
    # which is why you need the extra index
    plt.xlabel('reg')
plt.show()

for i in range(3):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_trainClear[i+5], cmap='gray')
    # The CIFAR labels happen to be arrays, 
    # which is why you need the extra index
    plt.xlabel('Clear')
plt.show()

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


def resnet_v1(input_shape, depth, num_classes=10):
    """ResNet Version 1 Model builder [a]

    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


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
        x = GaussianNoise(.1)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


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

###############################################

import tensorflow as tf
train_images=x_train
train_labels=y_train
test_images= x_test
test_labels= y_test
validate_images = test_images[0:2000]
validate_labels = test_labels[0:2000]
train_imagesClear = x_trainClear

noise=[0]
parameters={}
for m in noise:
  print('starting model')
  if m==0:
      model= tf.keras.models.load_model('/om/user/aunell/data/grayblur')
  else:
      model = resnet_v2(input_shape=input_shape, depth=depth, noise=True)
  model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=lr_schedule(0)),
              metrics=['accuracy'])
  if m!=0:
#       model.fit(x_train, y_train,
#               batch_size=batch_size,
#               epochs=epochs,
#               validation_data=(validate_images, validate_labels),
#               shuffle=True)
#       history = model.fit(train_imagesClear, train_labels, batch_size=batch_size, epochs=epochs, 
#                         validation_data=(validate_images, validate_labels), callbacks=callbacks)
#       name = 'grayblur'
#   else:
      callbacks.pop()
      history = model.fit(train_images, train_labels, batch_size=batch_size,  epochs=epochs, 
                        validation_data=(validate_images, validate_labels), callbacks=callbacks)
      name = 'noise'
      print('ending model')
    
      model1=model
      path='/om/user/aunell/data/'+name
      model.save(path)
   #biomimetic training
  if m==0:
        #CleartoNoise
        weights0=model.get_weights()
        model=None
        print('starting biomodel')
        model = resnet_v2(input_shape=input_shape, depth=depth, noise=True)
        model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=lr_schedule(0)),
              metrics=['accuracy'])
#         model.compile(optimizer='adam',
#                       loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#                       metrics=['accuracy'])
        model.set_weights(weights0)
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=20, mode= 'max')
        callbacks.append(callback)
        history = model.fit(train_images, train_labels, batch_size=batch_size,
                            validation_data=(validate_images, validate_labels), epochs=50,
                             callbacks=callbacks)
        
        print('ending biomodel')
        modelName=str(m)+'Biomimetic'
        path='/om/user/aunell/data/Biomimetic'
        model.save(path)
        callbacks.pop()
  else:
    #NoisetoClear
        weights0=model.get_weights()
        model=None
        print('starting biomodel')
        model = resnet_v2(input_shape=input_shape, depth=depth)
        model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=lr_schedule(0)),
              metrics=['accuracy'])
        model.set_weights(weights0)
            
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=20, mode= 'max')
        callbacks.append(callback)
        history = model.fit(train_imagesClear, train_labels, batch_size=batch_size, 
                            validation_data=(validate_images, validate_labels), epochs=50,
                             callbacks=callback)
        print('ending biomodel')
        name=str(m)+'N2C'
        modelName=str(m)+'N2C'
        path='/om/user/aunell/data/AntiBiomimetic'
          
        print('test model')
        fintest_trial=[]
        weights=model.get_weights()
        model.save(path)
        
 ##########################
import tensorflow as tf

parameters={}
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0
test_labels = keras.utils.to_categorical(test_labels, 10)
test_images=test_images[2000:]
test_labels=test_labels[2000:]

paths = ['/om/user/aunell/data/baseline', '/om/user/aunell/data/Biomimetic',
         '/om/user/aunell/data/grayblur', '/om/user/aunell/data/noise', '/om/user/aunell/data/AntiBiomimetic']

for n in range(0,101):
            testNoise=n/300
            print(testNoise)
        #GAUSS
            test_imagesNoise=np.copy(test_images)
            for i in range(len(test_images)):
                gauss = np.random.normal(0,testNoise,(32,32,3))
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
                plt.xlabel(test_labels[i+5])
            plt.show()
#         #SP
#             test_imagesNoise=np.copy(test_images)
#             for i in range(len(test_images)):
#                 image=test_imagesNoise[i]
#                 image =add_noise(image, n)
#                 image=np.clip(image, 0, 1)
#                 test_imagesNoise[i]=image

#         #SP2
#             test_imagesNoise=np.copy(test_images)
#             for i in range(len(test_images)):
#                 image=test_imagesNoise[i]
#                 image = random_noise(image, mode='s&p', amount=testNoise)
#                 image =add_noise(image, n)
#                 image=np.clip(image, 0, 1)
#                 test_imagesNoise[i]=image
#     #    SPECKLE       
#             test_imagesNoise=np.copy(test_images)
#             for i in range(len(test_images)):
#                 image=test_imagesNoise[i]
#                 image = random_noise(image, mode='speckle', mean=testNoise)
#                 test_imagesNoise[i]=image
            for path in paths:
                m= path[21:]
                model = tf.keras.models.load_model(path)
                weights1=model.get_weights()
                fintest_trial=[]
                
                test_imagesUse=test_imagesNoise
                netNoise=0.1
                
                del(model)
                tf.compat.v1.reset_default_graph()


                model = resnet_v2(input_shape=input_shape, depth=depth, noise=False)
                model.compile(loss='categorical_crossentropy',
                       optimizer=Adam(lr=lr_schedule(0)),
                       metrics=['accuracy'])
                model.set_weights(weights1)
                test_loss, test_acc = model.evaluate(test_imagesUse,  test_labels, verbose=0)
                print(test_acc)
                if m in parameters.keys():
                    parameters[m].append(test_acc)
                else:
                    parameters[m]=[test_acc]
                    
##################################################

x=range(0,101)
key_list = list(parameters.keys())
for i in key_list:
    plt.plot(x, parameters[i],label=i)
#title=str('epoch'+str(i))
#plt.title('NetworkNoiseSTD=.1, Speckle Noise test images, noise in penultimate layer, CIFAR')
plt.title('Noise in ResNet')
#plt.title('Speckle Noise in Image')
#plt.title('NetworkNoiseSTD=.1, Salt and Pepper Noisy Test Images(colored noisy pixels), noise in penultimate layer of network, CIFAR')
plt.xlabel('Noise in Test Image')
#plt.xlabel('Noise in Test Image (mean*100)')
#plt.xticks(np.arange(0, .055, step=0.005))
plt.ylabel('Accuracy')
plt.legend(bbox_to_anchor=(1.05, 1))
#plt.xticks([0, 50, 100], ['-0.5','0','0.5'])
# plt.savefig('/om/user/aunell/data/TestNoise/results/LastLayerNoise.png')
#plt.savefig('/om/user/aunell/data/Post-Activation/20TrainingConsistent/results/epoch_all.png')
plt.show()

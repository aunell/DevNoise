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
import foolbox as fb

paths = ['/om/user/aunell/data/0model', '/om/user/aunell/data/0.1model', '/om/user/aunell/data/0.1N2C', 
         '/om/user/aunell/data/0C2N', '/om/user/aunell/data/blurNoise', '/om/user/aunell/data/baseline']
paths = ['/om/user/aunell/data/0C2N', '/om/user/aunell/data/baseline']

from foolbox import TensorFlowModel, accuracy, samples, Model
from foolbox.attacks import LinfPGD

for path in paths:
    model = tf.keras.models.load_model(path)
    preprocessing = dict()
    bounds = (0, 32)
    fmodel = TensorFlowModel(model, bounds=bounds, preprocessing=preprocessing)

    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0
    #validate_images, validate_labels= test_images[0:200], test_labels[0:200]
    test_images, test_labels= test_images[:200], test_labels[:200]


    #images, labels = fb.utils.samples(fmodel, dataset=datasets.cifar10.load_data(), batchsize=16)

    #print('pre-accuracy', fb.utils.accuracy(fmodel, test_images, test_labels))

    epsilons = np.linspace(0.0, 0.005, num=20)

    attack = fb.attacks.LinfDeepFoolAttack()

    raw, clipped, is_adv = attack(fmodel, test_images, test_labels, epsilons=epsilons)

    robust_accuracy = 1 - is_adv.float32().mean(axis=-1)
    plt.plot(epsilons, robust_accuracy.numpy())
    plt.title(path[21:])

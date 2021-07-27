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
    images, labels = fb.utils.samples(fmodel, dataset='cifar10', batchsize=16)
    print('pre-accuracy', fb.utils.accuracy(fmodel, images, labels))
    epsilons = np.linspace(0.0, 0.5, num=20)
    attack = fb.attacks.LinfDeepFoolAttack()
    raw, clipped, is_adv = attack(fmodel, images, labels, epsilons=epsilons)
    is_adv=is_adv.numpy()
    print(is_adv)
    robust_accuracy = 1 - is_adv.mean(axis=-1)
    print(is_adv.mean(axis=-1))
    plt.plot(epsilons, robust_accuracy, label=path[21:])
    plt.ylabel('Accuracy')
    plt.legend()

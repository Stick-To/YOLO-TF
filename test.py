import tensorflow as tf
import keras
from keras.datasets import cifar10
from keras.datasets import cifar100
import YOLOv1 as net
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
device_name = tf.test.gpu_device_name()
if device_name is not '':
    print('Found GPU Device!')
else:
    print('Found GPU Device Failed!')

config = {
    'mode': 'pretraining',     # 'detection',
    'B': 2,
    'S': 7,
    'batch_size': 64,
    'coord': 5.0,
    'noobj': 0.5,
}

mean = np.array([123.68, 116.779, 103.979]).reshape((1, 1, 1, 3))
data_shape = (224, 224, 3)
num_train = 50000
num_test = 10000
num_classes = 20
train_batch_size = 128
test_batch_size = 200
epochs = 200
weight_decay = 1e-4
lr = 0.01

testnet = net.YOLOv1(config, data_shape, num_classes, weight_decay, 'channels_last')

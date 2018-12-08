import tensorflow as tf
import YOLOv2 as net
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
device_name = tf.test.gpu_device_name()
if device_name is not '':
    print('Found GPU Device!')
else:
    print('Found GPU Device Failed!')

config = {
    'mode': 'detection',     # 'detection',
    'B': 2,
    'S': 7,
    'batch_size': 64,
    'coord': 5.0,
    'noobj': 0.5,

    'nms_score_threshold': 0.2,
    'nms_max_boxes': 20,
    'nms_iou_threshold': 0.5,

    'author_boxes_priors': [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]]
}

mean = np.array([123.68, 116.779, 103.979]).reshape((1, 1, 1, 3))
data_shape = (224, 224, 3)
num_train = 50000
num_test = 10000
num_classes = 20
epochs = 135
keep_prob = 0.5
weight_decay = 5e-4
lr = 0.01

testnet = net.YOLOv2(config, data_shape, num_classes, weight_decay, keep_prob, 'channels_last')

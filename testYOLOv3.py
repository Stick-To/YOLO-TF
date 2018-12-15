import tensorflow as tf
import YOLOv3 as net
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
device_name = tf.test.gpu_device_name()
if device_name is not '':
    print('Found GPU Device!')
else:
    print('Found GPU Device Failed!')

config = {
    'mode': 'detection',     # 'pretraining', 'detection'
    'most_labels_per_image': 10,
    'batch_size': 64,
    'coord_scale': 1.0,
    'noobj_scale': 0.5,
    'obj_scale': 1.,
    'classifier_scale': 1.,

    'nms_score_threshold': 0.2,
    'nms_max_boxes': 20,
    'nms_iou_threshold': 0.5,
    # if True the ground truth is 1.0 else iou of anchor predicted bbox and ground truth bbox
    'rescore_confidence': True,
    'anchor_boxes_priors': [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]]
}

mean = np.array([123.68, 116.779, 103.979]).reshape((1, 1, 1, 3))
data_shape = (416, 416, 3)
num_train = 50000
num_test = 10000
num_classes = 20
epochs = 135
keep_prob = 0.5
weight_decay = 5e-4
lr = 0.01

testnet = net.YOLOv3(config, data_shape, num_classes, weight_decay, keep_prob, 'channels_last')

import tensorflow as tf
import YOLOv2 as net
import numpy as np
import os
import utils.tfrecord_voc_utils as voc_utils
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
device_name = tf.test.gpu_device_name()
if device_name is not '':
    print('Found GPU Device!')
else:
    print('Found GPU Device Failed!')


mean = np.array([123.68, 116.779, 103.979]).reshape((1, 1, 1, 3))
lr = 0.01
batch_size = 1
buffer_size = 2

config = {
    'mode': 'train',     # 'train', 'test'
    'is_pretraining': True,
    'data_shape': [448, 448, 3],
    'num_classes': 20,
    'weight_decay': 1e-4,
    'keep_prob': 0.5,
    'epochs': 200,
    'data_format': 'channels_last',
    'most_labels_per_image': 10,
    'batch_size': batch_size,
    'coord_scale': 1.0,
    'noobj_scale': 0.5,
    'obj_scale': 1.,
    'classifier_scale': 1.,

    'nms_score_threshold': 0.2,
    'nms_max_boxes': 20,
    'nms_iou_threshold': 0.5,

    # if True the ground truth is 1.0 else iou of anchor predicted bbox and ground truth bbox
    'rescore_confidence': True,
    'priors': [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]]
}


train_gen = voc_utils.get_generator(['D:\\Workspace\\YOLO\\utils\\test\\test_00000-of-00001.tfrecord'],
                                    batch_size, buffer_size)
data_provider = {
    'num_train': 11,
    'num_val': 0,
    'train_generator': train_gen,
    'val_generator': None,
}

testnet = net.YOLOv2(config, data_provider)

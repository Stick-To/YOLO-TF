import tensorflow as tf
import numpy as np
import os
import utils.tfrecord_voc_utils as voc_utils
import YOLOv2 as yolov2
import matplotlib.pyplot as plt
from skimage import io, transform


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
device_name = tf.test.gpu_device_name()
if device_name is not '':
    print('Found GPU Device!')
else:
    print('Found GPU Device Failed!')

lr = 0.01
batch_size = 24
buffer_size = 100
epochs = 500
reduce_lr_epoch = [100, 300]
config = {
    'mode': 'train',     # 'train', 'test'
    'is_pretraining': False,
    'data_shape': [448, 448, 3],
    'num_classes': 20,
    'weight_decay': 0,
    'keep_prob': 0.5,
    'data_format': 'channels_last',
    'batch_size': batch_size,
    'coord_scale': 1.0,
    'noobj_scale': 0.5,
    'obj_scale': 1.,
    'class_scale': 1.,

    'nms_score_threshold': 0.5,
    'nms_max_boxes': 10,
    'nms_iou_threshold': 0.5,

    'rescore_confidence': False,
    'priors': [[1.08, 1.19], [3.42, 4.41], [6.63, 11.38], [9.42, 5.11], [16.62, 10.52]]
}
#
image_preprocess_config = {
    'data_format': 'channels_last',
    'target_size': [448, 448],
    'shorter_side': 480,
    'is_random_crop': False,
    'random_horizontal_flip': 0.5,
    'random_vertical_flip': 0,
    'pad_truth_to': 50
}

data = ['./data/test_00000-of-00005.tfrecord',
        './data/test_00001-of-00005.tfrecord',
        './data/test_00002-of-00005.tfrecord',
        './data/test_00003-of-00005.tfrecord',
        './data/test_00004-of-00005.tfrecord']

train_gen = voc_utils.get_generator(data,
                                    batch_size, buffer_size, image_preprocess_config)
trainset_provider = {
    'data_shape': [448, 448, 3],
    'num_train': 5011,
    'num_val': 0,
    'train_generator': train_gen,
    'val_generator': None
}

testnet = yolov2.YOLOv2(config, trainset_provider)
testnet.load_weight('/home/test/Desktop/YOLO-TF-master/weight/test-49088')
for i in range(epochs):
    print('-'*25, 'epoch', i, '-'*25)
    if i in reduce_lr_epoch:
        lr = lr/10.
        print('reduce lr, lr=', lr, 'now')
    mean_loss = testnet.train_one_epoch(lr)
    print('>> mean loss', mean_loss)
    testnet.save_weight('latest', './weight/test')
# img = io.imread('/home/test/Desktop/YOLO-TF-master/VOC2007/JPEGImages/000005.jpg')
# img = transform.resize(img, [448,448])
# img = np.expand_dims(img, 0)
# result = testnet.test_one_image(img)
# print(result[0])
# print(result[1]*448)
# print(result[2])
# plt.imshow(np.squeeze(img))
# plt.show()

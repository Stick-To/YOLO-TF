from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import utils.tfrecord_voc_utils as voc_utils
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

image_preprocess_config = {
    'data_format': 'channels_last',
    'target_size': [300, 300],
    'shorter_side': 301,
    'is_random_crop': False,
    'random_horizontal_flip': 0.5,
    'random_vertical_flip': 0.5,
    'pad_truth_to': 50

}
data = ['..\\test\\test_00000-of-00005.tfrecord',
        '..\\test\\test_00001-of-00005.tfrecord',
        '..\\test\\test_00002-of-00005.tfrecord',
        '..\\test\\test_00003-of-00005.tfrecord',
        '..\\test\\test_00004-of-00005.tfrecord']
init_op, iterator = voc_utils.get_generator(data, 2, 10, image_preprocess_config)
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
sess.run(init_op)
img, truth = iterator.get_next()
for i in range(1000):
    print(i)
    img1, truth1 = sess.run([img, truth])




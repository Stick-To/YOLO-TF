from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import utils.tfrecord_voc_utils as voc_utils
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


tfrecord = voc_utils.dataset2tfrecord('D:\\Workspace\\YOLO\\testfiles', 'D:\\Workspace\\YOLO\\VOC2007\\JPEGImages',
                                      '.\\test\\', 'test', 1)
iterator, init_op = voc_utils.get_generator(tfrecord, 1, 1)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(init_op)
img, ground_truth = iterator.get_next()
img = sess.run(img)
ground_truth = sess.run(ground_truth)
print(img.shape)
print(ground_truth.shape)
img = np.squeeze(img)
print(ground_truth.shape)
plt.imshow(img)
plt.show()

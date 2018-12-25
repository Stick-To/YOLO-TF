from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import utils.tfrecord_voc_utils as voc_utils
from utils.image_preprocessing import image_preprocess
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

image_preprocess_config = {
    'data_format': 'channels_last',
    'target_size': [512, 512],
    'shorter_side': 513,
    'is_random_crop': True,
    'random_horizontal_flip': .5,
    'random_vertical_flip': .5,
    'pad_truth_to': 10

}
tfrecord = voc_utils.dataset2tfrecord('..\\testfiles', '..\\VOC2007\\JPEGImages',
                                      '.\\test\\', 'test', 1)
init_op, iterator = voc_utils.get_generator(tfrecord, 4, 4, image_preprocess_config)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(init_op)
img, truth = iterator.get_next()
img, truth = sess.run([img, truth])
img = img.astype(np.int32)
print(truth)
plt.imshow(img[0])
plt.show()
plt.imshow(img[1])
plt.show()
plt.imshow(img[2])
plt.show()
plt.imshow(img[3])
plt.show()

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf


def image_preprocess(image, shape, data_format, target_size, shorter_side=None,
                     is_random_crop=False, random_horizontal_flip=0., random_vertical_flip=0.,
                     ground_truth=None, pad_truth_to=None):
    """
    :param image:  batch_size must be 1
    :param shape: [1, a, b, c]
    :param data_format: 'channels_last' or 'channels_first'
    :param target_size: generate images with target_size
    :param shorter_side: if random_crop is True, resize image the shorter side to this
            , keep aspect ratio
    :param is_random_crop: resize according to shorter_side and random crop target_size from it
    :param random_horizontal_flip: probability to random horizontal flip
    :param random_vertical_flip: probability to random vertical flip
    :param ground_truth: [ymin, ymax, xmin, xmax, class_id]
    :param pad_truth_to: padding ground truth to fix length if ground truth is not None
    :return: image:
             ground_truth: [y_center,x_center,h,w,class_id]

    """
    assert data_format in ['channels_first', 'channels_last']
    ground_truth = ground_truth
    shape = tf.reshape(shape, [-1, ])
    image = tf.expand_dims(image, 0)
    if data_format == 'channels_first':
        image = tf.transpose(image, [0, 2, 3, 1])
        shape = [1, shape[1], shape[2], shape[0]]
    else:
        shape = [1, shape[0], shape[1], shape[2]]
    if ground_truth is not None:
        if is_random_crop:
            image, ground_truth = tf.cond(shape[1] > shape[2],
                                          lambda: random_crop(image, shape, target_size, shorter_side, 2, ground_truth),
                                          lambda: random_crop(image, shape, target_size, shorter_side, 1, ground_truth)
                                          )
        else:
            image = tf.image.resize_images(image, target_size)
            height_scale = target_size[0] / shape[1]
            width_scale = target_size[1] / shape[2]
            if ground_truth is not None:
                scale = tf.convert_to_tensor([height_scale, height_scale, width_scale, width_scale, 1.], tf.float32)
                scale = tf.reshape(scale, [1, 1, 5])
                ground_truth = ground_truth * scale
        shape = [shape[0], target_size[0], target_size[1], shape[3]]
        image, ground_truth = tf.cond(
            tf.random.uniform([], 0., 1.) < random_horizontal_flip,
            lambda: horizontal_flip(image, shape, ground_truth),
            lambda: (image, ground_truth)
        )

        image, ground_truth = tf.cond(
            tf.random.uniform([], 0., 1.) < random_vertical_flip,
            lambda: vertical_flip(image, shape, ground_truth),
            lambda: (image, ground_truth)
        )
        if data_format == 'channels_first':
            image = tf.transpose(image, [0, 3, 1, 2])
        ymin = tf.expand_dims(ground_truth[:, :, 0], -1)
        ymax = tf.expand_dims(ground_truth[:, :, 1], -1)
        xmin = tf.expand_dims(ground_truth[:, :, 2], -1)
        xmax = tf.expand_dims(ground_truth[:, :, 3], -1)
        class_id = tf.expand_dims(ground_truth[:, :, 4], -1)
        h = ymax - ymin
        w = xmax - xmin
        y = ymin + h / 2
        x = xmin + w / 2
        ground_truth = tf.concat([y, x, h, w, class_id], -1)
        if pad_truth_to is not None:
            ground_truth = tf.image.pad_to_bounding_box(
                ground_truth,
                0, 0, tf.shape(ground_truth)[0], pad_truth_to
            )
        return tf.squeeze(image), tf.squeeze(ground_truth)
    else:
        if is_random_crop:
            image = tf.cond(shape[1] > shape[2],
                            lambda: random_crop(image, shape, target_size, shorter_side, 2),
                            lambda: random_crop(image, shape, target_size, shorter_side, 1)
                            )
        else:
            image = tf.image.resize_images(image, target_size)
            height_scale = target_size[0] / shape[1]
            width_scale = target_size[1] / shape[2]
            if ground_truth is not None:
                scale = tf.convert_to_tensor([height_scale, height_scale, width_scale, width_scale, 1.], tf.float32)
                scale = tf.reshape(scale, [1, 1, 5])
                ground_truth = ground_truth * scale
        shape = [shape[0], target_size[0], target_size[1], shape[3]]
        image = tf.cond(
            tf.random.uniform([], 0., 1.) < random_horizontal_flip,
            lambda: horizontal_flip(image, shape),
            lambda: image
        )

        image = tf.cond(
            tf.random.uniform([], 0., 1.) < random_vertical_flip,
            lambda: vertical_flip(image, shape),
            lambda: image
        )
        return image


def random_crop(image, shape, target_size, shorter_side, index, ground_truth=None):
    if index == 1:
        height = shorter_side
        width = tf.cast(shape[2] * shorter_side / shape[1], tf.int32)
    else:
        height = tf.cast(shape[1] * shorter_side / shape[2], tf.int32)
        width = shorter_side
    image = tf.image.resize_images(image, [height, width])
    random_height = height - target_size[0]
    random_width = width - target_size[1]
    random_height = tf.random_uniform([], 0, random_height, tf.int32)
    random_width = tf.random_uniform([], 0, random_width, tf.int32)
    image = tf.slice(
        image, [0, random_height, random_width, 0],
        [shape[0], target_size[0], target_size[1], shape[3]]
    )
    if ground_truth is not None:
        height_scale = height / shape[1]
        width_scale = width / shape[2]
        scale = tf.convert_to_tensor([height_scale, height_scale, width_scale, width_scale, 1], tf.float32)
        scale = tf.reshape(scale, [1, 1, 5])
        ground_truth = ground_truth * scale
        bias = tf.convert_to_tensor([random_height, random_height, random_width, random_width, 0.], tf.float32)
        bias = tf.reshape(bias, [1, 1, 5])
        ground_truth = ground_truth - bias
        ymin = tf.expand_dims(ground_truth[:, :, 0], -1)
        ymax = tf.expand_dims(ground_truth[:, :, 1], -1)
        xmin = tf.expand_dims(ground_truth[:, :, 2], -1)
        xmax = tf.expand_dims(ground_truth[:, :, 3], -1)
        class_id = tf.expand_dims(ground_truth[:, :, 4], -1)
        ymin = tf.where(ymin < 0, ymin-ymin, ymin)
        ymax = tf.where(ymax > target_size[0], ymax-ymax+target_size[0], ymax)
        xmin = tf.where(xmin < 0, xmin-xmin, xmin)
        xmax = tf.where(xmax > target_size[1], xmax-xmax+target_size[1], xmax)
        ground_truth = tf.concat([ymin, ymax, xmin, xmax, class_id], -1)
        return image, ground_truth
    else:
        return image


def horizontal_flip(image, shape, ground_truth=None):
    image = tf.reverse(image, [2])
    if ground_truth is not None:
        bias = tf.convert_to_tensor([0, 0, shape[2], shape[2], 0], tf.float32)
        bias = tf.reshape(bias, [1, 1, 5])
        scale = tf.convert_to_tensor([1, 1, -1, -1, 1], tf.float32)
        scale = tf.reshape(scale, [1, 1, 5])
        ground_truth = ground_truth - bias
        ground_truth = ground_truth * scale
        ground_truth = tf.gather(ground_truth, [0, 1, 3, 2, 4], axis=-1)
        return image, ground_truth
    else:
        return image


def vertical_flip(image, shape, ground_truth=None):
    image = tf.reverse(image, [1])
    if ground_truth is not None:
        bias = tf.convert_to_tensor([shape[1], shape[1], 0, 0, 0], tf.float32)
        bias = tf.reshape(bias, [1, 1, 5])
        scale = tf.convert_to_tensor([-1, -1, 1, 1, 1], tf.float32)
        scale = tf.reshape(scale, [1, 1, 5])
        ground_truth = ground_truth - bias
        ground_truth = ground_truth * scale
        ground_truth = tf.gather(ground_truth, [1, 0, 2, 3, 4], axis=-1)
        return image, ground_truth
    else:
        return image

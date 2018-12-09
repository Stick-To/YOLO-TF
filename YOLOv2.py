from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import os


class YOLOv2:
    def __init__(self, config, input_shape, num_classes, weight_decay, keep_prob, data_format):

        assert len(input_shape) == 3
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.weight_decay = weight_decay
        self.prob = 1. - keep_prob
        assert data_format in ['channels_first', 'channels_last']
        self.data_format = data_format
        self.config = config
        assert config['mode'] in ['pretraining', 'detection']
        self.mode = config['mode']
        self.batch_size = config['batch_size']
        self.coord = config['coord']
        self.noobj = config['noobj']
        self.nms_score_threshold = config['nms_score_threshold']
        self.nms_max_boxes = config['nms_max_boxes']
        self.nms_iou_threshold = config['nms_iou_threshold']
        self.author_boxes_priors = tf.convert_to_tensor(config['author_boxes_priors'], dtype=tf.float32)
        self.num_priors = len(config['author_boxes_priors'])
        self.final_units = self.num_classes + self.num_priors * 5

        self.pretraining_global_step = tf.get_variable(name='pretraining_global_step', initializer=tf.constant(0), trainable=False)
        self.global_step = tf.get_variable(name='global_step', initializer=tf.constant(0), trainable=False)
        self.is_training = True
        self._define_inputs()
        self._build_graph()
        self._init_session()
        self._create_saver()
        self._create_summary()

        pass

    def _define_inputs(self):
        shape = [None]
        shape.extend(self.input_shape)
        self.images = tf.placeholder(dtype=tf.float32, shape=shape, name='images')
        self.labels = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, None, self.num_classes], name='labels')
        self.pretraining_labels = tf.placeholder(dtype=tf.int32, shape=[None, self.num_classes], name='pre_training_labels')
        self.bbox_ground_truth = tf.placeholder(dtype=tf.float32, shape=[None, None, 4], name='bbox_ground_truth')
        self.lr = tf.placeholder(dtype=tf.float32, shape=[], name='lr')

    def _build_graph(self):

        with tf.variable_scope('feature_extractor'):
            features, passthrough = self._feature_extractor(self.images)
        with tf.variable_scope('pretraining'):
            conv = self._conv_layer(features, self.num_classes, 1, 1, 'conv1')
            axes = [1, 2] if self.data_format == 'channels_last' else [2, 3]
            global_pool = tf.reduce_mean(conv, axis=axes, name='global_pool')
            pre_loss = tf.losses.softmax_cross_entropy(self.pretraining_labels, global_pool, reduction=tf.losses.Reduction.MEAN)
            self.pre_category_pred = tf.argmax(global_pool, 1)
            self.pretraining_accuracy = tf.reduce_mean(
                tf.cast(tf.equal(self.pre_category_pred, tf.argmax(self.pretraining_labels, 1)), tf.float32), name='accuracy'
            )
        with tf.variable_scope('regressor'):
            conv1 = self._conv_layer(features, 1024, 3, 1, 'conv1')
            lrelu1 = tf.nn.leaky_relu(conv1, 0.1, 'lrelu1')
            conv2 = self._conv_layer(lrelu1, self.final_units, 1, 1, 'conv2')
            lrelu2 = tf.nn.leaky_relu(conv2, 0.1, 'lrelu2')
            conv3 = self._conv_layer(lrelu2, 1024, 3, 1, 'conv3')
            lrelu3 = tf.nn.leaky_relu(conv3, 0.1, 'lrelu3')
            conv4 = self._conv_layer(lrelu3, self.final_units, 1, 1, 'conv4')
            lrelu4 = tf.nn.leaky_relu(conv4, 0.1, 'lrelu4')
            conv5 = self._conv_layer(lrelu4, 1024, 3, 1, 'conv5')
            lrelu5 = tf.nn.leaky_relu(conv5, 0.1, 'lrelu5')
            axes = 3 if self.data_format == 'channels_last' else 1
            lrelu5 = tf.concat([passthrough, lrelu5], axis=axes)
            predictions = self._conv_layer(lrelu5, self.final_units, 1, 1, 'predictions')
            downsampling_rate = float(self.input_shape[1] / int(predictions.get_shape()[2]))
            if self.data_format == 'channels_first':
                predictions = tf.transpose(predictions, [0, 2, 3, 1])

            cells_centroid_x = [downsampling_rate*i for i in range(predictions.get_shape()[1])]
            cells_centroid_y = [downsampling_rate*j for j in range(predictions.get_shape()[2])]
            cells_centroid_x, cells_centroid_y = tf.meshgrid(cells_centroid_x, cells_centroid_y)
            cells_centroid_x = tf.reshape(cells_centroid_x, [-1, 1])
            cells_centroid_y = tf.reshape(cells_centroid_y, [-1, 1])
            cells_centroid = tf.concat([cells_centroid_x, cells_centroid_y], axis=1)

            classifier = predictions[..., :self.num_classes]
            bbox_xy = tf.nn.sigmoid(predictions[..., self.num_classes:self.num_classes+self.num_priors*2])
            bbox_hw = predictions[..., self.num_classes+self.num_priors*2:self.num_classes+self.num_priors*4]
            confidence = tf.nn.sigmoid(predictions[..., self.num_classes+self.num_priors*4:])
        with tf.variable_scope('train'):
            total_loss = []
            for i in range(self.batch_size):
                classifier_i = tf.reshape(tf.gather(classifier, i), [-1, self.num_classes])
                bbox_xy_i = tf.reshape(tf.gather(bbox_xy, i), [-1, self.num_priors, 2])
                bbox_hw_i = tf.reshape(tf.gather(bbox_hw, i), [-1, self.num_priors, 2])
                confidence_i = tf.reshape(tf.gather(confidence, i), [-1, self.num_priors, 1])

                bbox_truth_xy = self.bbox_ground_truth[i, :, :2]
                bbox_truth_hw = self.bbox_ground_truth[i, :, 2:]
                slice_index = tf.argmin(bbox_truth_xy, axis=0)
                bbox_truth_xy = tf.gather(bbox_truth_xy, tf.range(slice_index[0]))
                bbox_truth_hw = tf.gather(bbox_truth_hw, tf.range(slice_index[0]))
                classifier_truth = tf.gather(self.labels[i, :, :], tf.range(slice_index[0]))

                dist_truth_cells = -2 * tf.matmul(bbox_truth_xy, cells_centroid, transpose_b=True) \
                    + tf.expand_dims(tf.reduce_sum(bbox_truth_xy**2, axis=1), axis=1) \
                    + tf.expand_dims(tf.reduce_sum(cells_centroid**2, axis=1), axis=0)
                responsible_cells = tf.argmin(dist_truth_cells, axis=1)
                responsible_cells_ = tf.reshape(responsible_cells, [-1, 1])
                sparse_mask = tf.sparse.SparseTensor(tf.concat([responsible_cells_, tf.zeros_like(responsible_cells_)], axis=1),
                                                     tf.squeeze(tf.ones_like(responsible_cells_)), dense_shape=[confidence_i.get_shape()[0], 1])
                mask = tf.reshape(tf.sparse.to_dense(sparse_mask) < 1, [-1, ])
                noobj_confidence = tf.boolean_mask(confidence_i, mask)

                responsible_classifier = tf.gather(classifier_i, responsible_cells)
                responsible_confidence = tf.gather(confidence_i, responsible_cells)
                responsible_bboxes_xy = tf.gather(bbox_xy_i, responsible_cells)
                responsible_bboxes_hw = tf.gather(bbox_hw_i, responsible_cells)
                responsible_cells_centroid = tf.expand_dims(tf.gather(cells_centroid, responsible_cells), 1)

                denorm_resp_bboxes_xy = (responsible_bboxes_xy - 0.5) * downsampling_rate + responsible_cells_centroid
                denorm_resp_bboxes_hw = tf.exp(responsible_bboxes_hw) * self.author_boxes_priors

                iou_area = tf.abs(tf.reduce_prod(denorm_resp_bboxes_xy - bbox_truth_xy, axis=2))
                iou_rate = iou_area / (tf.abs(tf.reduce_prod(denorm_resp_bboxes_hw, axis=2)) + tf.expand_dims(tf.abs(tf.reduce_prod(bbox_truth_hw, axis=1)), 1) - iou_area)

                selected_bbox_index = tf.concat([tf.expand_dims(tf.cast(tf.range(tf.shape(iou_rate)[0]), tf.int64), 1), tf.expand_dims(tf.argmax(iou_rate, axis=1), 1)], 1)
                predicted_bbox_xy = tf.gather_nd(responsible_bboxes_xy, selected_bbox_index)
                predicted_bbox_hw = tf.gather_nd(responsible_bboxes_hw, selected_bbox_index)
                predicted_bbox_priors = tf.gather(self.author_boxes_priors, tf.argmax(iou_rate, axis=1))
                norm_bbox_truth_xy = (bbox_truth_xy / downsampling_rate) - tf.math.floor(bbox_truth_xy / downsampling_rate)
                location_loss = self.coord * tf.reduce_sum(
                    tf.square(predicted_bbox_xy - norm_bbox_truth_xy) +
                    tf.square(tf.sqrt(predicted_bbox_hw) - tf.sqrt(tf.log(bbox_truth_hw/predicted_bbox_priors)))
                )
                confidence_loss = tf.reduce_sum(tf.square(responsible_confidence - iou_rate))\
                    + self.noobj * tf.reduce_sum(tf.square(noobj_confidence))
                classifier_loss = tf.reduce_sum(tf.cast(classifier_truth, tf.float32) - responsible_classifier)
                loss = location_loss + confidence_loss + classifier_loss
                total_loss.append(loss)
            total_loss = tf.reduce_mean(total_loss)
        with tf.variable_scope('inference'):
            classifier_ = tf.reshape(classifier[0, :, :, :], [-1, 1, self.num_classes])
            classifier_ = tf.concat([classifier_]*self.num_priors, axis=1)
            confidence_ = tf.reshape(confidence[0, :, :, :], [-1, self.num_priors, 1])
            cells_centroid_ = tf.expand_dims(cells_centroid, 1)
            cells_centroid_ = tf.concat([cells_centroid_]*self.num_priors, 1)
            bbox_xy_ = tf.reshape(bbox_xy[0, :, :, :], [-1, self.num_priors, 2])
            bbox_hw_ = tf.reshape(bbox_hw[0, :, :, :], [-1, self.num_priors, 2])
            denorm_bbox_xy_ = (bbox_xy_ - 0.5) * downsampling_rate + cells_centroid_
            denorm_bbox_hw_ = tf.exp(bbox_hw_) * self.author_boxes_priors
            denorm_bbox_ = tf.expand_dims(tf.concat([denorm_bbox_xy_, denorm_bbox_hw_], axis=2), 0)
            denorm_bbox_ = tf.concat([denorm_bbox_[..., 1]-denorm_bbox_[..., 2]/2, denorm_bbox_[..., 0]-denorm_bbox_[..., 2]/2,
                                      denorm_bbox_[..., 1]+denorm_bbox_[..., 2]/2, denorm_bbox_[..., 0]-denorm_bbox_[..., 2]/2], axis=2)
            confidence_ = tf.reshape(confidence_, [-1, 1])
            classifier_ = tf.reshape(classifier_, [-1, self.num_classes])
            denorm_bbox_ = tf.reshape(denorm_bbox_, [-1, 4])
            class_specific_confidence = classifier_ * confidence_
            selected_mask = []
            for i in range(self.num_classes):
                selected_indices = tf.image.non_max_suppression(
                     denorm_bbox_, class_specific_confidence[:, i], self.nms_max_boxes, self.nms_iou_threshold, self.nms_score_threshold
                 )
                selected_indices = tf.cast(tf.reshape(selected_indices, [-1, 1]), tf.int64)
                sparse_mask_ = tf.sparse.SparseTensor(tf.concat([selected_indices, tf.zeros_like(selected_indices)], axis=1),
                                                      tf.squeeze(tf.ones_like(selected_indices)),
                                                      dense_shape=[self.num_priors*predictions.get_shape()[1]*predictions.get_shape()[2], 1])
                mask_ = tf.sparse.to_dense(sparse_mask_)
                selected_mask.append(mask_)
            selected_mask = tf.concat(selected_mask, axis=1)
            class_specific_confidence = class_specific_confidence * tf.cast(selected_mask, tf.float32)
            classes = tf.reshape(tf.argmax(class_specific_confidence, 1), [-1, 1])
            gather_index = tf.concat([tf.expand_dims(tf.range(self.num_priors*predictions.get_shape()[1]*predictions.get_shape()[2]), 1), tf.cast(classes, tf.int32)], axis=1)
            gathered_scores = tf.gather_nd(class_specific_confidence, gather_index)
            gathered_bbox = tf.gather_nd(denorm_bbox_, gather_index)
            gathered_mask = gathered_scores > 0
            gathered_scores = tf.boolean_mask(gathered_scores, gathered_mask)
            gathered_bbox = tf.boolean_mask(gathered_bbox, gathered_mask)
            sort_index = tf.contrib.framework.argsort(gathered_scores)
            self.classes = tf.gather(classes, sort_index)
            self.sorted_scores = tf.gather(gathered_scores, sort_index)
            self.sorted_bbox = tf.gather(gathered_bbox, sort_index)
        with tf.variable_scope('optimizer'):
            optimizer = tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=0.9)
            if self.mode == 'pretraining':
                self.loss = pre_loss + self.weight_decay * tf.add_n(
                    [tf.nn.l2_loss(var) for var in tf.trainable_variables('feature_extractor')]
                ) + self.weight_decay * tf.add_n(
                    [tf.nn.l2_loss(var) for var in tf.trainable_variables('pretraining')]
                )
                self.train_op = optimizer.minimize(self.loss, global_step=self.pretraining_global_step)
            else:
                self.loss = total_loss + self.weight_decay * tf.add_n(
                    [tf.nn.l2_loss(var) for var in tf.trainable_variables('feature_extractor')]
                ) + self.weight_decay * tf.add_n(
                    [tf.nn.l2_loss(var) for var in tf.trainable_variables('regressor')]
                )
                self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)

            pass

    def _init_session(self):
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

    def _create_saver(self):
        pretraining_weights = tf.trainable_variables(scope='feature_extractor')
        self.pretraining_saver = tf.train.Saver(pretraining_weights)
        self.pretraining_bestsaver = tf.train.Saver(pretraining_weights)
        detection_weights = tf.trainable_variables('feature_extractor') + tf.trainable_variables('regressor')
        self.saver = tf.train.Saver(detection_weights)
        self.best_saver = tf.train.Saver(detection_weights)

    def _create_summary(self):
        with tf.variable_scope('summaries'):
            if self.mode == 'pretraining':
                tf.summary.scalar('pretraining_loss', self.loss)
                tf.summary.scalar('pretraining_accuracy', self.pretraining_accuracy)
                self.summary_op = tf.summary.merge_all()
            else:
                tf.summary.scalar('detection_loss', self.loss)
                self.summary_op = tf.summary.merge_all()

    def _feature_extractor(self, image):
        conv1 = self._conv_layer(image, 32, 3, 1, 'conv1')
        lrelu1 = tf.nn.leaky_relu(conv1, 0.1, 'lrelu1')
        pool1 = self._max_pooling(lrelu1, 2, 2, 'pool1')

        conv2 = self._conv_layer(pool1, 64, 3, 1, 'conv2')
        lrelu2 = tf.nn.leaky_relu(conv2, 0.1, 'lrelu2')
        pool2 = self._max_pooling(lrelu2, 2, 2, 'pool2')

        conv3 = self._conv_layer(pool2, 128, 3, 1, 'conv3')
        lrelu3 = tf.nn.leaky_relu(conv3, 0.1, 'lrelu3')
        conv4 = self._conv_layer(lrelu3, 64, 1, 1, 'conv4')
        lrelu4 = tf.nn.leaky_relu(conv4, 0.1, 'lrelu4')
        conv5 = self._conv_layer(lrelu4, 128, 3, 1, 'conv5')
        lrelu5 = tf.nn.leaky_relu(conv5, 0.1, 'lrelu5')
        pool3 = self._max_pooling(lrelu5, 2, 2, 'pool3')

        conv6 = self._conv_layer(pool3, 256, 3, 1, 'conv6')
        lrelu6 = tf.nn.leaky_relu(conv6, 0.1, 'lrelu6')
        conv7 = self._conv_layer(lrelu6, 128, 1, 1, 'conv7')
        lrelu7 = tf.nn.leaky_relu(conv7, 0.1, 'lrelu7')
        conv8 = self._conv_layer(lrelu7, 256, 3, 1, 'conv8')
        lrelu8 = tf.nn.leaky_relu(conv8, 0.1, 'lrelu8')
        pool4 = self._max_pooling(lrelu8, 2, 2, 'pool4')

        conv9 = self._conv_layer(pool4, 512, 3, 1, 'conv9')
        lrelu9 = tf.nn.leaky_relu(conv9, 0.1, 'lrelu9')
        conv10 = self._conv_layer(lrelu9, 256, 1, 1, 'conv10')
        lrelu10 = tf.nn.leaky_relu(conv10, 0.1, 'lrelu10')
        conv11 = self._conv_layer(lrelu10, 512, 3, 1, 'conv11')
        lrelu11 = tf.nn.leaky_relu(conv11, 0.1, 'lrelu11')
        pool4 = self._max_pooling(lrelu11, 2, 2, 'pool4')
        conv12 = self._conv_layer(pool4, 256, 1, 1, 'conv12')
        lrelu12 = tf.nn.leaky_relu(conv12, 0.1, 'lrelu12')
        conv13 = self._conv_layer(lrelu12, 512, 3, 1, 'conv13')
        lrelu13 = tf.nn.leaky_relu(conv13, 0.1, 'lrelu13')
        pool5 = self._max_pooling(lrelu13, 2, 2, 'pool5')

        conv14 = self._conv_layer(pool5, 1024, 3, 1, 'conv14')
        lrelu14 = tf.nn.leaky_relu(conv14, 0.1, 'lrelu14')
        conv15 = self._conv_layer(lrelu14, 512, 1, 1, 'conv15')
        lrelu15 = tf.nn.leaky_relu(conv15, 0.1, 'lrelu15')
        conv16 = self._conv_layer(lrelu15, 1024, 3, 1, 'conv16')
        lrelu16 = tf.nn.leaky_relu(conv16, 0.1, 'lrelu16')
        pool4 = self._max_pooling(lrelu16, 2, 2, 'pool4')
        conv17 = self._conv_layer(pool4, 512, 1, 1, 'conv17')
        lrelu17 = tf.nn.leaky_relu(conv17, 0.1, 'lrelu17')
        conv18 = self._conv_layer(lrelu17, 1024, 3, 1, 'conv18')
        lrelu18 = tf.nn.leaky_relu(conv18, 0.1, 'lrelu18')
        return lrelu18, lrelu17

    def train_one_batch(self, images, labels, lr, mode='detection', writer=None, sess=None):
        self.is_training = True
        assert mode in ['detection', 'pretraining']
        if sess is None:
            sess_ = self.sess
        else:
            sess_ = sess
        if mode == 'pretraining':
            accuracy = self.pretraining_accuracy
            global_step = self.pretraining_global_step
            feed_dict = {self.images: images,
                         self.pretraining_labels: labels['pretraining_labels'],
                         self.lr: lr}
            _, loss, acc, summaries = sess_.run([self.train_op, self.loss, accuracy, self.summary_op], feed_dict=feed_dict)
            if writer is not None:
                writer.add_summary(summaries, global_step=global_step)
            return loss, acc
        else:
            global_step = self.global_step
            _, loss, summaries = sess_.run([self.train_op, self.loss, self.summary_op],
                                           feed_dict={self.images: images,
                                                      self.labels: labels['labels'],
                                                      self.bbox_ground_truth: labels['bbox'],
                                                      self.lr: lr})
            if writer is not None:
                writer.add_summary(summaries, global_step=global_step)
            return loss

    def valiate_one_batch(self, images, labels, mode='pretraining', sess=None):
        self.is_training = False
        if mode != 'pretraining':
            raise Exception('No validate for detection!')
        if sess is None:
            sess_ = self.sess
        else:
            sess_ = sess
        if mode == 'pretraining':
            accuracy = self.pretraining_accuracy
            feed_dict = {self.images: images,
                         self.pretraining_labels: labels['pretraining_labels'],
                         self.lr: 0.}
            loss, acc, summaries = sess_.run([self.loss, accuracy], feed_dict=feed_dict)
            return loss, acc

    def test_one_batch(self, images, labels, mode='detection', sess=None):
        self.is_training = False
        assert mode in ['detection', 'pretraining']
        if sess is None:
            sess_ = self.sess
        else:
            sess_ = sess
        if mode == 'pretraining':
            category, acc = sess_.run([self.pre_category_pred, self.pretraining_accuracy], feed_dict={
                                     self.images: images,
                                     self.pretraining_labels: labels['pretraining_labels'],
                                     self.lr: 0.
                                 })
            return category, acc
        else:
            category, scores, bbox = sess_.run([self.classes, self.sorted_scores, self.sorted_bbox], feed_dict={
                                     self.images: images,
                                     self.labels: labels['labels'],
                                     self.bbox_ground_truth: labels['bbox'],
                                     self.lr: 0.
                                 })
            return category, scores, bbox

    def save_weight(self, mode, path, sess=None):
        assert(mode in ['pretraining_latest', 'pretraining_best', 'detection_latest', 'detection_best'])
        if sess is None:
            sess_ = self.sess
        else:
            sess_ = sess
        if mode == 'pretraining_latest':
            saver = self.pretraining_saver
            global_step = self.pretraining_global_step
        elif mode == 'pretraining_best':
            saver = self.pretraining_bestsaver
            global_step = self.pretraining_global_step
        elif mode == 'detection_latest':
            saver = self.saver
            global_step = self.global_step
        else:
            saver = self.best_saver
            global_step = self.global_step

        if not tf.gfile.Exists(os.path.dirname(path)):
            tf.gfile.MakeDirs(os.path.dirname(path))
            print(os.path.dirname(path), 'does not exist, create it done')

        saver.save(sess_, path, global_step=global_step)
        print('save', mode, 'model in', path, 'successfully')

    def load_weight(self, mode, path, sess=None):
        assert(mode in ['pretraining_latest', 'pretraining_best', 'detection_latest', 'detection_best'])
        if sess is None:
            sess_ = self.sess
        else:
            sess_ = sess
        if mode == 'pretraining_latest':
            saver = self.pretraining_saver
        elif mode == 'pretraining_best':
            saver = self.pretraining_bestsaver
        elif mode == 'detection_latest':
            saver = self.saver
        else:
            saver = self.best_saver

        ckpt = tf.train.get_checkpoint_state(path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess_, path)
            print('load', mode, 'model in', path, 'successfully')
        else:
            raise FileNotFoundError('Not Found Model File!')

    def _bn(self, bottom):
        bn = tf.layers.batch_normalization(
            inputs=bottom,
            axis=3 if self.data_format == 'channels_last' else 1,
            training=self.is_training
        )
        return bn

    def _conv_layer(self, bottom, filters, kernel_size, strides, name):
        conv = tf.layers.conv2d(
            inputs=bottom,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='same',
            name=name,
            data_format=self.data_format,
            kernel_initializer=tf.contrib.layers.variance_scaling_initializer()
        )
        bn = self._bn(conv)
        return bn

    def _max_pooling(self, bottom, pool_size, strides, name):
        return tf.layers.max_pooling2d(
            inputs=bottom,
            pool_size=pool_size,
            strides=strides,
            padding='same',
            data_format=self.data_format,
            name=name
        )

    def _avg_pooling(self, bottom, pool_size, strides, name):
        return tf.layers.average_pooling2d(
            inputs=bottom,
            pool_size=pool_size,
            strides=strides,
            padding='same',
            data_format=self.data_format,
            name=name
        )

    def _dropout(self, bottom, name):
        return tf.layers.dropout(
            inputs=bottom,
            rate=self.prob,
            training=self.is_training,
            name=name
        )

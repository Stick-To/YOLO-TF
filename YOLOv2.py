from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import warnings
import os
import numpy as np


class YOLOv2:
    def __init__(self, config, data_provider):

        assert len(config['data_shape']) == 3
        assert config['mode'] in ['train', 'test']
        assert config['data_format'] in ['channels_first', 'channels_last']
        self.config = config
        self.data_provider = data_provider
        self.is_pretraining = config['is_pretraining']
        self.data_shape = config['data_shape']
        self.num_classes = config['num_classes']
        self.weight_decay = config['weight_decay']
        self.prob = 1. - config['keep_prob']
        self.data_format = config['data_format']
        self.mode = config['mode']
        self.batch_size = config['batch_size'] if config['mode'] == 'train' else 1
        self.most_labels_per_image = config['most_labels_per_image']
        self.coord_sacle = config['coord_scale']
        self.noobj_scale = config['noobj_scale']
        self.obj_scale = config['obj_scale']
        self.classifier_scale = config['classifier_scale']
        self.nms_score_threshold = config['nms_score_threshold']
        self.nms_max_boxes = config['nms_max_boxes']
        self.nms_iou_threshold = config['nms_iou_threshold']
        self.rescore_confidence = config['rescore_confidence']
        priors = tf.convert_to_tensor(config['priors'], dtype=tf.float32)
        for i in range(3):
            priors = tf.expand_dims(priors, 0)
        self.priors = priors
        self.num_priors = len(config['priors'])
        self.final_units = (self.num_classes + 5) * self.num_priors

        if self.mode == 'train':
            self.num_train = data_provider['num_train']
            self.num_val = data_provider['num_val']
            self.train_generator = data_provider['train_generator']
            self.train_initializer, self.train_iterator = self.train_generator
            if data_provider['val_generator'] is not None:
                self.val_generator = data_provider['val_generator']
                self.val_initializer, self.val_iterator = self.val_generator

        self.global_step = tf.get_variable(name='global_step', initializer=tf.constant(0), trainable=False)
        self.is_training = True

        if self.is_pretraining:
            self._define_pretraining_inputs()
            self._build_pretraining_graph()
            self._create_pretraining_saver()
            self._create_pretraining_summary()
            self.save_weight = self._save_pretraining_weight
            self.train_one_epoch = self._train_pretraining_epoch
        else:
            self._define_detection_inputs()
            self._build_detection_graph()
            self._create_detection_saver()
            self._create_detection_summary()
            self.save_weight = self._save_detection_weight
            self.train_one_epoch = self._train_detection_epoch
        self._init_session()

    def _define_pretraining_inputs(self):
        shape = [self.batch_size]
        shape.extend(self.data_shape)
        if self.mode == 'train':
            self.images, self.labels = self.train_iterator.get_next()
            self.images = tf.reshape(self.images, shape)
            self.labels = tf.cast(self.labels, tf.int32)
        else:
            self.images = tf.placeholder(tf.float32, shape, name='images')
            self.labels = tf.placeholder(tf.int32, [self.batch_size, 1], name='labels')
        self.lr = tf.placeholder(dtype=tf.float32, shape=[], name='lr')

    def _define_detection_inputs(self):
        shape = [self.batch_size]
        shape.extend(self.data_shape)
        if self.mode == 'train':
            self.images, self.ground_truth = self.train_iterator.get_next()
            self.images.set_shape(shape)
        else:
            self.images = tf.placeholder(tf.float32, shape, name='images')
            self.ground_truth = tf.placeholder(tf.float32, [self.batch_size, self.most_labels_per_image, 5], name='labels')
        self.lr = tf.placeholder(dtype=tf.float32, shape=[], name='lr')

    def _build_pretraining_graph(self):
        with tf.variable_scope('feature_extractor'):
            features, _ = self._feature_extractor(self.images)
        with tf.variable_scope('pretraining'):
            conv = self._conv_layer(features, self.num_classes, 1, 1, 'conv1')
            axes = [1, 2] if self.data_format == 'channels_last' else [2, 3]
            global_pool = tf.reduce_mean(conv, axis=axes, name='global_pool')
            labels = tf.one_hot(self.labels, self.num_classes)
            labels = tf.reshape(labels, [self.batch_size, self.num_classes])
            loss = tf.losses.softmax_cross_entropy(labels, global_pool, reduction=tf.losses.Reduction.MEAN)
            self.pred = tf.argmax(global_pool, 1)
            self.accuracy = tf.reduce_mean(
                tf.cast(tf.equal(self.pred, tf.argmax(labels, 1)), tf.float32), name='accuracy'
            )
            self.loss = loss + self.weight_decay * tf.add_n(
                [tf.nn.l2_loss(var) for var in tf.trainable_variables('feature_extractor')]
            ) + self.weight_decay * tf.add_n(
                [tf.nn.l2_loss(var) for var in tf.trainable_variables('pretraining')]
            )
            optimizer = tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=0.9)
            self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)

    def _build_detection_graph(self):
        with tf.variable_scope('feature_extractor'):
            features, passthrough = self._feature_extractor(self.images)
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
            pred = self._conv_layer(lrelu5, self.final_units, 1, 1, 'predictions')
            if self.data_format != 'channels_last':
                pred = tf.transpose(pred, [0, 2, 3, 1])
            pshape = pred.get_shape()
            if self.data_format == 'channels_last':
                if self.data_shape[0] % int(pshape[1]) != 0 or self.data_shape[1] % int(pshape[2]):
                    warnings.warn('Input shape is not a multiple of downsampling!', UserWarning)
            else:
                if self.data_shape[1] % int(pshape[1]) != 0 or self.data_shape[2] % int(pshape[2]):
                    warnings.warn('Input shape is not a multiple of downsampling!', UserWarning)
            downsampling_rate = float(self.data_shape[1] / int(pshape[2]))
        with tf.variable_scope('train'):
            topleft_y = tf.constant([i for i in range(pshape[1])], dtype=tf.float32)
            topleft_x = tf.constant([j for j in range(pshape[2])], dtype=tf.float32)
            for i in range(3):
                topleft_y = tf.expand_dims(topleft_y, -1)
            topleft_y = tf.expand_dims(topleft_y, 0)
            for i in range(2):
                topleft_x = tf.expand_dims(topleft_x, -1)
                topleft_x = tf.expand_dims(topleft_x, 0)
            topleft_y = tf.concat([topleft_y]*pshape[2], 2)
            topleft_x = tf.concat([topleft_x]*pshape[1], 1)
            topleft = tf.concat([topleft_y, topleft_x], -1)
            pclass = tf.nn.softmax(pred[..., :self.num_classes*self.num_priors])
            pbbox_yx = tf.nn.sigmoid(pred[..., self.num_classes*self.num_priors:self.num_classes*self.num_priors+self.num_priors*2])
            pbbox_hw = pred[..., self.num_classes*self.num_priors+self.num_priors*2:self.num_classes*self.num_priors+self.num_priors*4]
            pconf = tf.nn.sigmoid(pred[..., self.num_classes*self.num_priors+self.num_priors*4:])

            pclassi = tf.reshape(pclass, [self.batch_size, pshape[1], pshape[2], self.num_priors, self.num_classes])
            pbbox_yx = tf.reshape(pbbox_yx, [self.batch_size, pshape[1], pshape[2], self.num_priors, 2]) + topleft
            pbbox_hw = tf.reshape(pbbox_hw, [self.batch_size, pshape[1], pshape[2], self.num_priors, 2])
            dpbbox_yx = pbbox_yx * downsampling_rate
            dpbbox_hw = tf.exp(pbbox_hw) * self.priors
            dpbbox_y1x1i = tf.concat([tf.expand_dims(dpbbox_yx[..., 0]-dpbbox_hw[..., 0]/2, -1), tf.expand_dims(dpbbox_yx[..., 1]-dpbbox_hw[..., 1]/2, -1)], -1)
            dpbbox_y2x2i = tf.concat([tf.expand_dims(dpbbox_yx[..., 0]+dpbbox_hw[..., 0]/2, -1), tf.expand_dims(dpbbox_yx[..., 1]+dpbbox_hw[..., 1]/2, -1)], -1)
            pconfi = tf.reshape(pconf, [self.batch_size, pshape[1], pshape[2], self.num_priors, 1])
            
            abbox_hw = tf.concat([self.priors]*pshape[2], axis=2)
            abbox_hw = tf.concat([abbox_hw]*pshape[1], axis=1)
            abbox_hw = tf.concat([abbox_hw]*self.batch_size, axis=0)
            abbox_yx = (topleft + 0.5) * downsampling_rate
            abbox_y1x1 = tf.concat([tf.expand_dims(abbox_yx[..., 0]-abbox_hw[..., 0]/2, -1), tf.expand_dims(abbox_yx[..., 1]-abbox_hw[..., 1]/2, -1)], -1)
            abbox_y2x2 = tf.concat([tf.expand_dims(abbox_yx[..., 0]+abbox_hw[..., 0]/2, -1), tf.expand_dims(abbox_yx[..., 1]+abbox_hw[..., 1]/2, -1)], -1)
            gbbox_yx = self.ground_truth[:, :, :2]
            gbbox_hw = self.ground_truth[:, :, 2:4]
            for i in range(2):
                gbbox_yx = tf.expand_dims(gbbox_yx, 1)
                gbbox_hw = tf.expand_dims(gbbox_hw, 1)
            gbbox_yx = tf.concat([gbbox_yx]*pshape[1], axis=1)
            gbbox_hw = tf.concat([gbbox_hw]*pshape[1], axis=1)
            gbbox_yx = tf.concat([gbbox_yx]*pshape[2], axis=2)
            gbbox_hw = tf.concat([gbbox_hw]*pshape[2], axis=2)

            gbbox_y1x1 = tf.concat([tf.expand_dims(gbbox_yx[..., 0]-gbbox_hw[..., 0]/2, -1), tf.expand_dims(gbbox_yx[..., 1]-gbbox_hw[..., 1]/2, -1)], -1)
            gbbox_y2x2 = tf.concat([tf.expand_dims(gbbox_yx[..., 0]+gbbox_hw[..., 0]/2, -1), tf.expand_dims(gbbox_yx[..., 1]+gbbox_hw[..., 1]/2, -1)], -1)
            abbox_y1x1 = tf.concat([tf.expand_dims(abbox_y1x1, -2)]*self.most_labels_per_image, -2)
            abbox_y2x2 = tf.concat([tf.expand_dims(abbox_y2x2, -2)]*self.most_labels_per_image, -2)
            gbbox_y1x1 = tf.concat([tf.expand_dims(gbbox_y1x1, -3)]*self.num_priors, -3)
            gbbox_y2x2 = tf.concat([tf.expand_dims(gbbox_y2x2, -3)]*self.num_priors, -3)

            agiou_y1x1 = tf.maximum(abbox_y1x1, gbbox_y1x1)
            agiou_y2x2 = tf.minimum(abbox_y2x2, gbbox_y2x2)
            agiou_area = tf.reduce_prod(tf.maximum(agiou_y2x2 - agiou_y1x1, 0), axis=-1)
            agiou_rate = agiou_area / (tf.matmul(abbox_hw, gbbox_hw, transpose_b=True) - agiou_area)
            best_agiou_rate = tf.reduce_max(agiou_rate, axis=[1, 2, 3], keepdims=True)
            detectors_mask = tf.cast(tf.equal(agiou_rate, best_agiou_rate), tf.float32) * (1 - tf.cast(tf.equal(agiou_rate, tf.zeros_like(agiou_rate)), tf.float32))

            dpbbox_y1x1 = tf.concat([tf.expand_dims(dpbbox_y1x1i, -2)]*self.most_labels_per_image, -2)
            dpbbox_y2x2 = tf.concat([tf.expand_dims(dpbbox_y2x2i, -2)]*self.most_labels_per_image, -2)
            dpgiou_y1x1 = tf.maximum(dpbbox_y1x1, gbbox_y1x1)
            dpgiou_y2x2 = tf.minimum(dpbbox_y2x2, gbbox_y2x2)
            dpgiou_area = tf.reduce_prod(tf.maximum(dpgiou_y2x2 - dpgiou_y1x1, 0), axis=-1)
            dpgiou_rate = dpgiou_area / (tf.matmul(dpbbox_hw, gbbox_hw, transpose_b=True) - dpgiou_area)
            noobject_mask = tf.cast(dpgiou_rate <= 0.6, tf.float32)
            pconf = tf.concat([pconfi]*self.most_labels_per_image, axis=-1)

            noobj_loss = tf.reduce_sum((1. - detectors_mask) * noobject_mask * tf.square(pconf))
            noobj_loss = noobj_loss / self.most_labels_per_image
            if self.rescore_confidence:
                obj_loss = tf.reduce_sum(detectors_mask * tf.square(dpgiou_rate - pconf), axis=[1, 2, 3, 4])
            else:
                obj_loss = tf.reduce_sum(detectors_mask * tf.square(1. - pconf), axis=[1, 2, 3, 4])
            conf_loss = self.noobj_scale * noobj_loss + self.obj_scale * obj_loss
            conf_loss = tf.reduce_mean(conf_loss)

            pbbox_yx = tf.concat([tf.expand_dims(pbbox_yx, -2)]*self.most_labels_per_image, -2)
            pbbox_hw = tf.concat([tf.expand_dims(pbbox_hw, -2)]*self.most_labels_per_image, -2)
            ngbbox_yx = tf.concat([tf.expand_dims(gbbox_yx/downsampling_rate, -3)]*self.num_priors, -3)
            ngbbox_hw = tf.concat([tf.expand_dims(tf.log(gbbox_hw), -3)]*self.num_priors, -3) / tf.expand_dims(self.priors, -2)
            coord_loss = self.coord_sacle * tf.reduce_sum(
                tf.expand_dims(detectors_mask, -1) * tf.square(pbbox_yx - ngbbox_yx) +
                tf.expand_dims(detectors_mask, -1) * tf.square(tf.sqrt(pbbox_hw) - tf.sqrt(ngbbox_hw)),
                axis=[1, 2, 3, 4, 5]
            )
            coord_loss = tf.reduce_mean(coord_loss)

            pclass = tf.concat([tf.expand_dims(pclassi, -2)]*self.most_labels_per_image, -2)
            gclass = tf.cast(self.ground_truth[:, :, 4:], tf.int32)
            gclass = tf.squeeze(tf.one_hot(gclass, self.num_classes))
            gclass = tf.expand_dims(gclass, 1)
            gclass = tf.expand_dims(gclass, 1)
            gclass = tf.expand_dims(gclass, 1)
            gclass = tf.concat([gclass]*pshape[1], 1)
            gclass = tf.concat([gclass]*pshape[2], 2)
            gclass = tf.concat([gclass]*self.num_priors, 3)
            class_loss = self.classifier_scale * tf.reduce_sum(
                tf.expand_dims(detectors_mask, -1) * tf.square(gclass - pclass),
                axis=[1, 2, 3, 4, 5]
            )
            class_loss = tf.reduce_mean(class_loss)
        total_loss = conf_loss + coord_loss + class_loss

        with tf.variable_scope('test'):

            scoresi = tf.reshape(pclassi * pconfi, [-1, pshape[1]*pshape[2]*self.num_priors, self.num_classes])
            dpbbox_y1x1i = tf.reshape(dpbbox_y1x1i, [-1, pshape[1]*pshape[2]*self.num_priors, 2])
            dpbbox_y2x2i = tf.reshape(dpbbox_y2x2i, [-1, pshape[1]*pshape[2]*self.num_priors, 2])
            dpbbox_y1x1x2y2i = tf.concat([dpbbox_y1x1i, dpbbox_y2x2i], -1)
            selected_mask = []
            scoresi = scoresi[0, :, :]
            dpbbox_y1x1x2y2i = dpbbox_y1x1x2y2i[0, :, :]
            for k in range(self.num_classes):
                selected_indices = tf.image.non_max_suppression(
                    dpbbox_y1x1x2y2i, scoresi[:, k], self.nms_max_boxes, self.nms_iou_threshold, self.nms_score_threshold
                )
                selected_indices = tf.cast(tf.reshape(selected_indices, [-1, 1]), tf.int64)
                sparse_mask = tf.sparse.SparseTensor(tf.concat([selected_indices, tf.zeros_like(selected_indices)], axis=1),
                                                     tf.squeeze(tf.ones_like(selected_indices)), dense_shape=[self.num_priors*pshape[1]*pshape[2], 1])
                dense_mask = tf.sparse.to_dense(sparse_mask)
                selected_mask.append(dense_mask)
            selected_mask = tf.cast(tf.concat(selected_mask, axis=-1), tf.float32)
            scoresi = scoresi * selected_mask
            classesi = tf.reshape(tf.argmax(scoresi, 1), [-1, 1])
            index = tf.concat([tf.expand_dims(tf.range(self.num_priors*pshape[1]*pshape[2]), 1), tf.cast(classesi, tf.int32)], axis=1)
            scoresi = tf.gather_nd(scoresi, index)
            bboxi = tf.gather_nd(dpbbox_y1x1x2y2i, index)
            maski = scoresi > 0
            scoresi = tf.boolean_mask(scoresi, maski)
            bboxi = tf.boolean_mask(bboxi, maski)
            sorted_indexi = tf.contrib.framework.argsort(scoresi)
            sorted_classesi = tf.gather(classesi, sorted_indexi)
            sorted_scoresi = tf.gather(scoresi, sorted_indexi)
            sorted_bboxi = tf.gather(bboxi, sorted_indexi)
            self.batch_bbox_score_class = [sorted_bboxi, sorted_scoresi, sorted_classesi]

        with tf.variable_scope('optimizer'):
            optimizer = tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=0.9)
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

    def _create_pretraining_saver(self):
        weights = tf.trainable_variables(scope='feature_extractor')
        self.saver = tf.train.Saver(weights)
        self.best_saver = tf.train.Saver(weights)

    def _create_detection_saver(self):
        weights = tf.trainable_variables(scope='feature_extractor') + tf.trainable_variables('regressor')
        self.saver = tf.train.Saver(weights)
        self.best_saver = tf.train.Saver(weights)

    def _create_pretraining_summary(self):
        with tf.variable_scope('summaries'):
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('accuracy', self.accuracy)
            self.summary_op = tf.summary.merge_all()

    def _create_detection_summary(self):
        with tf.variable_scope('summaries'):
            tf.summary.scalar('loss', self.loss)
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
        conv12 = self._conv_layer(lrelu11, 256, 1, 1, 'conv12')
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
        conv17 = self._conv_layer(lrelu16, 512, 1, 1, 'conv17')
        lrelu17 = tf.nn.leaky_relu(conv17, 0.1, 'lrelu17')
        conv18 = self._conv_layer(lrelu17, 1024, 3, 1, 'conv18')
        lrelu18 = tf.nn.leaky_relu(conv18, 0.1, 'lrelu18')
        return lrelu18, lrelu17

    def _train_pretraining_epoch(self, lr, writer=None):
        self.is_training = True
        self.sess.run(self.train_initializer)
        mean_loss = []
        mean_acc = []
        for i in range(self.num_train // self.batch_size):
            _, loss, acc, summaries = self.sess.run([self.train_op, self.loss, self.accuracy, self.summary_op],
                                                    feed_dict={self.lr: lr})
            mean_loss.append(loss)
            mean_acc.append(acc)
            if writer is not None:
                writer.add_summary(summaries, global_step=self.global_step)
        mean_loss = np.mean(mean_loss)
        mean_acc = np.mean(mean_acc)
        return mean_loss, mean_acc

    def _train_detection_epoch(self, lr, writer=None):
        self.is_training = True
        self.sess.run(self.train_initializer)
        mean_loss = []
        for i in range(self.num_train // self.batch_size):
            _, loss, summaries = self.sess.run([self.train_op, self.loss, self.summary_op],
                                               feed_dict={self.lr: lr})
            mean_loss.append(loss)
            if writer is not None:
                writer.add_summary(summaries, global_step=self.global_step)
        mean_loss = np.mean(mean_loss)
        return mean_loss

    def _test_one_pretraining_batch(self, images):
        self.is_training = False
        pred = self.sess.run(self.pred, feed_dict={
            self.images: images
        })
        return pred

    def _test_one_image(self, images):
        self.is_training = False
        self.images = images
        pred = self.sess.run(self.batch_bbox_score_class, feed_dict={
            self.images: images
        })
        return pred

    def _save_pretraining_weight(self, mode, path):
        assert(mode in ['latest', 'best'])
        if mode == 'latest':
            saver = self.saver
        else:
            saver = self.best_saver
        if not tf.gfile.Exists(os.path.dirname(path)):
            tf.gfile.MakeDirs(os.path.dirname(path))
            print(os.path.dirname(path), 'does not exist, create it done')
        saver.save(self.sess, path, global_step=self.global_step)
        print('save', mode, 'model in', path, 'successfully')

    def _save_detection_weight(self, mode, path):
        assert(mode in ['latest', 'best'])
        if mode == 'latest':
            saver = self.saver
        else:
            saver = self.best_saver
        if not tf.gfile.Exists(os.path.dirname(path)):
            tf.gfile.MakeDirs(os.path.dirname(path))
            print(os.path.dirname(path), 'does not exist, create it done')
        saver.save(self.sess, path, global_step=self.global_step)
        print('save', mode, 'model in', path, 'successfully')

    def load_weight(self, path):
        ckpt = tf.train.get_checkpoint_state(path)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, path)
            print('load model in', path, 'successfully')
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

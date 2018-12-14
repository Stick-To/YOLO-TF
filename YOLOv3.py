from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import warnings
import os


class YOLOv3:
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
        self.most_labels_per_image = config['most_labels_per_image']
        self.coord_sacle = config['coord_scale']
        self.noobj_scale = config['noobj_scale']
        self.obj_scale = config['obj_scale']
        self.classifier_scale = config['classifier_scale']
        self.nms_score_threshold = config['nms_score_threshold']
        self.nms_max_boxes = config['nms_max_boxes']
        self.nms_iou_threshold = config['nms_iou_threshold']
        self.rescore_confidence = config['rescore_confidence']
        assert len(config['anchor_boxes_priors']) >= 3
        anchor_boxes_priors = tf.convert_to_tensor(config['anchor_boxes_priors'], dtype=tf.float32)
        for i in range(3):
            anchor_boxes_priors = tf.expand_dims(anchor_boxes_priors, 0)
        self.num_priors = len(config['anchor_boxes_priors']) // 3
        self.final_units = (self.num_classes + 5) * self.num_priors
        self.anchor_boxes_priors = []
        for i in range(2, -1, -1):
            self.anchor_boxes_priors.append(anchor_boxes_priors[..., self.num_priors*i:self.num_priors*(i+1), :])

        self.pretraining_global_step = tf.get_variable(name='pretraining_global_step', initializer=tf.constant(0), trainable=False)
        self.global_step = tf.get_variable(name='global_step', initializer=tf.constant(0), trainable=False)
        self.is_training = True
        self._define_inputs()
        self._build_graph()
        self._init_session()
        # self._create_saver()
        # self._create_summary()

        pass

    def _define_inputs(self):
        shape = [self.batch_size]
        shape.extend(self.input_shape)
        self.images = tf.placeholder(dtype=tf.float32, shape=shape, name='images')
        self.labels = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.most_labels_per_image, self.num_classes], name='labels')
        self.pretraining_labels = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.num_classes], name='pre_training_labels')
        self.bbox_ground_truth = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.most_labels_per_image, 4], name='bbox_ground_truth')
        self.lr = tf.placeholder(dtype=tf.float32, shape=[], name='lr')

    def _build_graph(self):

        with tf.variable_scope('feature_extractor'):
            pyramid1, pyramid2, pyramid3 = self._feature_extractor(self.images)
        with tf.variable_scope('pretraining'):
            conv = self._conv_layer(pyramid1, self.num_classes, 1, 1)
            axes = [1, 2] if self.data_format == 'channels_last' else [2, 3]
            global_pool = tf.reduce_mean(conv, axis=axes, name='global_pool')
            pre_loss = tf.losses.softmax_cross_entropy(self.pretraining_labels, global_pool, reduction=tf.losses.Reduction.MEAN)
            self.pre_category_pred = tf.argmax(global_pool, 1)
            self.pretraining_accuracy = tf.reduce_mean(
                tf.cast(tf.equal(self.pre_category_pred, tf.argmax(self.pretraining_labels, 1)), tf.float32), name='accuracy'
            )
        with tf.variable_scope('regressor'):
            pred1, top_down = self._yolo3_header(pyramid1, 1024, 'pyramid1', )
            pred2, top_down = self._yolo3_header(pyramid2, 256, 'pyramid2', top_down)
            pred3, _ = self._yolo3_header(pyramid3, 128, 'pyramid3', top_down)

            if self.data_format != 'channels_last':
                pred1 = tf.transpose(pred1, [0, 2, 3, 1])
                pred2 = tf.transpose(pred2, [0, 2, 3, 1])
                pred3 = tf.transpose(pred3, [0, 2, 3, 1])
            p1shape = pred1.get_shape()
            p2shape = pred2.get_shape()
            p3shape = pred3.get_shape()
            downsampling_rate1 = float(self.input_shape[1] / int(p1shape[2]))
            downsampling_rate2 = float(self.input_shape[1] / int(p2shape[2]))
            downsampling_rate3 = float(self.input_shape[1] / int(p3shape[2]))
            if self.input_shape[1] % int(p3shape[2]) != 0:
                warnings.warn('downsampling rate is not a interget', UserWarning)

        with tf.variable_scope('train'):
            topleft_x1 = tf.constant([i for i in range(p1shape[1])], dtype=tf.float32)
            topleft_y1 = tf.constant([j for j in range(p1shape[2])], dtype=tf.float32)
            topleft_x2 = tf.constant([i for i in range(p2shape[1])], dtype=tf.float32)
            topleft_y2 = tf.constant([j for j in range(p2shape[2])], dtype=tf.float32)
            topleft_x3 = tf.constant([i for i in range(p3shape[1])], dtype=tf.float32)
            topleft_y3 = tf.constant([j for j in range(p3shape[2])], dtype=tf.float32)
            for i in range(3):
                topleft_x1 = tf.expand_dims(topleft_x1, -1)
                topleft_x2 = tf.expand_dims(topleft_x2, -1)
                topleft_x3 = tf.expand_dims(topleft_x3, -1)
            topleft_x1 = tf.expand_dims(topleft_x1, 0)
            topleft_x2 = tf.expand_dims(topleft_x2, 0)
            topleft_x3 = tf.expand_dims(topleft_x3, 0)
            for i in range(2):
                topleft_y1 = tf.expand_dims(topleft_y1, -1)
                topleft_y1 = tf.expand_dims(topleft_y1, 0)
                topleft_y2 = tf.expand_dims(topleft_y2, -1)
                topleft_y2 = tf.expand_dims(topleft_y2, 0)
                topleft_y3 = tf.expand_dims(topleft_y3, -1)
                topleft_y3 = tf.expand_dims(topleft_y3, 0)
            topleft_x1 = tf.concat([topleft_x1]*p1shape[2], 2)
            topleft_y1 = tf.concat([topleft_y1]*p1shape[1], 1)
            topleft_x2 = tf.concat([topleft_x2]*p2shape[2], 2)
            topleft_y2 = tf.concat([topleft_y2]*p2shape[1], 1)
            topleft_x3 = tf.concat([topleft_x3]*p3shape[2], 2)
            topleft_y3 = tf.concat([topleft_y3]*p3shape[1], 1)
            topleft1 = tf.concat([topleft_x1, topleft_y1], -1)
            topleft2 = tf.concat([topleft_x2, topleft_y2], -1)
            topleft3 = tf.concat([topleft_x3, topleft_y3], -1)

            p1class = pred1[..., :self.num_classes*self.num_priors]
            p1bbox_xy = tf.nn.sigmoid(pred1[..., self.num_classes*self.num_priors:self.num_classes*self.num_priors+self.num_priors*2])
            p1bbox_hw = pred1[..., self.num_classes*self.num_priors+self.num_priors*2:self.num_classes*self.num_priors+self.num_priors*4]
            p1conf = tf.nn.sigmoid(pred1[..., self.num_classes*self.num_priors+self.num_priors*4:])
            p2class = pred2[..., :self.num_classes*self.num_priors]
            p2bbox_xy = tf.nn.sigmoid(pred2[..., self.num_classes*self.num_priors:self.num_classes*self.num_priors+self.num_priors*2])
            p2bbox_hw = pred2[..., self.num_classes*self.num_priors+self.num_priors*2:self.num_classes*self.num_priors+self.num_priors*4]
            p2conf = tf.nn.sigmoid(pred2[..., self.num_classes*self.num_priors+self.num_priors*4:])
            p3class = pred3[..., :self.num_classes*self.num_priors]
            p3bbox_xy = tf.nn.sigmoid(pred3[..., self.num_classes*self.num_priors:self.num_classes*self.num_priors+self.num_priors*2])
            p3bbox_hw = pred3[..., self.num_classes*self.num_priors+self.num_priors*2:self.num_classes*self.num_priors+self.num_priors*4]
            p3conf = tf.nn.sigmoid(pred3[..., self.num_classes*self.num_priors+self.num_priors*4:])

            p1classi = tf.reshape(p1class, [self.batch_size, p1shape[1], p1shape[2], self.num_priors, self.num_classes])
            p1bbox_xy = tf.reshape(p1bbox_xy, [self.batch_size, p1shape[1], p1shape[2], self.num_priors, 2]) + topleft1
            p1bbox_hw = tf.reshape(p1bbox_hw, [self.batch_size, p1shape[1], p1shape[2], self.num_priors, 2])
            dp1bbox_xy = p1bbox_xy * downsampling_rate1
            dp1bbox_hw = tf.exp(p1bbox_hw) * self.anchor_boxes_priors[0]
            dp1bbox_y1x1i = tf.concat([tf.expand_dims(dp1bbox_xy[..., 0]-dp1bbox_hw[..., 0]/2, -1), tf.expand_dims(dp1bbox_xy[..., 1]-dp1bbox_hw[..., 1]/2, -1)], -1)
            dp1bbox_y2x2i = tf.concat([tf.expand_dims(dp1bbox_xy[..., 0]+dp1bbox_hw[..., 0]/2, -1), tf.expand_dims(dp1bbox_xy[..., 1]+dp1bbox_hw[..., 1]/2, -1)], -1)
            p1confi = tf.reshape(p1conf, [self.batch_size, p1shape[1], p1shape[2], self.num_priors, 1])

            p2classi = tf.reshape(p2class, [self.batch_size, p2shape[1], p2shape[2], self.num_priors, self.num_classes])
            p2bbox_xy = tf.reshape(p2bbox_xy, [self.batch_size, p2shape[1], p2shape[2], self.num_priors, 2]) + topleft2
            p2bbox_hw = tf.reshape(p2bbox_hw, [self.batch_size, p2shape[1], p2shape[2], self.num_priors, 2])
            dp2bbox_xy = p2bbox_xy * downsampling_rate2
            dp2bbox_hw = tf.exp(p2bbox_hw) * self.anchor_boxes_priors[1]
            dp2bbox_y1x1i = tf.concat([tf.expand_dims(dp2bbox_xy[..., 0]-dp2bbox_hw[..., 0]/2, -1), tf.expand_dims(dp2bbox_xy[..., 1]-dp2bbox_hw[..., 1]/2, -1)], -1)
            dp2bbox_y2x2i = tf.concat([tf.expand_dims(dp2bbox_xy[..., 0]+dp2bbox_hw[..., 0]/2, -1), tf.expand_dims(dp2bbox_xy[..., 1]+dp2bbox_hw[..., 1]/2, -1)], -1)
            p2confi = tf.reshape(p2conf, [self.batch_size, p2shape[1], p2shape[2], self.num_priors, 1])

            p3classi = tf.reshape(p3class, [self.batch_size, p3shape[1], p3shape[2], self.num_priors, self.num_classes])
            p3bbox_xy = tf.reshape(p3bbox_xy, [self.batch_size, p3shape[1], p3shape[2], self.num_priors, 2]) + topleft3
            p3bbox_hw = tf.reshape(p3bbox_hw, [self.batch_size, p3shape[1], p3shape[2], self.num_priors, 2])
            dp3bbox_xy = p3bbox_xy * downsampling_rate3
            dp3bbox_hw = tf.exp(p3bbox_hw) * self.anchor_boxes_priors[2]
            dp3bbox_y1x1i = tf.concat([tf.expand_dims(dp3bbox_xy[..., 0]-dp3bbox_hw[..., 0]/2, -1), tf.expand_dims(dp3bbox_xy[..., 1]-dp3bbox_hw[..., 1]/2, -1)], -1)
            dp3bbox_y2x2i = tf.concat([tf.expand_dims(dp3bbox_xy[..., 0]+dp3bbox_hw[..., 0]/2, -1), tf.expand_dims(dp3bbox_xy[..., 1]+dp3bbox_hw[..., 1]/2, -1)], -1)
            p3confi = tf.reshape(p3conf, [self.batch_size, p3shape[1], p3shape[2], self.num_priors, 1])

            a1bbox_hw = tf.concat([self.anchor_boxes_priors[0]]*p1shape[2], axis=2)
            a1bbox_hw = tf.concat([a1bbox_hw]*p1shape[1], axis=1)
            a1bbox_hw = tf.concat([a1bbox_hw]*self.batch_size, axis=0)
            a1bbox_xy = (topleft1 + 0.5) * downsampling_rate1
            a1bbox_y1x1 = tf.concat([tf.expand_dims(a1bbox_xy[..., 0]-a1bbox_hw[..., 0]/2, -1), tf.expand_dims(a1bbox_xy[..., 1]-a1bbox_hw[..., 1]/2, -1)], -1)
            a1bbox_y2x2 = tf.concat([tf.expand_dims(a1bbox_xy[..., 0]+a1bbox_hw[..., 0]/2, -1), tf.expand_dims(a1bbox_xy[..., 1]+a1bbox_hw[..., 1]/2, -1)], -1)

            a2bbox_hw = tf.concat([self.anchor_boxes_priors[1]]*p2shape[2], axis=2)
            a2bbox_hw = tf.concat([a2bbox_hw]*p2shape[1], axis=1)
            a2bbox_hw = tf.concat([a2bbox_hw]*self.batch_size, axis=0)
            a2bbox_xy = (topleft2 + 0.5) * downsampling_rate2
            a2bbox_y1x1 = tf.concat([tf.expand_dims(a2bbox_xy[..., 0]-a2bbox_hw[..., 0]/2, -1), tf.expand_dims(a2bbox_xy[..., 1]-a2bbox_hw[..., 1]/2, -1)], -1)
            a2bbox_y2x2 = tf.concat([tf.expand_dims(a2bbox_xy[..., 0]+a2bbox_hw[..., 0]/2, -1), tf.expand_dims(a2bbox_xy[..., 1]+a2bbox_hw[..., 1]/2, -1)], -1)

            a3bbox_hw = tf.concat([self.anchor_boxes_priors[2]]*p3shape[2], axis=2)
            a3bbox_hw = tf.concat([a3bbox_hw]*p3shape[1], axis=1)
            a3bbox_hw = tf.concat([a3bbox_hw]*self.batch_size, axis=0)
            a3bbox_xy = (topleft3 + 0.5) * downsampling_rate3
            a3bbox_y1x1 = tf.concat([tf.expand_dims(a3bbox_xy[..., 0]-a3bbox_hw[..., 0]/2, -1), tf.expand_dims(a3bbox_xy[..., 1]-a3bbox_hw[..., 1]/2, -1)], -1)
            a3bbox_y2x2 = tf.concat([tf.expand_dims(a3bbox_xy[..., 0]+a3bbox_hw[..., 0]/2, -1), tf.expand_dims(a3bbox_xy[..., 1]+a3bbox_hw[..., 1]/2, -1)], -1)

            g1bbox_xy = self.bbox_ground_truth[:, :, :2]
            g1bbox_hw = self.bbox_ground_truth[:, :, 2:]
            for i in range(2):
                g1bbox_xy = tf.expand_dims(g1bbox_xy, 1)
                g1bbox_hw = tf.expand_dims(g1bbox_hw, 1)
            g1bbox_xy = tf.concat([g1bbox_xy]*p1shape[1], axis=1)
            g1bbox_hw = tf.concat([g1bbox_hw]*p1shape[1], axis=1)
            g1bbox_xy = tf.concat([g1bbox_xy]*p1shape[2], axis=2)
            g1bbox_hw = tf.concat([g1bbox_hw]*p1shape[2], axis=2)
            g1bbox_y1x1 = tf.concat([tf.expand_dims(g1bbox_xy[..., 0]-g1bbox_hw[..., 0]/2, -1), tf.expand_dims(g1bbox_xy[..., 1]-g1bbox_hw[..., 1]/2, -1)], -1)
            g1bbox_y2x2 = tf.concat([tf.expand_dims(g1bbox_xy[..., 0]+g1bbox_hw[..., 0]/2, -1), tf.expand_dims(g1bbox_xy[..., 1]+g1bbox_hw[..., 1]/2, -1)], -1)
            a1bbox_y1x1 = tf.concat([tf.expand_dims(a1bbox_y1x1, -2)]*self.most_labels_per_image, -2)
            a1bbox_y2x2 = tf.concat([tf.expand_dims(a1bbox_y2x2, -2)]*self.most_labels_per_image, -2)
            g1bbox_y1x1 = tf.concat([tf.expand_dims(g1bbox_y1x1, -3)]*self.num_priors, -3)
            g1bbox_y2x2 = tf.concat([tf.expand_dims(g1bbox_y2x2, -3)]*self.num_priors, -3)

            g2bbox_xy = self.bbox_ground_truth[:, :, :2]
            g2bbox_hw = self.bbox_ground_truth[:, :, 2:]
            for i in range(2):
                g2bbox_xy = tf.expand_dims(g2bbox_xy, 1)
                g2bbox_hw = tf.expand_dims(g2bbox_hw, 1)
            g2bbox_xy = tf.concat([g2bbox_xy]*p2shape[1], axis=1)
            g2bbox_hw = tf.concat([g2bbox_hw]*p2shape[1], axis=1)
            g2bbox_xy = tf.concat([g2bbox_xy]*p2shape[2], axis=2)
            g2bbox_hw = tf.concat([g2bbox_hw]*p2shape[2], axis=2)
            g2bbox_y1x1 = tf.concat([tf.expand_dims(g2bbox_xy[..., 0]-g2bbox_hw[..., 0]/2, -1), tf.expand_dims(g2bbox_xy[..., 1]-g2bbox_hw[..., 1]/2, -1)], -1)
            g2bbox_y2x2 = tf.concat([tf.expand_dims(g2bbox_xy[..., 0]+g2bbox_hw[..., 0]/2, -1), tf.expand_dims(g2bbox_xy[..., 1]+g2bbox_hw[..., 1]/2, -1)], -1)
            a2bbox_y1x1 = tf.concat([tf.expand_dims(a2bbox_y1x1, -2)]*self.most_labels_per_image, -2)
            a2bbox_y2x2 = tf.concat([tf.expand_dims(a2bbox_y2x2, -2)]*self.most_labels_per_image, -2)
            g2bbox_y1x1 = tf.concat([tf.expand_dims(g2bbox_y1x1, -3)]*self.num_priors, -3)
            g2bbox_y2x2 = tf.concat([tf.expand_dims(g2bbox_y2x2, -3)]*self.num_priors, -3)

            g3bbox_xy = self.bbox_ground_truth[:, :, :2]
            g3bbox_hw = self.bbox_ground_truth[:, :, 2:]
            for i in range(2):
                g3bbox_xy = tf.expand_dims(g3bbox_xy, 1)
                g3bbox_hw = tf.expand_dims(g3bbox_hw, 1)
            g3bbox_xy = tf.concat([g3bbox_xy]*p3shape[1], axis=1)
            g3bbox_hw = tf.concat([g3bbox_hw]*p3shape[1], axis=1)
            g3bbox_xy = tf.concat([g3bbox_xy]*p3shape[2], axis=2)
            g3bbox_hw = tf.concat([g3bbox_hw]*p3shape[2], axis=2)
            g3bbox_y1x1 = tf.concat([tf.expand_dims(g3bbox_xy[..., 0]-g3bbox_hw[..., 0]/2, -1), tf.expand_dims(g3bbox_xy[..., 1]-g3bbox_hw[..., 1]/2, -1)], -1)
            g3bbox_y2x2 = tf.concat([tf.expand_dims(g3bbox_xy[..., 0]+g3bbox_hw[..., 0]/2, -1), tf.expand_dims(g3bbox_xy[..., 1]+g3bbox_hw[..., 1]/2, -1)], -1)
            a3bbox_y1x1 = tf.concat([tf.expand_dims(a3bbox_y1x1, -2)]*self.most_labels_per_image, -2)
            a3bbox_y2x2 = tf.concat([tf.expand_dims(a3bbox_y2x2, -2)]*self.most_labels_per_image, -2)
            g3bbox_y1x1 = tf.concat([tf.expand_dims(g3bbox_y1x1, -3)]*self.num_priors, -3)
            g3bbox_y2x2 = tf.concat([tf.expand_dims(g3bbox_y2x2, -3)]*self.num_priors, -3)

            ag1iou_y1x1 = tf.maximum(a1bbox_y1x1, g1bbox_y1x1)
            ag1iou_y2x2 = tf.minimum(a1bbox_y2x2, g1bbox_y2x2)
            ag1iou_area = tf.reduce_prod(tf.maximum(ag1iou_y2x2 - ag1iou_y1x1, 0), axis=-1)
            ag1iou_rate = ag1iou_area / (tf.matmul(a1bbox_hw, g1bbox_hw, transpose_b=True) - ag1iou_area)
            best1_agiou_rate = tf.reduce_max(ag1iou_rate, axis=[1, 2, 3], keepdims=True)

            ag2iou_y1x1 = tf.maximum(a2bbox_y1x1, g2bbox_y1x1)
            ag2iou_y2x2 = tf.minimum(a2bbox_y2x2, g2bbox_y2x2)
            ag2iou_area = tf.reduce_prod(tf.maximum(ag2iou_y2x2 - ag2iou_y1x1, 0), axis=-1)
            ag2iou_rate = ag2iou_area / (tf.matmul(a2bbox_hw, g2bbox_hw, transpose_b=True) - ag2iou_area)
            best2_agiou_rate = tf.reduce_max(ag2iou_rate, axis=[1, 2, 3], keepdims=True)

            ag3iou_y1x1 = tf.maximum(a3bbox_y1x1, g3bbox_y1x1)
            ag3iou_y2x2 = tf.minimum(a3bbox_y2x2, g3bbox_y2x2)
            ag3iou_area = tf.reduce_prod(tf.maximum(ag3iou_y2x2 - ag3iou_y1x1, 0), axis=-1)
            ag3iou_rate = ag3iou_area / (tf.matmul(a3bbox_hw, g3bbox_hw, transpose_b=True) - ag3iou_area)
            best3_agiou_rate = tf.reduce_max(ag3iou_rate, axis=[1, 2, 3], keepdims=True)

            best1_agiou_rate_ = tf.cast(best1_agiou_rate > best2_agiou_rate, tf.float32) * tf.cast(best1_agiou_rate > best3_agiou_rate, tf.float32)
            best2_agiou_rate_ = tf.cast(best2_agiou_rate > best1_agiou_rate, tf.float32) * tf.cast(best2_agiou_rate > best3_agiou_rate, tf.float32)
            best3_agiou_rate_ = tf.cast(best3_agiou_rate > best1_agiou_rate, tf.float32) * tf.cast(best3_agiou_rate > best2_agiou_rate, tf.float32)
            best1_agiou_rate = best1_agiou_rate_ * best1_agiou_rate
            best2_agiou_rate = best2_agiou_rate_ * best2_agiou_rate
            best3_agiou_rate = best3_agiou_rate_ * best3_agiou_rate

            detectors_mask1 = tf.cast(tf.equal(ag1iou_rate, best1_agiou_rate), tf.float32) * (1 - tf.cast(tf.equal(ag1iou_rate, tf.zeros_like(ag1iou_rate)), tf.float32))
            detectors_mask2 = tf.cast(tf.equal(ag2iou_rate, best2_agiou_rate), tf.float32) * (1 - tf.cast(tf.equal(ag2iou_rate, tf.zeros_like(ag2iou_rate)), tf.float32))
            detectors_mask3 = tf.cast(tf.equal(ag3iou_rate, best3_agiou_rate), tf.float32) * (1 - tf.cast(tf.equal(ag3iou_rate, tf.zeros_like(ag3iou_rate)), tf.float32))

            dp1bbox_y1x1 = tf.concat([tf.expand_dims(dp1bbox_y1x1i, -2)]*self.most_labels_per_image, -2)
            dp1bbox_y2x2 = tf.concat([tf.expand_dims(dp1bbox_y2x2i, -2)]*self.most_labels_per_image, -2)
            dpg1iou_y1x1 = tf.maximum(dp1bbox_y1x1, g1bbox_y1x1)
            dpg1iou_y2x2 = tf.minimum(dp1bbox_y2x2, g1bbox_y2x2)
            dpg1iou_area = tf.reduce_prod(tf.maximum(dpg1iou_y2x2 - dpg1iou_y1x1, 0), axis=-1)
            dpg1iou_rate = dpg1iou_area / (tf.matmul(dp1bbox_hw, g1bbox_hw, transpose_b=True) - dpg1iou_area)
            noobject_mask1 = tf.cast(dpg1iou_rate <= 0.6, tf.float32)
            p1conf = tf.concat([p1confi]*self.most_labels_per_image, axis=-1)

            dp2bbox_y1x1 = tf.concat([tf.expand_dims(dp2bbox_y1x1i, -2)]*self.most_labels_per_image, -2)
            dp2bbox_y2x2 = tf.concat([tf.expand_dims(dp2bbox_y2x2i, -2)]*self.most_labels_per_image, -2)
            dpg2iou_y1x1 = tf.maximum(dp2bbox_y1x1, g2bbox_y1x1)
            dpg2iou_y2x2 = tf.minimum(dp2bbox_y2x2, g2bbox_y2x2)
            dpg2iou_area = tf.reduce_prod(tf.maximum(dpg2iou_y2x2 - dpg2iou_y1x1, 0), axis=-1)
            dpg2iou_rate = dpg2iou_area / (tf.matmul(dp2bbox_hw, g2bbox_hw, transpose_b=True) - dpg2iou_area)
            noobject_mask2 = tf.cast(dpg2iou_rate <= 0.6, tf.float32)
            p2conf = tf.concat([p2confi]*self.most_labels_per_image, axis=-1)

            dp3bbox_y1x1 = tf.concat([tf.expand_dims(dp3bbox_y1x1i, -2)]*self.most_labels_per_image, -2)
            dp3bbox_y2x2 = tf.concat([tf.expand_dims(dp3bbox_y2x2i, -2)]*self.most_labels_per_image, -2)
            dpg3iou_y1x1 = tf.maximum(dp3bbox_y1x1, g3bbox_y1x1)
            dpg3iou_y2x2 = tf.minimum(dp3bbox_y2x2, g3bbox_y2x2)
            dpg3iou_area = tf.reduce_prod(tf.maximum(dpg3iou_y2x2 - dpg3iou_y1x1, 0), axis=-1)
            dpg3iou_rate = dpg3iou_area / (tf.matmul(dp3bbox_hw, g3bbox_hw, transpose_b=True) - dpg3iou_area)
            noobject_mask3 = tf.cast(dpg3iou_rate <= 0.6, tf.float32)
            p3conf = tf.concat([p3confi]*self.most_labels_per_image, axis=-1)

            noobj_loss1 = tf.reduce_sum((1. - detectors_mask1) * noobject_mask1 * tf.square(p1conf))
            if self.rescore_confidence:
                obj_loss1 = tf.reduce_sum(detectors_mask1 * tf.square(dpg1iou_rate - p1conf))
            else:
                obj_loss1 = tf.reduce_sum(detectors_mask1 * tf.square(1. - p1conf))
            conf_loss1 = self.noobj_scale * noobj_loss1 + self.obj_scale * obj_loss1

            noobj_loss2 = tf.reduce_sum((1. - detectors_mask2) * noobject_mask2 * tf.square(p2conf))
            if self.rescore_confidence:
                obj_loss2 = tf.reduce_sum(detectors_mask2 * tf.square(dpg2iou_rate - p2conf))
            else:
                obj_loss2 = tf.reduce_sum(detectors_mask2 * tf.square(1. - p2conf))
            conf_loss2 = self.noobj_scale * noobj_loss2 + self.obj_scale * obj_loss2

            noobj_loss3 = tf.reduce_sum((1. - detectors_mask3) * noobject_mask3 * tf.square(p3conf))
            if self.rescore_confidence:
                obj_loss3 = tf.reduce_sum(detectors_mask3 * tf.square(dpg3iou_rate - p3conf))
            else:
                obj_loss3 = tf.reduce_sum(detectors_mask3 * tf.square(1. - p3conf))
            conf_loss3 = self.noobj_scale * noobj_loss3 + self.obj_scale * obj_loss3

            p1bbox_xy = tf.concat([tf.expand_dims(p1bbox_xy, -2)]*self.most_labels_per_image, -2)
            p1bbox_hw = tf.concat([tf.expand_dims(p1bbox_hw, -2)]*self.most_labels_per_image, -2)
            ng1bbox_xy = tf.concat([tf.expand_dims(g1bbox_xy/downsampling_rate1, -3)]*self.num_priors, -3)
            ng1bbox_hw = tf.concat([tf.expand_dims(tf.log(g1bbox_hw), -3)]*self.num_priors, -3) / tf.expand_dims(self.anchor_boxes_priors[0], -2)
            coord_loss1 = self.coord_sacle * tf.reduce_sum(
                tf.expand_dims(detectors_mask1, -1) * tf.square(p1bbox_xy - ng1bbox_xy) +
                tf.expand_dims(detectors_mask1, -1) * tf.square(tf.sqrt(p1bbox_hw) - tf.sqrt(ng1bbox_hw))
            )

            p2bbox_xy = tf.concat([tf.expand_dims(p2bbox_xy, -2)]*self.most_labels_per_image, -2)
            p2bbox_hw = tf.concat([tf.expand_dims(p2bbox_hw, -2)]*self.most_labels_per_image, -2)
            ng2bbox_xy = tf.concat([tf.expand_dims(g2bbox_xy/downsampling_rate2, -3)]*self.num_priors, -3)
            ng2bbox_hw = tf.concat([tf.expand_dims(tf.log(g2bbox_hw), -3)]*self.num_priors, -3) / tf.expand_dims(self.anchor_boxes_priors[1], -2)
            coord_loss2 = self.coord_sacle * tf.reduce_sum(
                tf.expand_dims(detectors_mask2, -1) * tf.square(p2bbox_xy - ng2bbox_xy) +
                tf.expand_dims(detectors_mask2, -1) * tf.square(tf.sqrt(p2bbox_hw) - tf.sqrt(ng2bbox_hw))
            )

            p3bbox_xy = tf.concat([tf.expand_dims(p3bbox_xy, -2)]*self.most_labels_per_image, -2)
            p3bbox_hw = tf.concat([tf.expand_dims(p3bbox_hw, -2)]*self.most_labels_per_image, -2)
            ng3bbox_xy = tf.concat([tf.expand_dims(g3bbox_xy/downsampling_rate3, -3)]*self.num_priors, -3)
            ng3bbox_hw = tf.concat([tf.expand_dims(tf.log(g3bbox_hw), -3)]*self.num_priors, -3) / tf.expand_dims(self.anchor_boxes_priors[2], -2)
            coord_loss3 = self.coord_sacle * tf.reduce_sum(
                tf.expand_dims(detectors_mask3, -1) * tf.square(p3bbox_xy - ng3bbox_xy) +
                tf.expand_dims(detectors_mask3, -1) * tf.square(tf.sqrt(p3bbox_hw) - tf.sqrt(ng3bbox_hw))
            )

            p1class = tf.concat([tf.expand_dims(p1classi, -2)]*self.most_labels_per_image, -2)
            g1class = tf.expand_dims(self.labels, 1)
            g1class = tf.expand_dims(g1class, 1)
            g1class = tf.expand_dims(g1class, 1)
            g1class = tf.concat([g1class]*p1shape[1], 1)
            g1class = tf.concat([g1class]*p1shape[2], 2)
            g1class = tf.concat([g1class]*self.num_priors, 3)
            class_loss1 = self.classifier_scale * tf.reduce_sum(
                tf.expand_dims(detectors_mask1, -1) * tf.square(g1class - p1class)
            )

            p2class = tf.concat([tf.expand_dims(p2classi, -2)]*self.most_labels_per_image, -2)
            g2class = tf.expand_dims(self.labels, 1)
            g2class = tf.expand_dims(g2class, 1)
            g2class = tf.expand_dims(g2class, 1)
            g2class = tf.concat([g2class]*p2shape[1], 1)
            g2class = tf.concat([g2class]*p2shape[2], 2)
            g2class = tf.concat([g2class]*self.num_priors, 3)
            class_loss2 = self.classifier_scale * tf.reduce_sum(
                tf.expand_dims(detectors_mask2, -1) * tf.square(g2class - p2class)
            )

            p3class = tf.concat([tf.expand_dims(p3classi, -2)]*self.most_labels_per_image, -2)
            g3class = tf.expand_dims(self.labels, 1)
            g3class = tf.expand_dims(g3class, 1)
            g3class = tf.expand_dims(g3class, 1)
            g3class = tf.concat([g3class]*p3shape[1], 1)
            g3class = tf.concat([g3class]*p3shape[2], 2)
            g3class = tf.concat([g3class]*self.num_priors, 3)
            class_loss3 = self.classifier_scale * tf.reduce_sum(
                tf.expand_dims(detectors_mask3, -1) * tf.square(g3class - p3class)
            )

            total_loss1 = conf_loss1 + coord_loss1 + class_loss1
            total_loss2 = conf_loss2 + coord_loss2 + class_loss2
            total_loss3 = conf_loss3 + coord_loss3 + class_loss3

            total_loss = total_loss1 + total_loss2 + total_loss3
            total_loss = total_loss / self.batch_size

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
        init_conv = self._conv_layer(image, 32, 3, 1)
        block1 = self._darknet_block(init_conv, 64, 1, 'block1')
        block2 = self._darknet_block(block1, 128, 2, 'block2')
        block3 = self._darknet_block(block2, 256, 8, 'block3')
        block4 = self._darknet_block(block3, 512, 8, 'block4')
        block5 = self._darknet_block(block4, 1024, 4, 'block5')
        return block5, block4, block3

    def _yolo3_header(self, bottom, filters, scope, pyramid=None):
            with tf.variable_scope(scope):
                if pyramid is not None:
                    if self.data_format == 'channels_last':
                        axes = 3
                        shape = [int(bottom.get_shape()[1]), int(bottom.get_shape()[2])]
                    else:
                        axes = 1
                        shape = [int(bottom.get_shape()[2]), int(bottom.get_shape()[3])]
                    conv = self._conv_layer(pyramid, filters, 1, 1, False)
                    conv = tf.image.resize_nearest_neighbor(conv, shape)
                    conv = tf.concat([bottom, conv], axis=axes)
                else:
                    conv = bottom
                conv1 = self._conv_layer(conv, filters/2, 1, 1)
                conv2 = self._conv_layer(conv1, filters, 3, 1)
                conv3 = self._conv_layer(conv2, filters/2, 1, 1)
                conv4 = self._conv_layer(conv3, filters, 3, 1)
                conv5 = self._conv_layer(conv4, filters/2, 1, 1)
                conv6 = self._conv_layer(conv5, filters, 3, 1)
                pred = self._conv_layer(conv6, self.final_units, 1, 1)
                return pred, conv5

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
        feed_dict = {self.images: images,
                     self.pretraining_labels: labels['pretraining_labels'],
                     self.lr: 0.}
        loss, acc, summaries = sess_.run([self.loss, self.pretraining_accuracy], feed_dict=feed_dict)
        return loss, acc

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
        assert mode in ['pretraining_latest', 'pretraining_best', 'detection_latest', 'detection_best']
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

    def _darknet_block(self, bottom, filters, blocks, scope):
        with tf.variable_scope(scope):
            conv = self._conv_layer(bottom, filters, 3, 2)
            for i in range(1, blocks+1):
                conv1 = self._conv_layer(conv, filters/2, 1, 1)
                conv2 = self._conv_layer(conv1, filters, 3, 1)
                conv = conv + conv2
            return conv

    def _conv_layer(self, bottom, filters, kernel_size, strides, is_activation=True):
        conv = tf.layers.conv2d(
            inputs=bottom,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='same',
            data_format=self.data_format,
            kernel_initializer=tf.contrib.layers.variance_scaling_initializer()
        )
        bn = self._bn(conv)
        if is_activation:
            bn = tf.nn.leaky_relu(bn, 0.1)
        return bn

    def _bn(self, bottom):
        bn = tf.layers.batch_normalization(
            inputs=bottom,
            axis=3 if self.data_format == 'channels_last' else 1,
            training=self.is_training
        )
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

import tensorflow as tf
from tensorflow.contrib.slim.nets import resnet_v2 as resnet_v2
from models.layers import conv_layer, max_pool, conv_bn_relu
from models.nn import DetectNet
import os
import numpy as np

slim = tf.contrib.slim

class YOLO(DetectNet):
    """YOLO class"""

    def __init__(self, input_shape, num_classes, anchors, **kwargs):

        self.grid_size = grid_size = [x // 32 for x in input_shape[:2]]
        self.num_anchors = len(anchors)
        self.anchors = anchors
        self.y = tf.placeholder(tf.float32, [None] +
                                [self.grid_size[0], self.grid_size[1], self.num_anchors, 5 + num_classes])
        super(YOLO, self).__init__(input_shape, num_classes, **kwargs)

    def _build_model(self, **kwargs):
        """
        Build model.
        :param kwargs: dict, extra arguments for building YOLO.
                -image_mean: np.ndarray, mean image for each input channel, shape: (C,).
        :return d: dict, containing outputs on each layer.
        """

        d = dict()
        x_mean = kwargs.pop('image_mean', 0.0)
        pretrain = kwargs.pop('pretrain', False)
        frontend = kwargs.pop('frontend', 'resnet_v2_50')

        # input
        X_input = self.X - x_mean
        is_train = self.is_train

        # Feature Extractor
        if pretrain:
            frontend_dir = os.path.join('pretrained_models', '{}.ckpt'.format(frontend))
            with slim.arg_scope(resnet_v2.resnet_arg_scope()):
                logits, end_points = resnet_v2.resnet_v2_50(self.X, is_training=self.is_train)
                d['init_fn'] = slim.assign_from_checkpoint_fn(model_path=frontend_dir,
                                                          var_list=slim.get_model_variables(frontend))
            convs = [end_points[frontend + '/block{}'.format(x)] for x in [4, 2, 1]]
            d['conv_s32'] = convs[0]
            d['conv_s16'] = convs[1]
        else:
            # Build ConvNet
            #conv1 - batch_norm1 - leaky_relu1 - pool1
            with tf.variable_scope('layer1'):
                d['conv1'] = conv_bn_relu(X_input, 32, (3, 3), is_train)
                d['pool1'] = max_pool(d['conv1'], 2, 2, padding='SAME')
            # (416, 416, 3) --> (208, 208, 32)

            #conv2 - batch_norm2 - leaky_relu2 - pool2
            with tf.variable_scope('layer2'):
                d['conv2'] = conv_bn_relu(d['pool1'], 64, (3, 3), is_train)
                d['pool2'] = max_pool(d['conv2'], 2, 2, padding='SAME')
            # (208, 208, 32) --> (104, 104, 64)

            #conv3 - batch_norm3 - leaky_relu3
            with tf.variable_scope('layer3'):
                d['conv3'] = conv_bn_relu(d['pool2'], 128, (3, 3), is_train)
            # (104, 104, 64) --> (104, 104, 128)

            #conv4 - batch_norm4 - leaky_relu4
            with tf.variable_scope('layer4'):
                d['conv4'] = conv_bn_relu(d['conv3'], 64, (1, 1), is_train)
            # (104, 104, 128) --> (104, 104, 64)

            #conv5 - batch_norm5 - leaky_relu5 - pool5
            with tf.variable_scope('layer5'):
                d['conv5'] = conv_bn_relu(d['conv4'], 128, (3, 3), is_train)
                d['pool5'] = max_pool(d['conv5'], 2, 2, padding='SAME')
            # (104, 104, 64) --> (52, 52, 128)

            #conv6 - batch_norm6 - leaky_relu6
            with tf.variable_scope('layer6'):
                d['conv6'] = conv_bn_relu(d['pool5'], 256, (3, 3), is_train)
            # (52, 52, 128) --> (52, 52, 256)

            #conv7 - batch_norm7 - leaky_relu7
            with tf.variable_scope('layer7'):
                d['conv7'] = conv_bn_relu(d['conv6'], 128, (1, 1), is_train)
            # (52, 52, 256) --> (52, 52, 128)

            #conv8 - batch_norm8 - leaky_relu8 - pool8
            with tf.variable_scope('layer8'):
                d['conv8'] = conv_bn_relu(d['conv7'], 256, (3, 3), is_train)
                d['pool8'] = max_pool(d['conv8'], 2, 2, padding='SAME')
            # (52, 52, 128) --> (26, 26, 256)

            #conv9 - batch_norm9 - leaky_relu9
            with tf.variable_scope('layer9'):
                d['conv9'] = conv_bn_relu(d['pool8'], 512, (3, 3), is_train)
            # (26, 26, 256) --> (26, 26, 512)

            #conv10 - batch_norm10 - leaky_relu10
            with tf.variable_scope('layer10'):
                d['conv10'] = conv_bn_relu(d['conv9'], 256, (1, 1), is_train)
            # (26, 26, 512) --> (26, 26, 256)

            #conv11 - batch_norm11 - leaky_relu11
            with tf.variable_scope('layer11'):
                d['conv11'] = conv_bn_relu(d['conv10'], 512, (3, 3), is_train)
            # (26, 26, 256) --> (26, 26, 512)

            #conv12 - batch_norm12 - leaky_relu12
            with tf.variable_scope('layer12'):
                d['conv12'] = conv_bn_relu(d['conv11'], 256, (1, 1), is_train)
            # (26, 26, 512) --> (26, 26, 256)

            #conv13 - batch_norm13 - leaky_relu13 - pool13
            with tf.variable_scope('layer13'):
                d['conv13'] = conv_bn_relu(d['conv12'], 512, (3, 3), is_train)
                d['pool13'] = max_pool(d['conv13'], 2, 2, padding='SAME')
            # (26, 26, 256) --> (13, 13, 512)

            #conv14 - batch_norm14 - leaky_relu14
            with tf.variable_scope('layer14'):
                d['conv14'] = conv_bn_relu(d['pool13'], 1024, (3, 3), is_train)
            # (13, 13, 512) --> (13, 13, 1024)

            #conv15 - batch_norm15 - leaky_relu15
            with tf.variable_scope('layer15'):
                d['conv15'] = conv_bn_relu(d['conv14'], 512, (1, 1), is_train)
            # (13, 13, 1024) --> (13, 13, 512)

            #conv16 - batch_norm16 - leaky_relu16
            with tf.variable_scope('layer16'):
                d['conv16'] = conv_bn_relu(d['conv15'], 1024, (3, 3), is_train)
            # (13, 13, 512) --> (13, 13, 1024)

            #conv17 - batch_norm16 - leaky_relu17
            with tf.variable_scope('layer17'):
                d['conv17'] = conv_bn_relu(d['conv16'], 512, (1, 1), is_train)
            # (13, 13, 1024) --> (13, 13, 512)

            #conv18 - batch_norm18 - leaky_relu18
            with tf.variable_scope('layer18'):
                d['conv18'] = conv_bn_relu(d['conv17'], 1024, (3, 3), is_train)
            # (13, 13, 512) --> (13, 13, 1024)

            #conv19 - batch_norm19 - leaky_relu19
            with tf.variable_scope('layer19'):
                d['conv19'] = conv_bn_relu(d['conv18'], 1024, (3, 3), is_train)
            # (13, 13, 1024) --> (13, 13, 1024)
            d['conv_s32'] = d['conv19']
            d['conv_s16'] = d['conv13']

        #Detection Layer
        #conv20 - batch_norm20 - leaky_relu20
        with tf.variable_scope('layer20'):
            d['conv20'] = conv_bn_relu(d['conv_s32'], 1024, (3, 3), is_train)
        # (13, 13, 1024) --> (13, 13, 1024)

        # concatenate layer20 and layer 13 using space to depth
        with tf.variable_scope('layer21'):
            d['skip_connection'] = conv_bn_relu(d['conv_s16'], 64, (1, 1), is_train)
            d['skip_space_to_depth_x2'] = tf.space_to_depth(
                d['skip_connection'], block_size=2)
            d['concat21'] = tf.concat(
                [d['skip_space_to_depth_x2'], d['conv20']], axis=-1)
        # (13, 13, 1024) --> (13, 13, 256+1024)

        #conv22 - batch_norm22 - leaky_relu22
        with tf.variable_scope('layer22'):
            d['conv22'] = conv_bn_relu(d['concat21'], 1024, (3, 3), is_train)
        # (13, 13, 1280) --> (13, 13, 1024)

        output_channel = self.num_anchors * (5 + self.num_classes)
        d['logits'] = conv_layer(d['conv22'], output_channel, (1, 1), (1, 1),
                                padding='SAME', use_bias=True)
        d['pred'] = tf.reshape(
            d['logits'], (-1, self.grid_size[0], self.grid_size[1], self.num_anchors, 5 + self.num_classes))
        # (13, 13, 1024) --> (13, 13, num_anchors , (5 + num_classes))
        return d

    def _build_loss(self, **kwargs):
        """
        Build loss function for the model training.
        :param kwargs: dict, extra arguments
                - loss_weights: list, [xy, wh, resp_confidence, no_resp_confidence, class_probs]
        :return tf.Tensor.
        """

        loss_weights = kwargs.pop('loss_weights', [5, 5, 5, 0.5, 1.0])
        # DEBUG
        # loss_weights = kwargs.pop('loss_weights', [1.0, 1.0, 1.0, 1.0, 1.0])
        with tf.variable_scope('losses'):
            grid_h, grid_w = self.grid_size
            num_classes = self.num_classes
            anchors = self.anchors
            grid_wh = np.reshape([grid_w, grid_h], [
                                 1, 1, 1, 1, 2]).astype(np.float32)
            cxcy = np.transpose([np.tile(np.arange(grid_w), grid_h),
                                 np.repeat(np.arange(grid_h), grid_w)])
            cxcy = np.reshape(cxcy, (1, grid_h, grid_w, 1, 2))

            txty, twth = self.pred[..., 0:2], self.pred[..., 2:4]
            confidence = tf.sigmoid(self.pred[..., 4:5])
            class_probs = tf.nn.softmax(
                self.pred[..., 5:], axis=-1) if num_classes > 1 else tf.sigmoid(self.pred[..., 5:])
            bxby = tf.sigmoid(txty) + cxcy
            pwph = np.reshape(anchors, (1, 1, 1, self.num_anchors, 2)) / 32
            bwbh = tf.exp(twth) * pwph

            # calculating for prediction
            nxny, nwnh = bxby / grid_wh, bwbh / grid_wh
            nx1ny1, nx2ny2 = nxny - 0.5 * nwnh, nxny + 0.5 * nwnh
            self.pred_y = tf.concat(
                (nx1ny1, nx2ny2, confidence, class_probs), axis=-1)

            # calculating IoU for metric
            num_objects = tf.reduce_sum(self.y[..., 4:5], axis=[1, 2, 3, 4])
            max_nx1ny1 = tf.maximum(self.y[..., 0:2], nx1ny1)
            min_nx2ny2 = tf.minimum(self.y[..., 2:4], nx2ny2)
            intersect_wh = tf.maximum(min_nx2ny2 - max_nx1ny1, 0.0)
            intersect_area = tf.reduce_prod(intersect_wh, axis=-1)
            intersect_area = tf.where(
                tf.equal(intersect_area, 0.0), tf.zeros_like(intersect_area), intersect_area)
            gt_box_area = tf.reduce_prod(
                self.y[..., 2:4] - self.y[..., 0:2], axis=-1)
            box_area = tf.reduce_prod(nx2ny2 - nx1ny1, axis=-1)
            iou = tf.truediv(
                intersect_area, (gt_box_area + box_area - intersect_area))
            sum_iou = tf.reduce_sum(iou, axis=[1, 2, 3])
            self.iou = tf.truediv(sum_iou, num_objects)

            gt_bxby = 0.5 * (self.y[..., 0:2] + self.y[..., 2:4]) * grid_wh
            gt_bwbh = (self.y[..., 2:4] - self.y[..., 0:2]) * grid_wh

            resp_mask = self.y[..., 4:5]
            no_resp_mask = 1.0 - resp_mask
            # gt_confidence = resp_mask * tf.expand_dims(iou, axis=-1)
            gt_confidence = resp_mask
            gt_class_probs = self.y[..., 5:]

            loss_bxby = loss_weights[0] * resp_mask * \
                tf.square(gt_bxby - bxby)
            loss_bwbh = loss_weights[1] * resp_mask * \
                tf.square(tf.sqrt(gt_bwbh) - tf.sqrt(bwbh))
            loss_resp_conf = loss_weights[2] * resp_mask * \
                tf.square(gt_confidence - confidence)
            loss_no_resp_conf = loss_weights[3] * no_resp_mask * \
                tf.square(gt_confidence - confidence)
            loss_class_probs = loss_weights[4] * resp_mask * \
                tf.square(gt_class_probs - class_probs)

            merged_loss = tf.concat((
                                    loss_bxby,
                                    loss_bwbh,
                                    loss_resp_conf,
                                    loss_no_resp_conf,
                                    loss_class_probs
                                    ),
                                    axis=-1)
            #self.merged_loss = merged_loss
            total_loss = tf.reduce_sum(merged_loss, axis=-1)
            total_loss = tf.reduce_mean(total_loss)
        return total_loss

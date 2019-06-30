#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-5-21 上午11:38
# @Author  : Luo Yao
# @Site    : http://icode.baidu.com/repos/baidu/personal-code/Luoyao
# @File    : lanenet_hnet_model.py
# @IDE: PyCharm Community Edition
"""
LaneNet中的HNet模型
"""
import tensorflow as tf

from encoder_decoder_model import cnn_basenet
from lanenet_model import hnet_loss


class HNet(cnn_basenet.CNNBaseModel):
    """
    实现lanenet中的hnet模型
    """
    def __init__(self, is_training):
        """

        :param phase:
        """
        super(HNet, self).__init__()
        self._is_training = is_training

        return

    def _conv_stage(self, inputdata, out_channel, name):
        """

        :param inputdata:
        :param out_channel:
        :param name:
        :return:
        """
        with tf.variable_scope(name):
            conv = self.conv2d(inputdata=inputdata, out_channel=out_channel, kernel_size=3, use_bias=False, name='conv')
            bn = self.layerbn(inputdata=conv, is_training=self._is_training, name='bn')
            relu = self.relu(inputdata=bn, name='relu')

        return relu

    def _build_model(self, input_tensor, name):
        """

        :param input_tensor:
        :param name:
        :return:
        """
        print(input_tensor)
        with tf.variable_scope(name):
            conv_stage_1 = self._conv_stage(inputdata=input_tensor, out_channel=16, name='conv_stage_1')
            conv_stage_2 = self._conv_stage(inputdata=conv_stage_1, out_channel=16, name='conv_stage_2')
            maxpool_1 = self.maxpooling(inputdata=conv_stage_2, kernel_size=2, stride=2, name='maxpool_1')
            conv_stage_3 = self._conv_stage(inputdata=maxpool_1, out_channel=32, name='conv_stage_3')
            conv_stage_4 = self._conv_stage(inputdata=conv_stage_3, out_channel=32, name='conv_stage_4')
            maxpool_2 = self.maxpooling(inputdata=conv_stage_4, kernel_size=2, stride=2, name='maxpool_2')
            conv_stage_5 = self._conv_stage(inputdata=maxpool_2, out_channel=64, name='conv_stage_5')
            conv_stage_6 = self._conv_stage(inputdata=conv_stage_5, out_channel=64, name='conv_stage_6')
            maxpool_3 = self.maxpooling(inputdata=conv_stage_6, kernel_size=2, stride=2, name='maxpool_3')
            fc = self.fullyconnect(inputdata=maxpool_3, out_dim=1024, use_bias=False, name='fc')
            bn = self.layerbn(inputdata=fc, is_training=self._is_training, name='bn')
            fc_relu = self.relu(inputdata=bn, name='fc_relu')
            output = self.fullyconnect(inputdata=fc_relu, out_dim=6, name='fc_output')

        return output

    def compute_loss(self, input_tensor, gt_label_pts, name):
        """
        计算hnet损失函数
        :param input_tensor: 原始图像[n, h, w, c]
        :param gt_label_pts: 原始图像对应的标签点集[x, y, 1]
        :param name:
        :return:
        """
        with tf.variable_scope(name):
            pre_H = tf.constant([-2.04835137e-01, -3.09995252e+00, 7.99098762e+01, -2.94687413e+00, 7.06836681e+01, -4.67392998e-02], shape=[6,], dtype=tf.float32)
            transformation_coefficient = self._build_model(input_tensor, name='transfomation_coefficient')
            pre_loss = tf.reduce_mean(tf.norm((transformation_coefficient - pre_H) / pre_H))
            loss = hnet_loss.hnet_loss(gt_pts=gt_label_pts,
                                               transformation_coeffcient=transformation_coefficient,
                                               name='hnet_loss')

            return loss, transformation_coefficient, pre_loss

    def inference(self, input_tensor, gt_label_pts, name):
        """

        :param input_tensor:
        :param name:
        :return:
        """
        with tf.variable_scope(name):
            ceof = self._build_model(input_tensor, name='transfomation_coefficient')
            return hnet_loss.hnet_transformation(gt_label_pts, ceof, 'inference')


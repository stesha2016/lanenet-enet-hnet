#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-5-21 下午1:12
# @Author  : Luo Yao
# @Site    : http://icode.baidu.com/repos/baidu/personal-code/Luoyao
# @File    : lanenet_hnet_loss.py
# @IDE: PyCharm Community Edition
"""
实现LaneNet的HNet损失函数
"""
import tensorflow as tf


def hnet_loss(gt_pts, transformation_coeffcient, name):
    """

    :param gt_pts: 原始的标签点对 [x, y, 1]
    :param transformation_coeffcient: 映射矩阵参数(6参数矩阵) [[a, b, c], [0, d, e], [0, f, 1]]
    :param name:
    :return:
    """
    with tf.variable_scope(name):
        output_ta_loss = tf.TensorArray(dtype=tf.float32,
                                        size=0,
                                        dynamic_size=True)

        def cond(i, labels_tensor, coeffcient, output_ta_loss):
            return i < coeffcient.shape[0]

        def body(i, labels_tensor, coeffcient, output_loss):
            coeffcient_slice = tf.concat([coeffcient[i], [1.0]], axis=-1)
            H_indices = tf.constant([[0], [1], [2], [4], [5], [7], [8]])
            H_shape = tf.constant([9])
            H = tf.scatter_nd(H_indices, coeffcient_slice, H_shape)
            H = tf.reshape(H, shape=[3, 3])
            gt_labels = labels_tensor[i]

            # lane 1
            lane_mask = tf.where(tf.equal(tf.cast(gt_labels[:, 2], tf.int32), 1))
            lane_pts = tf.gather(gt_labels, lane_mask)[:, 0, :]

            lane_pts = tf.transpose(lane_pts)
            lane_trans = tf.matmul(H, lane_pts)

            # 求解最小二乘二阶多项式拟合参数矩阵
            Y = tf.transpose(lane_trans[1, :] / lane_trans[2, :])
            X = tf.transpose(lane_trans[0, :] / lane_trans[2, :])
            Y_One = tf.ones_like(Y, dtype=tf.float32)
            Y_stack = tf.stack([tf.pow(Y, 3), tf.pow(Y, 2), Y, Y_One], axis=1)
            w = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(tf.transpose(Y_stack), Y_stack)),
                                    tf.transpose(Y_stack)), tf.expand_dims(X, -1))
            # 利用二阶多项式参数求解拟合位置并反算到原始投影空间计算损失
            x_preds = tf.matmul(Y_stack, w)
            lane_trans_pred = tf.transpose(tf.stack([tf.squeeze(x_preds, -1) * lane_trans[2, :],
                                                     Y * lane_trans[2, :], lane_trans[2, :]], axis=1))
            lane_trans_back = tf.matmul(tf.matrix_inverse(H), lane_trans_pred)

            loss = tf.reduce_mean(tf.pow(lane_pts[0, :] - lane_trans_back[0, :], 2))

            output_loss = output_loss.write(i, loss)
            return i + 1, labels_tensor, coeffcient, output_loss

        _, _, _, losses = tf.while_loop(cond, body, [0, gt_pts, transformation_coeffcient, output_ta_loss])
        losses = losses.stack()
        loss = tf.reduce_mean(losses)

    return loss


def hnet_transformation(gt_pts, transformation_coeffcient, name):
    """

    :param gt_pts:
    :param transformation_coeffcient:
    :param name:
    :return:
    """
    with tf.variable_scope(name):
        # 首先映射原始标签点对
        transformation_coeffcient = tf.concat([tf.squeeze(transformation_coeffcient), [1.0]], axis=-1)
        multiplier = tf.constant([1., 1., 4., 1., 4., 0.25, 1.])
        transformation_coeffcient = transformation_coeffcient * multiplier
        H_indices = tf.constant([[0], [1], [2], [4], [5], [7], [8]])
        H_shape = tf.constant([9])
        H = tf.scatter_nd(H_indices, transformation_coeffcient, H_shape)
        H = tf.reshape(H, shape=[3, 3])

        gt_pts = tf.transpose(gt_pts)
        pts_projects = tf.matmul(H, gt_pts)

        # 求解最小二乘二阶多项式拟合参数矩阵
        Y = tf.transpose(pts_projects[1, :] / pts_projects[2, :])
        X = tf.transpose(pts_projects[0, :] / pts_projects[2, :])
        Y_One = tf.ones_like(Y)
        Y_stack = tf.stack([tf.pow(Y, 3), tf.pow(Y, 2), Y, Y_One], axis=1)
        w = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(tf.transpose(Y_stack), Y_stack)),
                                tf.transpose(Y_stack)), tf.expand_dims(X, -1))

        # 利用二阶多项式参数求解拟合位置
        x_preds = tf.matmul(Y_stack, w)
        preds = tf.transpose(tf.stack([tf.squeeze(x_preds, -1) * pts_projects[2, :],
                                       Y * pts_projects[2, :], pts_projects[2, :]], axis=1))
        x_transformation_back = tf.matmul(tf.matrix_inverse(H), preds)

    return x_transformation_back, H

def test():
    import numpy as np
    labels = [[401, 260, 1], [427, 270, 1], [441, 280, 1], [434, 290, 1], [412, 300, 1], [390, 310, 1], [368, 320, 1], [347, 330, 1], [325, 340, 1], [303, 350, 1], [277, 360, 1], [247, 370, 1], [216, 380, 1], [185, 390, 1], [154, 400, 1], [124, 410, 1], [94, 420, 1], [64, 430, 1], [34, 440, 1], [4, 450, 1], [507, 270, 2], [521, 280, 2], [530, 290, 2], [539, 300, 2], [539, 310, 2], [538, 320, 2], [537, 330, 2], [536, 340, 2], [534, 350, 2], [530, 360, 2], [521, 370, 2], [512, 380, 2], [504, 390, 2], [495, 400, 2], [486, 410, 2], [478, 420, 2], [469, 430, 2], [460, 440, 2], [452, 450, 2], [443, 460, 2], [434, 470, 2], [426, 480, 2], [417, 490, 2], [408, 500, 2], [400, 510, 2], [391, 520, 2], [382, 530, 2], [374, 540, 2], [365, 550, 2], [355, 560, 2], [346, 570, 2], [337, 580, 2], [328, 590, 2], [318, 600, 2], [309, 610, 2], [300, 620, 2], [291, 630, 2], [282, 640, 2], [272, 650, 2], [263, 660, 2], [254, 670, 2], [245, 680, 2], [236, 690, 2], [226, 700, 2], [217, 710, 2], [709, 320, 3], [729, 330, 3], [748, 340, 3], [764, 350, 3], [780, 360, 3], [795, 370, 3], [811, 380, 3], [827, 390, 3], [842, 400, 3], [855, 410, 3], [868, 420, 3], [881, 430, 3], [894, 440, 3], [907, 450, 3], [920, 460, 3], [933, 470, 3], [946, 480, 3], [959, 490, 3], [972, 500, 3], [985, 510, 3], [999, 520, 3], [1012, 530, 3], [1025, 540, 3], [1039, 550, 3], [1053, 560, 3], [1066, 570, 3], [1080, 580, 3], [1094, 590, 3], [1108, 600, 3], [1122, 610, 3], [1135, 620, 3], [1149, 630, 3], [1163, 640, 3], [1177, 650, 3], [1191, 660, 3], [1205, 670, 3], [1218, 680, 3], [1232, 690, 3], [1246, 700, 3], [1260, 710, 3], [726, 290, 4], [777, 300, 4], [817, 310, 4], [858, 320, 4], [897, 330, 4], [935, 340, 4], [974, 350, 4], [1012, 360, 4], [1050, 370, 4], [1087, 380, 4], [1121, 390, 4], [1155, 400, 4], [1189, 410, 4], [1223, 420, 4], [1257, 430, 4]]
    labels = np.array(labels)
    coffecient = tf.constant([[0.58348501, -0.79861236, 2.30343866, -0.09976104, -1.22268307, 2.43086767]],
                             dtype=tf.float32, shape=[6])

    labels_tensor = tf.placeholder(dtype=tf.float32, shape=[None, 3])
    loss = hnet_loss(labels_tensor, coffecient, 'test')


    with tf.Session() as sess:
        _loss = sess.run(loss, feed_dict={labels_tensor:labels})
        print(_loss)

if __name__ == '__main__':
    test()

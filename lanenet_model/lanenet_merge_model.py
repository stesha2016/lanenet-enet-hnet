#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-5-11 下午5:28
# @Author  : Luo Yao
# @Site    : https://github.com/MaybeShewill-CV/lanenet-lane-detection
# @File    : lanenet_merge_model.py
# @IDE: PyCharm Community Edition
"""
实现LaneNet模型
"""
import tensorflow as tf

from encoder_decoder_model import vgg_encoder
from encoder_decoder_model import fcn_decoder
from encoder_decoder_model import dense_encoder
from encoder_decoder_model import cnn_basenet
from encoder_decoder_model import enet_stage
from lanenet_model import lanenet_discriminative_loss


class LaneNet(cnn_basenet.CNNBaseModel):
    """
    实现语义分割模型
    """
    def __init__(self, phase, net_flag='vgg'):
        """
        Init for LaneNet class
        """
        super(LaneNet, self).__init__()
        self._net_flag = net_flag
        self._phase = phase
        if self._net_flag.lower() == 'vgg':
            self._encoder = vgg_encoder.VGG16Encoder(phase=phase)
        elif self._net_flag.lower() == 'dense':
            self._encoder = dense_encoder.DenseEncoder(l=20, growthrate=8,
                                                       with_bc=True,
                                                       phase=phase,
                                                       n=5)
        elif self._net_flag.lower() == 'enet':
            # Need not encoder and decoder for enet.
            return
        self._decoder = fcn_decoder.FCNDecoder(phase=phase)
        return

    def __str__(self):
        """

        :return:
        """
        info = 'Semantic Segmentation use {:s} as basenet to encode'.format(self._net_flag)
        return info

    def _build_model(self, input_tensor, name):
        """
        前向传播过程
        :param input_tensor:
        :param name:
        :return:
        """
        with tf.variable_scope(name):
            if self._net_flag.lower() == 'enet':
                '''
                LaneNet only shares the first two stages(1 and 2) between the two branches, leaving stage 3 of the ENet encoder
                and the full ENet decoder as the backbone of each separate branch.
                '''
                # shared stages
                with tf.variable_scope('LaneNetBase'):
                    initial = enet_stage.iniatial_block(input_tensor, isTraining=self._phase)
                    stage1, pooling_indices_1, inputs_shape_1 = enet_stage.ENet_stage1(initial, isTraining=self._phase)
                    stage2, pooling_indices_2, inputs_shape_2 = enet_stage.ENet_stage2(stage1, isTraining=self._phase)

                # Segmentation branch
                with tf.variable_scope('LaneNetSeg'):
                    segStage3 = enet_stage.ENet_stage3(stage2, isTraining=self._phase)
                    segStage4 = enet_stage.ENet_stage4(segStage3, pooling_indices_2, inputs_shape_2, stage1, isTraining=self._phase)
                    segStage5 = enet_stage.ENet_stage5(segStage4, pooling_indices_1, inputs_shape_1, initial, isTraining=self._phase)
                    segLogits = tf.layers.conv2d_transpose(segStage5, 2, [2, 2], strides=2, padding='same', name='fullconv')

                # Embedding branch
                with tf.variable_scope('LaneNetEm'):
                    emStage3 = enet_stage.ENet_stage3(stage2, isTraining=self._phase)
                    emStage4 = enet_stage.ENet_stage4(emStage3, pooling_indices_2, inputs_shape_2, stage1, isTraining=self._phase)
                    emStage5 = enet_stage.ENet_stage5(emStage4, pooling_indices_1, inputs_shape_1, initial, isTraining=self._phase)
                    emLogits = tf.layers.conv2d_transpose(emStage5, 4, [2, 2], strides=2, padding='same', name='fullconv')

                ret = {
                    'logits': segLogits,
                    'deconv': emLogits
                }

                return ret
            else:
                # first encode
                encode_ret = self._encoder.encode(input_tensor=input_tensor,
                                                  name='encode')

                # second decode
                if self._net_flag.lower() == 'vgg':
                    decode_ret = self._decoder.decode(input_tensor_dict=encode_ret,
                                                      name='decode',
                                                      decode_layer_list=['pool5',
                                                                         'pool4',
                                                                         'pool3'])
                    return decode_ret
                elif self._net_flag.lower() == 'dense':
                    decode_ret = self._decoder.decode(input_tensor_dict=encode_ret,
                                                      name='decode',
                                                      decode_layer_list=['Dense_Block_5',
                                                                         'Dense_Block_4',
                                                                         'Dense_Block_3'])
                    return decode_ret

    def compute_loss(self, input_tensor, binary_label, instance_label, name):
        """
        计算LaneNet模型损失函数
        :param input_tensor:
        :param binary_label:
        :param instance_label:
        :param name:
        :return:
        """
        with tf.variable_scope(name):
            # 前向传播获取logits
            inference_ret = self._build_model(input_tensor=input_tensor, name='inference')
            decode_logits = inference_ret['logits']
            decode_deconv = inference_ret['deconv']

            # 计算二值分割损失函数
            binary_label_plain = tf.reshape(
                binary_label,
                shape=[binary_label.get_shape().as_list()[0] *
                       binary_label.get_shape().as_list()[1] *
                       binary_label.get_shape().as_list()[2]])
            # 加入class weights
            unique_labels, unique_id, counts = tf.unique_with_counts(binary_label_plain)
            counts = tf.cast(counts, tf.float32)
            inverse_weights = tf.divide(1.0,
                                        tf.log(tf.add(tf.divide(tf.constant(1.0), counts),
                                                      tf.constant(1.02))))
            inverse_weights = tf.concat([tf.constant([5.]), inverse_weights[1:]], axis=0)
            inverse_weights = tf.gather(inverse_weights, binary_label)
            binary_segmenatation_loss = tf.losses.sparse_softmax_cross_entropy(
                labels=binary_label, logits=decode_logits, weights=inverse_weights)
            binary_segmenatation_loss = tf.reduce_mean(binary_segmenatation_loss)

            # 计算discriminative loss损失函数
            # 像素嵌入
            pix_embedding = self.conv2d(inputdata=decode_deconv, out_channel=4, kernel_size=1,
                                        use_bias=False, name='pix_embedding_conv')
            pix_embedding = self.relu(inputdata=pix_embedding, name='pix_embedding_relu')
            # 计算discriminative loss
            image_shape = (pix_embedding.get_shape().as_list()[1], pix_embedding.get_shape().as_list()[2])
            disc_loss, l_var, l_dist, l_reg = \
                lanenet_discriminative_loss.discriminative_loss(
                    pix_embedding, instance_label, 4, image_shape, 0.5, 3.0, 1.0, 1.0, 0.001)

            # 合并损失
            l2_reg_loss = tf.constant(0.0, tf.float32)
            for vv in tf.trainable_variables():
                if 'bn' in vv.name or 'batchnorm' in vv.name or 'batch_norm' in vv.name\
                    and 'alpha' in vv.name:
                    continue
                else:
                    l2_reg_loss = tf.add(l2_reg_loss, tf.nn.l2_loss(vv))
            l2_reg_loss *= 0.001
            total_loss = 0.5 * binary_segmenatation_loss + 0.5 * disc_loss + l2_reg_loss
            # total_loss = binary_segmenatation_loss

            ret = {
                'total_loss': total_loss,
                'binary_seg_logits': decode_logits,
                'instance_seg_logits': pix_embedding,
                'binary_seg_loss': binary_segmenatation_loss,
                'discriminative_loss': disc_loss
            }

            return ret

    def inference(self, input_tensor, name):
        """

        :param input_tensor:
        :param name:
        :return:
        """
        with tf.variable_scope(name):
            # 前向传播获取logits
            inference_ret = self._build_model(input_tensor=input_tensor, name='inference')
            # 计算二值分割损失函数
            decode_logits = inference_ret['logits']
            binary_seg_ret = tf.nn.softmax(logits=decode_logits)
            binary_seg_ret = tf.argmax(binary_seg_ret, axis=-1)
            # 计算像素嵌入
            decode_deconv = inference_ret['deconv']
            # 像素嵌入
            pix_embedding = self.conv2d(inputdata=decode_deconv, out_channel=4, kernel_size=1,
                                        use_bias=False, name='pix_embedding_conv')
            pix_embedding = self.relu(inputdata=pix_embedding, name='pix_embedding_relu')

            return binary_seg_ret, pix_embedding


if __name__ == '__main__':
    model = LaneNet(tf.constant('train', dtype=tf.string))
    input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input')
    binary_label = tf.placeholder(dtype=tf.int64, shape=[1, 256, 512, 1], name='label')
    instance_label = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 1], name='label')
    ret = model.compute_loss(input_tensor=input_tensor, binary_label=binary_label,
                             instance_label=instance_label, name='loss')
    for vv in tf.trainable_variables():
        if 'bn' in vv.name:
            continue
        print(vv.name)

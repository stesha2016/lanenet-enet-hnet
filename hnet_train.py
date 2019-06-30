import tensorflow as tf
from lanenet_model import hnet_model
from data_provider import hnet_data_processor
import numpy as np
import cv2
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--phase', type=str, help='The phase is train or pretrain')
parser.add_argument('--pre_hnet_weights', type=str, help='The pre hnet weights path')
parser.add_argument('--hnet_weights', type=str, help='The hnet model weights path')
args = parser.parse_args()

batch_size = 10
tensor_in = tf.placeholder(dtype=tf.float32, shape=[batch_size, 64, 128, 3])
gt_label_pts = tf.placeholder(dtype=tf.float32, shape=[batch_size, 56, 3])

net = hnet_model.HNet(is_training=True)
c_loss, coef, pre_loss = net.compute_loss(tensor_in, gt_label_pts=gt_label_pts, name='hnet')

var_list = tf.trainable_variables()
g_list = tf.global_variables()
bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
var_list += bn_moving_vars
saver = tf.train.Saver(var_list=var_list, max_to_keep=5)

train_dataset = hnet_data_processor.DataSet(glob.glob('./data/tusimple_data/*.json'))

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    pre_optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss=pre_loss, var_list=tf.trainable_variables())
    optimizer = tf.train.AdamOptimizer(learning_rate=0.00005).minimize(loss=c_loss, var_list=tf.trainable_variables())

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # step1: training pre loss to initialize H matrix
    if args.phase == 'pretrain':
        print('Start pre train hnet......')
        if args.pre_hnet_weights:
            saver.restore(sess, args.pre_hnet_weights)
        for epoch in range(20005):
            image, label_pts = train_dataset.next_batch(batch_size)
            image = np.array(image)
            _, loss, coefficient = sess.run([pre_optimizer, pre_loss, coef], feed_dict={tensor_in: image})
            if epoch % 100 == 0:
                print('[{}] pretrain hnet pre loss = {}'.format(epoch, loss))
            if epoch % 1000 == 0:
                predict = coefficient[0]
                R = np.zeros([3, 3], np.float32)
                R[0, 0] = predict[0]
                R[0, 1] = predict[1]
                R[0, 2] = predict[2]
                R[1, 1] = predict[3]
                R[1, 2] = predict[4]
                R[2, 1] = predict[5]
                R[2, 2] = 1
                print(R)
                warp_image = cv2.warpPerspective(image[0], R, dsize=(image[0].shape[1], image[0].shape[0]))
                cv2.imwrite("src.png", image[0])
                cv2.imwrite("ret.png", warp_image)
            if epoch % 5000 == 0:
                saver.save(sess=sess, save_path='./model/hnet/pre_hnet', global_step=epoch)
    elif args.phase == 'train':
        print('Start train hnet......')
        if args.hnet_weights:
            print('restore hnet weights......')
            saver.restore(sess, args.hnet_weights)
        elif args.pre_hnet_weights:
            print('restore pre hnet weights......')
            saver.restore(sess, args.pre_hnet_weights)
        else:
            print('train from scratch without H matrix initialize.')
        for epoch in range(20005):
            image, label_pts = train_dataset.next_batch(batch_size)
            label_pts = np.array(label_pts)
            label_pts[:, :, 0] = label_pts[:, :, 0] * (512. / 1280.) * 0.25
            label_pts[:, :, 1] = label_pts[:, :, 1] * (256. / 720.) * 0.25
            image = np.array(image)
            _, loss, coefficient = sess.run([optimizer, c_loss, coef], feed_dict={tensor_in: image, gt_label_pts: label_pts})
            if epoch % 50 == 0:
                print('epoch[{}], hnet training loss = {}'.format(epoch, loss))
            if epoch % 1000 == 0:
                predict = coefficient[0]
                R = np.zeros([3, 3], np.float32)
                R[0, 0] = predict[0]
                R[0, 1] = predict[1]
                R[0, 2] = predict[2]
                R[1, 1] = predict[3]
                R[1, 2] = predict[4]
                R[2, 1] = predict[5]
                R[2, 2] = 1
                print(R)
                pts = label_pts[0]
                new_pts = []
                for k in range(len(pts)):
                    if pts[k][2] == 1:
                        new_pts.append(pts[k])
                new_pts = np.float32(new_pts)
                new_pts = np.transpose(new_pts, (1, 0))
                print(new_pts)
                trans_pts = np.matmul(R, new_pts)
                trans_pts = trans_pts / trans_pts[2, :]
                print(trans_pts)
                for k in range(len(trans_pts)):
                    cv2.circle(image[0], (trans_pts[0][k], trans_pts[1][k]), 1, (0, 0, 255), 2)
                #warp_image = cv2.warpPerspective(image[0], R, dsize=(image[0].shape[1], image[0].shape[0]))
                cv2.imwrite("src.png", image[0])
                #cv2.imwrite("ret.png", warp_image)
            if epoch % 1000 == 0:
                saver.save(sess=sess, save_path='./model/hnet/hnet', global_step=epoch)
            epoch += 1
    else:
        print('Wrong phase!!!!!!')
import tensorflow as tf

with tf.device('/cpu:0'):
    meta_path = './model/tusimple_lanenet/tusimple_lanenet_enet_2019-04-11-13-59-36.ckpt-2000.meta' # Your .meta file
    output_node_names = ['lanenet_model/inference/LaneNetSeg/fullconv/conv2d_transpose',
                         'lanenet_model/pix_embedding_relu']    # Output nodes

    with tf.Session() as sess:

        # Restore the graph
        saver = tf.train.import_meta_graph(meta_path)

        # Load weights
        saver.restore(sess, './model/tusimple_lanenet/tusimple_lanenet_enet_2019-04-11-13-59-36.ckpt-2000')

        # Freeze the graph
        frozen_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,
            sess.graph_def,
            output_node_names)

        # Save the frozen graph
        with open('./model/tusimple_lanenet/lanenet.pb', 'wb') as f:
            f.write(frozen_graph_def.SerializeToString())
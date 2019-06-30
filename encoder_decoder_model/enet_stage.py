import tensorflow as tf

'''
============================================================================
ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation
============================================================================
Based on the paper: https://arxiv.org/pdf/1606.02147.pdf

LaneNet only shares the first two stages(1 and 2) between the two branchs, leaving stage 3 of ENet encoder
and the full ENet decoder(4 and 5) as the backbone of each separate branch.
So define ENet_stage1/2/3/4/5 for embedding branch and segmentation branch to build their backbone.
'''

REGULAR = 1
DOWNSAMPLING = 2
UPSAMPLING = 3
DILATED = 4
ASYMMETRIC = 5

def prelu(x, scope, decoder=False):
    """
    Performs the parametric relu operation. This implementation is based on:
    https://stackoverflow.com/questions/39975676/how-to-implement-prelu-activation-in-tensorflow
    For the decoder portion, prelu becomes just a normal prelu
    :param x(Tensor): a 4D Tensor that undergoes prelu
    :param scope(str): the string to name your prelu operation's alpha variable.
    :param decoder(bool): if True, prelu becomes a normal relu.
    :return pos + neg / x (Tensor): gives prelu output only during training; otherwise, just return x.
    """
    # If decoder, then perform relu and just return the output
    if decoder:
        return tf.nn.relu(x, name=scope)

    alpha= tf.get_variable(scope + 'alpha', x.get_shape()[-1],
                           initializer=tf.constant_initializer(0.0), dtype=tf.float32)
    # Also same as tf.maximum(alpha*x, x)
    pos = tf.nn.relu(x)
    neg = alpha * (x - abs(x)) * 0.5
    return pos + neg

def spatial_dropout(x, p, seed, scope, is_training=True):
    '''
    Performs a 2D spatial dropout that drops layers instead of individual elements in an input feature map.
    Note that p stands for the probability of dropping, but tf.nn.dropout uses probability of keeping.
    ------------------
    Technical Details
    ------------------
    The noise shape must be of shape [batch_size, 1, 1, num_channels], with the height and width set to 1, because
    it will represent either a 1 or 0 for each layer, and these 1 or 0 integers will be broadcasted to the entire
    dimensions of each layer they interact with such that they can decide whether each layer should be entirely
    'dropped'/set to zero or have its activations entirely kept.
    --------------------------
    INPUTS:
    :param x(Tensor): a 4D Tensor of the input feature map.
    :param p(float): a float representing the probability of dropping a layer
    :param seed(int): an integer for random seeding the random_uniform distribution that runs under tf.nn.relu
    :param scope(str): the string name for naming the spatial_dropout
    :param is_training(bool): to turn on dropout only when training. Optional.
    OUTPUTS:
    :return (Tensor): a 4D Tensor that is in exactly the same size as the input x,
                      with certain layers having their elements all set to 0 (i.e. dropped).
    '''
    if is_training:
        keep_prob = 1.0 - p
        input_shape = x.get_shape().as_list()
        noise_shape = tf.constant(value=[input_shape[0], 1, 1, input_shape[3]])
        output = tf.nn.dropout(x, keep_prob, noise_shape, seed=seed, name=scope)

        return output

    return x

def unpool(updates, mask, k_size=[1, 2, 2, 1], output_shape=None, scope=''):
    '''
    Unpooling function based on the implementation by Panaetius at https://github.com/tensorflow/tensorflow/issues/2169
    INPUTS:
    :param inputs(Tensor): a 4D tensor of shape [batch_size, height, width, num_channels] that represents the input block to be upsampled
    :param mask(Tensor): a 4D tensor that represents the argmax values/pooling indices of the previously max-pooled layer
    :param k_size(list): a list of values representing the dimensions of the unpooling filter.
    :param output_shape(list): a list of values to indicate what the final output shape should be after unpooling
    :param scope(str): the string name to name your scope
    OUTPUTS:
    :return ret(Tensor): the returned 4D tensor that has the shape of output_shape.
    '''
    with tf.variable_scope(scope):
        mask = tf.cast(mask, tf.int32)
        input_shape = tf.shape(updates, out_type=tf.int32)
        #  calculation new shape
        if output_shape is None:
            output_shape = (input_shape[0], input_shape[1] * k_size[1], input_shape[2] * k_size[2], input_shape[3])

        # calculation indices for batch, height, width and feature maps
        one_like_mask = tf.ones_like(mask, dtype=tf.int32)
        batch_shape = tf.concat([[input_shape[0]], [1], [1], [1]], 0)
        batch_range = tf.reshape(tf.range(output_shape[0], dtype=tf.int32), shape=batch_shape)
        b = one_like_mask * batch_range
        y = mask // (output_shape[2] * output_shape[3])
        x = (mask // output_shape[3]) % output_shape[2] #mask % (output_shape[2] * output_shape[3]) // output_shape[3]
        feature_range = tf.range(output_shape[3], dtype=tf.int32)
        f = one_like_mask * feature_range

        # transpose indices & reshape update values to one dimension
        updates_size = tf.size(updates)
        indices = tf.transpose(tf.reshape(tf.stack([b, y, x, f]), [4, updates_size]))
        values = tf.reshape(updates, [updates_size])
        ret = tf.scatter_nd(indices, values, output_shape)
    return ret

def bottleneck(inputs,
               output_depth,
               filter_size,
               regularizer_prob,
               projection_ratio=4,
               type=REGULAR,
               seed=0,
               is_training=True,
               pooling_indices=None,
               output_shape=None,
               dilation_rate=None,
               decoder=False,
               scope='bottleneck'):
    '''
    The bottleneck module has three different kinds of variants:
    1. A regular convolution which you can decide whether or not to downsample.
    2. A dilated convolution, which requires you to have a dilation factor.
    3. An asymmetric convolution that has a decomposed filter size of 5x1 and 1x5 separately.
    INPUTS:
    :param inputs(Tensor): a 4D Tensor of the previous convolutional block of shape [batch_size, height, width, num_channels].
    :param output_depth(int): an integer indicating the output depth of the output convolutional block.
    :param filter_size(int): an integer that gives the height and width of the filter size to use for a regular/dilated convolution.
    :param regularizer_prob(float): the float p that represents the prob of dropping a layer for spatial dropout regularization.
    :param projection_ratio(int): the amount of depth to reduce for initial 1x1 projection. Depth is divided by projection ratio. Default is 4.
    :param seed(int): an integer for the random seed used in the random normal distribution within dropout.
    :param is_training: a boolean value or tensor boolean to indicate whether or not is training. Decides batch_norm and prelu activity.
    :param type(int):
      1,regular:default
      2,downsampling:a max-pool2D layer is added to downsample the spatial sizes.
      3,upsampling:the upsampling bottleneck is activated but requires pooling indices to upsample.
      4,dilated:then dilated convolution is done, but requires a dilation rate to be given.
      5,asymmtric:then asymmetric convolution is done, and the only filter size used here is 5.
    :param pooling_indices(Tensor): the argmax values that are obtained after performing tf.nn.max_pool_with_argmax.
    :param output_shape(list): A list of integers indicating the output shape of the unpooling layer.
    :param dilation_rate(int): the dilation factor for performing atrous convolution/dilated convolution.
    :param decoder(bool): if True, then all the prelus become relus according to ENet author.
    :param scope(str): a string name that names your bottleneck.
    OUTPUTS:
    :return net(Tensor): The convolution block output after a bottleneck
    :return pooling_indices(Tensor): If downsample, then this tensor is produced for use in upooling later.
    :return inputs_shape(list): The shape of the input to the downsampling conv block. For use in unpooling later.
    '''
    #Calculate the depth reduction based on the projection ratio used in 1x1 convolution.
    reduced_depth = int(inputs.get_shape().as_list()[3] / projection_ratio)

    #=============DOWNSAMPLING BOTTLENECK====================
    if type == DOWNSAMPLING:
        #=============MAIN BRANCH=============
        #Just perform a max pooling
        net_main, pooling_indices = tf.nn.max_pool_with_argmax(inputs,
                                                               ksize=[1,2,2,1],
                                                               strides=[1,2,2,1],
                                                               padding='SAME',
                                                               name=scope+'_main_max_pool')

        #First get the difference in depth to pad, then pad with zeros only on the last dimension.
        inputs_shape = inputs.get_shape().as_list()
        depth_to_pad = abs(inputs_shape[3] - output_depth)
        paddings = tf.convert_to_tensor([[0,0], [0,0], [0,0], [0, depth_to_pad]])
        net_main = tf.pad(net_main, paddings=paddings, name=scope+'_main_padding')

        #=============SUB BRANCH==============
        #First projection that has a 2x2 kernel and stride 2
        net = tf.layers.conv2d(inputs, reduced_depth, [2, 2], strides=2, padding='same', name=scope+'_conv1')
        net = tf.layers.batch_normalization(net, training=is_training, name=scope+'_batch_norm1')
        net = prelu(net, scope=scope+'_prelu1', decoder=decoder)

        #Second conv block
        net = tf.layers.conv2d(net, reduced_depth, [filter_size, filter_size], padding='same', name=scope+'_conv2')
        net = tf.layers.batch_normalization(net, training=is_training, name=scope+'_batch_norm2')
        net = prelu(net, scope=scope+'_prelu2', decoder=decoder)

        #Final projection with 1x1 kernel
        net = tf.layers.conv2d(net, output_depth, [1, 1], padding='same', name=scope+'_conv3')
        net = tf.layers.batch_normalization(net, training=is_training, name=scope+'_batch_norm3')
        net = prelu(net, scope=scope+'_prelu3', decoder=decoder)

        #Regularizer
        net = spatial_dropout(net, p=regularizer_prob, seed=seed, scope=scope+'_spatial_dropout')

        #Finally, combine the two branches together via an element-wise addition
        net = tf.add(net, net_main, name=scope+'_add')
        net = prelu(net, scope=scope+'_last_prelu', decoder=decoder)

        #also return inputs shape for convenience later
        return net, pooling_indices, inputs_shape

    #============DILATION CONVOLUTION BOTTLENECK====================
    #Everything is the same as a regular bottleneck except for the dilation rate argument
    elif type == DILATED:
        #Check if dilation rate is given
        if not dilation_rate:
            raise ValueError('Dilation rate is not given.')

        #Save the main branch for addition later
        net_main = inputs

        #First projection with 1x1 kernel (dimensionality reduction)
        net = tf.layers.conv2d(inputs, reduced_depth, [1, 1], name=scope+'_conv1')
        net = tf.layers.batch_normalization(net, training=is_training, name=scope+'_batchnorm1')
        net = prelu(net, scope=scope+'_prelu1', decoder=decoder)

        #Second conv block --- apply dilated convolution here
        net = tf.layers.conv2d(net, reduced_depth, [filter_size, filter_size], padding='same', dilation_rate=dilation_rate, name=scope+'_dilated_conv2')
        net = tf.layers.batch_normalization(net, training=is_training, name=scope+'_batchnorm2')
        net = prelu(net, scope=scope+'_prelu2', decoder=decoder)

        #Final projection with 1x1 kernel (Expansion)
        net = tf.layers.conv2d(net, output_depth, [1,1], name=scope+'_conv3')
        net = tf.layers.batch_normalization(net, training=is_training, name=scope+'_batchnorm3')
        net = prelu(net, scope=scope+'_prelu3', decoder=decoder)

        #Regularizer
        net = spatial_dropout(net, p=regularizer_prob, seed=seed, scope=scope+'_spatial_dropout')
        net = prelu(net, scope=scope+'_prelu4', decoder=decoder)

        #Add the main branch
        net = tf.add(net_main, net, name=scope+'_add_dilated')
        net = prelu(net, scope=scope+'_last_prelu', decoder=decoder)

        return net

    #===========ASYMMETRIC CONVOLUTION BOTTLENECK==============
    #Everything is the same as a regular bottleneck except for a [5,5] kernel decomposed into two [5,1] then [1,5]
    elif type == ASYMMETRIC:
        #Save the main branch for addition later
        net_main = inputs

        #First projection with 1x1 kernel (dimensionality reduction)
        net = tf.layers.conv2d(inputs, reduced_depth, [1, 1], name=scope+'_conv1')
        net = tf.layers.batch_normalization(net, training=is_training, name=scope+'_batchnorm1')
        net = prelu(net, scope=scope+'_prelu1', decoder=decoder)

        #Second conv block --- apply asymmetric conv here
        net = tf.layers.conv2d(net, reduced_depth, [filter_size, 1], padding='same', name=scope+'_asymmetric_conv2a')
        net = tf.layers.conv2d(net, reduced_depth, [1, filter_size], padding='same', name=scope+'_asymmetric_conv2b')
        net = tf.layers.batch_normalization(net, training=is_training, name=scope+'_batchnorm2')
        net = prelu(net, scope=scope+'_prelu2', decoder=decoder)

        #Final projection with 1x1 kernel
        net = tf.layers.conv2d(net, output_depth, [1, 1], name=scope+'_conv3')
        net = tf.layers.batch_normalization(net, training=is_training, name=scope+'_batchnorm3')
        net = prelu(net, scope=scope+'_prelu3', decoder=decoder)

        #Regularizer
        net = spatial_dropout(net, p=regularizer_prob, seed=seed, scope=scope+'_spatial_dropout')
        net = prelu(net, scope=scope+'_prelu4', decoder=decoder)

        #Add the main branch
        net = tf.add(net_main, net, name=scope+'_add_asymmetric')
        net = prelu(net, scope=scope+'_last_prelu', decoder=decoder)

        return net

    #============UPSAMPLING BOTTLENECK================
    #Everything is the same as a regular one, except convolution becomes transposed.
    elif type == UPSAMPLING:
        #Check if pooling indices is given
        if pooling_indices == None:
            raise ValueError('Pooling indices are not given.')

        #Check output_shape given or not
        if output_shape == None:
            raise ValueError('Output depth is not given')

        #=======MAIN BRANCH=======
        #Main branch to upsample. output shape must match with the shape of the layer that was pooled initially, in order
        #for the pooling indices to work correctly. However, the initial pooled layer was padded, so need to reduce dimension
        #before unpooling. In the paper, padding is replaced with convolution for this purpose of reducing the depth!
        net_unpool = tf.layers.conv2d(inputs, output_depth, [1, 1], name=scope+'_main_conv1')
        net_unpool = tf.layers.batch_normalization(net_unpool, training=is_training, name=scope+'_batchnorm1')
        net_unpool = unpool(net_unpool, pooling_indices, output_shape=output_shape, scope='unpool')

        #======SUB BRANCH=======
        #First 1x1 projection to reduce depth
        net = tf.layers.conv2d(inputs, reduced_depth, [1, 1], name=scope+'_conv1')
        net = tf.layers.batch_normalization(net, training=is_training, name=scope+'_batchnorm2')
        net = prelu(net, scope=scope+'_prelu1', decoder=decoder)

        #Second conv block -----------------------------> NOTE: using tf.nn.conv2d_transpose for variable input shape.
        net = tf.layers.conv2d_transpose(net, reduced_depth, [filter_size, filter_size], strides=2, padding='same', name=scope+'_transposed_conv2')
        net = tf.layers.batch_normalization(net, training=is_training, name=scope+'_batchnorm3')
        net = prelu(net, scope=scope+'_prelu2', decoder=decoder)

        #Final projection with 1x1 kernel
        net = tf.layers.conv2d(net, output_depth, [1, 1], name=scope+'_conv3')
        net = tf.layers.batch_normalization(net, training=is_training, name=scope+'_batchnorm4')
        net = prelu(net, scope=scope+'_prelu3', decoder=decoder)

        #Regularizer
        net = spatial_dropout(net, p=regularizer_prob, seed=seed, scope=scope+'_spatial_dropout')
        net = prelu(net, scope=scope+'_prelu4', decoder=decoder)

        #Finally, add the unpooling layer and the sub branch together
        net = tf.add(net, net_unpool, name=scope+'_add_upsample')
        net = prelu(net, scope=scope+'_last_prelu', decoder=decoder)

        return net

    #OTHERWISE, just perform a regular bottleneck!
    #==============REGULAR BOTTLENECK==================
    #Save the main branch for addition later
    else:
        net_main = inputs

        #First projection with 1x1 kernel
        net = tf.layers.conv2d(inputs, reduced_depth, [1, 1], name=scope+'_conv1')
        net = tf.layers.batch_normalization(net, training=is_training, name=scope+'_batchnorm1')
        net = prelu(net, scope=scope+'_prelu1', decoder=decoder)

        #Second conv block
        net = tf.layers.conv2d(net, reduced_depth, [filter_size, filter_size], padding='same', name=scope+'_conv2')
        net = tf.layers.batch_normalization(net, training=is_training, name=scope+'_batchnorm2')
        net = prelu(net, scope=scope+'_prelu2', decoder=decoder)

        #Final projection with 1x1 kernel
        net = tf.layers.conv2d(net, output_depth, [1, 1], name=scope+'_conv3')
        net = tf.layers.batch_normalization(net, training=is_training, name=scope+'_batchnorm3')
        net = prelu(net, scope=scope+'_prelu3', decoder=decoder)

        #Regularizer
        net = spatial_dropout(net, p=regularizer_prob, seed=seed, scope=scope+'_spatial_dropout')
        net = prelu(net, scope=scope+'_prelu4', decoder=decoder)

        #Add the main branch
        net = tf.add(net_main, net, name=scope+'_add_regular')
        net = prelu(net, scope=scope+'_last_prelu', decoder=decoder)

        return net

def iniatial_block(inputs, isTraining=True, scope='iniatial_block'):
    '''
    The initial block for Enet has 2 branches: The convolution branch and Maxpool branch.
    The conv branch has 13 filters, while the maxpool branch gives 3 channels corresponding to the RGB channels.
    Both output layers are then concatenated to give an output of 16 channels.

    :param inputs(Tensor): A 4D tensor of shape [batch_size, height, width, channels]
    :return net_concatenated(Tensor): a 4D Tensor of new shape [batch_size, height, width, channels]
    '''
    # Convolutional branch
    net_conv = tf.layers.conv2d(inputs, 13, [3, 3], strides=2, padding='same', name=scope+'_conv')
    net_conv = tf.layers.batch_normalization(net_conv, training=isTraining, name=scope+'_batchnorm')
    net_conv = prelu(net_conv, scope=scope+'_prelu')

    # Max pool branch
    net_pool = tf.layers.max_pooling2d(inputs, [2, 2], strides=2, padding='same', name=scope+'_max_pool')

    # Concatenated output - does it matter max pool comes first or conv comes first? probably not.
    net_concatenated = tf.concat([net_conv, net_pool], axis=3, name=scope+'_concat')
    return net_concatenated

def ENet_stage1(inputs, isTraining=True, scope='stage1_block'):
    '''
    stage 1 is encoder process, with one downsampling and 4 regular bottlenecks
    :param inputs(Tensor): A 4D tensor of shape [batch_size, height, width, channels]
    :param isTraining: bool, is training stage or not
    :param scope: str, scope string
    :return (Tensor): a 4D Tensor of new shape [batch_size, height, width, channels]
    :return (Tensor): max pooling indices.
    :return (List) inputs shape
    '''
    net, pooling_indices_1, inputs_shape_1 \
      = bottleneck(inputs, output_depth=64, filter_size=3, regularizer_prob=0.01, type=DOWNSAMPLING,
                   scope='bottleneck1_0', is_training=isTraining)
    net = bottleneck(net, output_depth=64, filter_size=3, regularizer_prob=0.01,
                     scope='bottleneck1_1', is_training=isTraining)
    net = bottleneck(net, output_depth=64, filter_size=3, regularizer_prob=0.01,
                     scope='bottleneck1_2', is_training=isTraining)
    net = bottleneck(net, output_depth=64, filter_size=3, regularizer_prob=0.01,
                     scope='bottleneck1_3', is_training=isTraining)
    net = bottleneck(net, output_depth=64, filter_size=3, regularizer_prob=0.01,
                     scope='bottleneck1_4', is_training=isTraining)
    return net, pooling_indices_1, inputs_shape_1

def ENet_stage2(inputs, isTraining=True, scope='stage2_block'):
    '''
    stage 2 is encoder process with one downsampling and two loops of regular + dilated + asymmetric + dilated.
    :param inputs(Tensor): A 4D tensor of shape [batch_size, height, width, channels]
    :param isTraining: bool, is training stage or not
    :param scope: str, scope string
    :return (Tensor): a 4D Tensor of new shape [batch_size, height, width, channels]
    :return (Tensor): max pooling indices.
    :return (List) inputs shape
    '''
    net, pooling_indices_2, inputs_shape_2 \
      = bottleneck(inputs, output_depth=128, filter_size=3, regularizer_prob=0.1, type=DOWNSAMPLING,
                   scope='bottleneck2_0', is_training=isTraining)
    for i in range(2):
        net = bottleneck(net, output_depth=128, filter_size=3, regularizer_prob=0.1,
                         scope='bottleneck2_{}'.format(str(4 * i + 1)), is_training=isTraining)
        net = bottleneck(net, output_depth=128, filter_size=3, regularizer_prob=0.1, type=DILATED, dilation_rate=(2 ** (2*i+1)),
                         scope='bottleneck2_{}'.format(str(4 * i + 2)), is_training=isTraining)
        net = bottleneck(net, output_depth=128, filter_size=5, regularizer_prob=0.1, type=ASYMMETRIC,
                         scope='bottleneck2_{}'.format(str(4 * i + 3)), is_training=isTraining)
        net = bottleneck(net, output_depth=128, filter_size=3, regularizer_prob=0.1, type=DILATED, dilation_rate=(2 ** (2*i+2)),
                         scope='bottleneck2_{}'.format(str(4 * i + 4)), is_training=isTraining)
    return net, pooling_indices_2, inputs_shape_2

def ENet_stage3(inputs, isTraining=True, scope='stage3_block'):
    '''
    stage 3 is encoder process, similar with stage2 but without downsampling.
    :param inputs(Tensor): A 4D tensor of shape [batch_size, height, width, channels]
    :param isTraining: bool, is training stage or not
    :param scope: str, scope string
    :return (Tensor): a 4D Tensor of new shape [batch_size, height, width, channels]
    '''
    for i in range(2):
        net = bottleneck(inputs, output_depth=128, filter_size=3, regularizer_prob=0.1,
                         scope='bottleneck3_{}'.format(str(4 * i + 0)), is_training=isTraining)
        net = bottleneck(net, output_depth=128, filter_size=3, regularizer_prob=0.1, type=DILATED, dilation_rate=(2 ** (2*i+1)),
                         scope='bottleneck3_{}'.format(str(4 * i + 1)), is_training=isTraining)
        net = bottleneck(net, output_depth=128, filter_size=5, regularizer_prob=0.1, type=ASYMMETRIC,
                         scope='bottleneck3_{}'.format(str(4 * i + 2)), is_training=isTraining)
        net = bottleneck(net, output_depth=128, filter_size=3, regularizer_prob=0.1, type=DILATED, dilation_rate=(2 ** (2*i+2)),
                         scope='bottleneck3_{}'.format(str(4 * i + 3)), is_training=isTraining)
    return net

def ENet_stage4(inputs, pooling_indices, inputs_shape, connect_tensor, skip_connections=True, isTraining=True, scope='stage4_block'):
    '''
    stage 4 is decoder process, with one upsampling and two regular.
    :param inputs(Tensor): A 4D tensor of shape [batch_size, height, width, channels]
    :param pooling_indices&inputs_shape: for unpool in upsampling
    :param connect_tensor: if skip_connections is true, will add with this tensor
    :param skip_connections: boolean skip connect or not
    :param isTraining: bool, is training stage or not
    :param scope: str, scope string
    :return (Tensor): a 4D Tensor of new shape [batch_size, height, width, channels]
    '''
    net = bottleneck(inputs, output_depth=64, filter_size=3, regularizer_prob=0.1, type=UPSAMPLING, decoder=True,
                     pooling_indices=pooling_indices, output_shape=inputs_shape,
                     scope='bottleneck4_0', is_training=isTraining)

    #Perform skip connections here
    if skip_connections:
        net = tf.add(net, connect_tensor, name='bottleneck4_skip_connection')

    net = bottleneck(net, output_depth=64, filter_size=3, regularizer_prob=0.1, decoder=True,
                     scope='bottleneck4_1', is_training=isTraining)
    net = bottleneck(net, output_depth=64, filter_size=3, regularizer_prob=0.1, decoder=True,
                     scope='bottleneck4_2', is_training=isTraining)
    return net

def ENet_stage5(inputs, pooling_indices, inputs_shape, connect_tensor, skip_connections=True, isTraining=True, scope='stage5_block'):
    '''
    stage 5 is decoder process, similar with stage4.
    :param inputs(Tensor): A 4D tensor of shape [batch_size, height, width, channels]
    :param pooling_indices&inputs_shape: for unpool in upsampling
    :param connect_tensor: if skip_connections is true, will add with this tensor
    :param skip_connections: boolean skip connect or not
    :param isTraining: bool, is training stage or not
    :param scope: str, scope string
    :return (Tensor): a 4D Tensor of new shape [batch_size, height, width, channels]
    '''
    net = bottleneck(inputs, output_depth=16, filter_size=3, regularizer_prob=0.1, type=UPSAMPLING, decoder=True,
                     pooling_indices=pooling_indices, output_shape=inputs_shape,
                     scope='bottleneck5_0', is_training=isTraining)

    #perform skip connections here
    if skip_connections:
        net = tf.add(net, connect_tensor, name='bottleneck5_skip_connection')

    net = bottleneck(net, output_depth=16, filter_size=3, regularizer_prob=0.1, decoder=True,
                     scope='bottleneck5_1', is_training=isTraining)
    return net

#=================================================================================================================

if __name__ == '__main__':
    input_tensor = tf.placeholder(dtype=tf.float32, shape=[16, 512, 512, 3])
    initial = iniatial_block(input_tensor)
    new_one = initial
    print('initial', initial)
    stage1, pooling_indices_1, inputs_shape_1 = ENet_stage1(initial)
    net_two = stage1
    print('stage1', stage1, pooling_indices_1, inputs_shape_1)
    stage2, pooling_indices_2, inputs_shape_2 = ENet_stage2(stage1)
    print('stage2', stage2, pooling_indices_2, inputs_shape_2)
    stage3 = ENet_stage3(stage2)
    print('stage3', stage3)
    stage4 = ENet_stage4(stage3, pooling_indices_2, inputs_shape_2, net_two)
    print('stage4', stage4)
    stage5 = ENet_stage5(stage4, pooling_indices_1, inputs_shape_1, new_one)
    print('stage5', stage5)
    logits = tf.layers.conv2d_transpose(stage5, 4, [2, 2], strides=2, padding='same', name='fullconv')
    probabilities = tf.nn.softmax(logits, name='logits_to_softmax')
    print(logits, probabilities)

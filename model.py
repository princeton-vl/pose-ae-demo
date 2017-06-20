import tensorflow as tf

def cnv(inp, kernel_shape, scope_name, stride=[1,1,1,1], dorelu=True,
        weight_init_fn=tf.random_normal_initializer,
        bias_init_fn=tf.constant_initializer, bias_init_val=0.0, pad='SAME',):

    with tf.variable_scope(scope_name):
        std = 1 / (kernel_shape[0] * kernel_shape[1] * kernel_shape[2])
        std = std ** .5
        weights = tf.get_variable('weights', kernel_shape, 
                                  initializer=weight_init_fn(stddev=std))
        biases = tf.get_variable('biases', [kernel_shape[-1]],
                                  initializer=bias_init_fn(bias_init_val))
        conv = tf.nn.conv2d(inp, weights, strides=stride, padding=pad) + biases
        # Add ReLU
        if dorelu: return tf.nn.relu(conv)
        else: return conv

def pool(inp, name=None, kernel=[2,2], stride=[2,2]):
    # Initialize max-pooling layer (default 2x2 window, stride 2)
    kernel = [1] + kernel + [1]
    stride = [1] + stride + [1]
    return tf.nn.max_pool(inp, kernel, stride, 'SAME', name=name)

def hourglass(inp, n, f, hg_id):
    # Upper branch
    nf = f + 128
    up1 = cnv(inp, [3, 3, f, f], '%d_%d_up1' % (hg_id, n))

    # Lower branch
    pool1 = pool(inp, '%d_%d_pool' % (hg_id, n))
    low1 = cnv(pool1, [3, 3, f, nf], '%d_%d_low1' % (hg_id, n))
    # Recursive hourglass
    if n > 1:
        low2 = hourglass(low1, n - 1, nf, hg_id)
    else:
        low2 = cnv(low1, [3, 3, nf, nf], '%d_%d_low2' % (hg_id, n))
    low3 = cnv(low2, [3, 3, nf, f], '%d_%d_low3' % (hg_id, n))

    up_size = tf.shape(up1)[1:3]
    up2 = tf.image.resize_nearest_neighbor(low3, up_size)
    return up1 + up2

def inference(inp_img, num_output_channel):
    f = 256
    cnv1 = cnv(inp_img, [7, 7, 3, 64], 'cnv1', stride=[1,2,2,1])
    cnv2 = cnv(cnv1, [3, 3, 64, 128], 'cnv2')
    pool1 = pool(cnv2, 'pool1')
    cnv2b = cnv(pool1, [3, 3, 128, 128], 'cnv2b')
    cnv3 = cnv(cnv2b, [3, 3, 128, 128], 'cnv3')
    cnv4 = cnv(cnv3, [3, 3, 128, f], 'cnv4')

    inter = cnv4

    preds = []
    for i in range(4):
        # Hourglass
        hg = hourglass(inter, 4, f, i)

        # Final output
        cnv5 = cnv(hg, [3, 3, f, f], 'cnv5_%d' % i)
        cnv6 = cnv(cnv5, [1, 1, f, f], 'cnv6_%d' % i)
        preds += [cnv(cnv6, [1, 1, f, num_output_channel], 'out_%d' % i, dorelu=False)]

        # Residual link across hourglasses
        if i < 3:
            inter = inter + cnv(cnv6, [1, 1, f, f], 'tmp_%d' % i, dorelu=False)\
            + cnv(preds[-1], [1, 1, num_output_channel, f], 'tmp_out_%d'%i, dorelu = False)
    return preds[-1]

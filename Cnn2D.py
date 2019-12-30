import tensorflow as tf
import tensorflow.contrib.slim as slim
# input size 20 x 6
def SeqCNN2D(input, num_OutputNodes, keep_pro=0.5):
    with tf.variable_scope('C2D'):
        with slim.arg_scope([slim.conv2d],
                            padding='SAME',
                            weights_regularizer=slim.l2_regularizer(0.0005),
                            activation_fn=tf.nn.relu,
                            # kernel_size=[10 ,3 ],
                            kernel_size=[4 ,4 ],
                            stride=[1,1 ]
                            ):
            net = slim.conv2d(input, 32, scope='conv1')
            net = slim.max_pool2d(net, kernel_size=[2, 1], stride=[2, 1], padding='SAME', scope='max_pool1')
            
            net = slim.conv2d(net, 64, scope='conv2')
            net = slim.max_pool2d(net, kernel_size=[2, 1], stride=[2, 1], padding='SAME', scope='max_pool2')
            
            net = slim.repeat(net, 2, slim.conv2d, 128, scope='conv3')
            net = slim.max_pool2d(net, kernel_size=[2, 1], stride=[2, 1], padding='SAME', scope='max_pool3')
            
            net = slim.repeat(net, 2, slim.conv2d, 256, scope='conv4')
            net = slim.max_pool2d(net, kernel_size=[2, 1], stride=[2, 1], padding='SAME', scope='max_pool4')
            
            
            #net = slim.repeat(net, 2, slim.conv2d, 256, scope='conv4')
            
            #net = slim.repeat(net, 2, slim.conv2d, 512, scope='conv5')
            


            net = tf.reshape(net, [-1, 6144]) # 使用kernel_size=[6 ,3 ], stride=[2,1 ] 1536
            
            net = slim.fully_connected (net ,6144, weights_regularizer=slim.l2_regularizer(0.0005), scope='fc1') 
            net = slim.dropout(net, keep_pro, scope='dropout1')
            
            #net = slim.fully_connected(net , 1536, weights_regularizer=slim.l2_regularizer(0.0005), scope='fc2') 
            #net = slim.dropout(net, keep_pro, scope='dropout2')
            
            out = slim.fully_connected(net, num_OutputNodes, weights_regularizer=slim.l2_regularizer(0.0005), \
                                       activation_fn=None, scope='out' )

            return out



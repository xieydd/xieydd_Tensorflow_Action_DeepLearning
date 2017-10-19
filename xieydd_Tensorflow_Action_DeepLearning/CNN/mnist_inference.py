#-*- coding: utf-8 -*-
##-*- coding: utf-8 -*-
#@Description: CNN 7层网络对MNIST的层级的实现
#@author xieydd xieydd@gmail.com  
#@date 2017-10-15 下午15:18:10
import tensorflow as tf

IMPUT_NODE = 784
OUT_NODE = 10

IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10

#第一层卷积层的尺寸和深度
CONV1_SIZE = 5
CONV1_DEEP = 32
#第二层卷积层的尺寸和深度
CONV2_SIZE = 5
CONV2_DEEP = 64
#全连接节点数
FC_SIZE = 512


#在训练时创建这些变量，在测试或者验证的时候使用保存过的模型加载，而且可以在加载变量的时候重命名以便滑动平均变量使用
'''
def get_weight_variable(shape,regularizer):
    weights = tf.get_variable("weights",shape,initializer = tf.truncated_normal_initializer(stddev=0.1))
    #这里可以自定义正则化项,将正则化损失加到loss这个集合中，但这个集合不是Tensorflow自动管理的
    if regularizer != None:
        tf.add_to_collection('losses',regularizer(weights))
    return weights
'''
#向前函数 输出x2 avg_class为是否执行滑动平均 train用于区分训练过程还是测试过程
def inference(input_tensor,train,regularizer):
    #声明命名空间,因为没有多次调用这个函数，所以不需要在参数中设置reuse=True
    with tf.variable_scope('layer1-conv1',reuse=None):
        #这里的卷积层使用全0填充 过滤器长短为1*1深度为32步长为1 所以输入为28*28*1 输出为28*28*32
        conv1_weights = tf.get_variable("weights",[CONV1_SIZE,CONV1_SIZE,NUM_CHANNELS,CONV1_DEEP],initializer = tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases1 = tf.get_variable("biases",[CONV1_DEEP],initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input_tensor,conv1_weights,strides=[1,1,1,1],padding="SAME")
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1,conv1_biases1))

    with tf.variable_scope('layer2-pool1',reuse=None):
        #第二层为池化层，这里使用最大池化层 使用全0填充 过滤器长宽为2*2深度为1 步长为2 输入为28*28*32 输出为14*14*32
        pool1 = tf.nn.max_pool(relu1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    with tf.variable_scope('layer3-conv2',reuse=None):
        conv2_weights = tf.get_variable("weights",[CONV2_SIZE,CONV2_SIZE,CONV1_DEEP,CONV2_DEEP],initializer = tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("biases",[CONV2_DEEP],initializer=tf.constant_initializer(0.0))
        #过滤器深度为64 步长为1
        conv2 = tf.nn.conv2d(pool1,conv2_weights,strides=[1,1,1,1],padding="SAME")
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2,conv2_biases))

    with tf.variable_scope("layer4-pool2",reuse=None):
        #第四层为池化层，这里使用最大池化层 使用全0填充 过滤器长宽为2*2深度为1 步长为2 输入为14*14*64 输出为7*7*64
        pool2 = tf.nn.max_pool(relu2,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

    #第四层输出的是7*7*64需要转换成第五层全连接需要的输入格式向量，这里直接拉长为一个向量，注意这里包含的是整个batch的矩阵 batch个数是pool_shape[0] 长宽深分别为1,2,3
    pool_shape = pool2.get_shape().as_list()
    node = pool_shape[1] * pool_shape[2] * pool_shape[3]
    reshaped = tf.reshape(pool2,[pool_shape[0],node])


    #这里使用DropOut防止过拟合输入长度为3136，输出为512
    with tf.variable_scope("layer5-fc1",reuse=None):
        fc1_weights = tf.get_variable("weights",[node,FC_SIZE],initializer = tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection("losses",regularizer(fc1_weights))
        fc1_biases = tf.get_variable("biases",[FC_SIZE],initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(reshaped,fc1_weights)+fc1_biases)
        if train: fc1 = tf.nn.dropout(fc1,0.5)

    with tf.variable_scope("layer6-fc2",reuse=None):
        fc2_weights = tf.get_variable("weights",[FC_SIZE,NUM_LABELS],initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection("losses",regularizer(fc2_weights))
        fc2_biases = tf.get_variable("biases",[NUM_LABELS],initializer=tf.constant_initializer(0.1))
        logit = tf.matmul(fc1,fc2_weights)+fc2_biases
    return logit

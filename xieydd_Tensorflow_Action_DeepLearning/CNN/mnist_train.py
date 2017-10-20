#-*- coding: utf-8 -*-
#@Description: CNN 7层网络对MNIST的训练过程实现
#@author xieydd xieydd@gmail.com  
#@date 2017-10-15 下午15:17:10
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from mnist_inference import *
import numpy as np
#配置神经网络参数
TRAINING_STEPS = 30000#迭代次数
LEARNING_RATE_BASE = 0.01#基础学习率
LEARNING_RATE_DECAY = 0.95#学习率衰减率
REGULARIZATION_RATE = 0.0001#R2的lambda
KEEP_PROB = tf.placeholder(tf.float32)#如果使用DropOut的参数
BATCH_SIZE = 64#batch数量
MOVING_AVERAGE_DECAY = 0.99#滑动平均衰减率
MODEL_SAVE_PATH = "G:/tensorflow/logs"#模型保存路径
MODEL_NAME = "model.ckpt"#模型保存文件名


def train(mnist,name="train"):
    #由于输入的是四维矩阵，所以修改格式 Batch中样例的个数 长 宽 深度 放在input名下
    with tf.name_scope('input1'):
        x = tf.placeholder(tf.float32,[BATCH_SIZE,IMAGE_SIZE,IMAGE_SIZE,NUM_CHANNELS],name='x-input')
        y_ = tf.placeholder(tf.float32,[None,NUM_LABELS],name='y-input')
    
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    #不使用滑动平均的输出y
    y = inference(x,False,regularizer)
    #使用滑动平均输出y
    #在Tensorflow中一般会将代表训练轮数的变量指定为不可训练的参数
    global_step = tf.Variable(0,trainable=False)
    #平滑处理放在moving_average下
    with tf.name_scope("moving_average"):
        #初始化滑动平均类
        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
        #这里的tf.trainable_variables()是返回图上所有集合相当于GRAPHKEYS.TRAINABLE_VARIABLES，这里讲所有代表神经网络的变量使用滑动平均
        variable_averages_op = variable_averages.apply(tf.trainable_variables())
        
    #计算损失放在loss_function下
    with tf.name_scope("loss_function"):
        cross_entrypy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
        #计算当前batch的平均交叉熵
        cross_entrypy_mean = tf.reduce_mean(cross_entrypy)
        loss = cross_entrypy_mean + tf.add_n(tf.get_collection('losses'))
        
    #将学习率优化方法以及每一轮的操作放在train_step下
    with tf.name_scope("train_step"):
        #设置指数衰减学习率 2,3,4参数分别为当前迭代的论数 过完全部数据需要的轮数 学习衰减速度
        learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,global_step,mnist.train.num_examples/BATCH_SIZE,LEARNING_RATE_DECAY,staircase=True)
        #向后优化
        #这里使用AdamOptimizer()发现效果没有GradientDescentOptimer好
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
        #反向更新参数和滑动平均值
        with tf.control_dependencies([train_step,variable_averages_op]):
            train_op = tf.no_op(name='train')
    #初始化Tensorflow持久化
    # 按需占用GPU内存
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    saver = tf.train.Saver()
    with tf.Session(config=config) as sess:
        tf.global_variables_initializer().run()
        for i in range(TRAINING_STEPS):
            xs,ys = mnist.train.next_batch(BATCH_SIZE)
            reshaped_xs = np.reshape(xs,(BATCH_SIZE,IMAGE_SIZE,IMAGE_SIZE,NUM_CHANNELS))
            _,loss_value,step = sess.run([train_op,loss,global_step],feed_dict={x:reshaped_xs,y_:ys})
            #1000轮保存一次模型
            if i%1000 == 0:
                print("After %s train steps,loss on train batch is %g." % (step,loss_value))
                saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=global_step)

    #将当前计算图输入到TensorBoard日志文件下
    writer = tf.train.SummaryWriter("MODEL_SAVE_PATH",tf.get_default_graph())
    writer.close()
def main(argv=None):
    mnist = input_data.read_data_sets("E:/tmp/data/",one_hot=True)
    train(mnist)

#主程序入口调用main
if __name__ =='__main__':
    tf.app.run()
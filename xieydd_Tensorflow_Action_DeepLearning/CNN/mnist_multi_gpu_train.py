#-*- coding: utf-8 -*-
#@Description: CNN 7层网络对MNIST的训练过程实现 多gpu并行计算并输出计算图在TensorBoard
#@author xieydd xieydd@gmail.com  
#@date 2017-10-20 下午14:24:10
import os
import time
from datetime import datetime 
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
N_GPU = 4#GPU数量

#要为不同的GPU提供不同的训练数据，如果使用palceholder需要手动准备多份数据，这里采用输入队列的方式从TFRecord中读取数据，这里的路径为将MNIST数据转换成TFRecord数据的路径
DATA_PATH = "/path/to/data/tfrecords"
def get_input():
    filename_queue = tf.train.string_input_producer([DATA_PATH])
    reader = tf.TFRecoderReader()
    _,serialized_example = reader.read(filename_queue)
    #定义数据解析格式
    features = tf.parse_single_example(serialized_example,features={
        'image_raw':tf.FixedLenFeature([],tf.string),
        'pixels':tf.FixedLenFeature([],tf.int64),
        'label':tf.FixedLenFeature([],tf.int64)})
    #解析图片和标签信息
    decode_image = tf.decode_raw(features['image_raw'],tf.uint8)
    reshaped_image = tf.reshape(decode_image,[784])
    retyped_image = tf.cast(reshaped_image,tf.float32)
    label = tf.cast(features['label'],tf.int32) 

    #定义输入队列并返回
    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3*BATCH_SIZE
    return tf.train.shuffle_batch([retyped_image,label],batch_size=BATCH_SIZE,capacity=capacity,min_after_dequeue=min_after_dequeue)

def get_loss(x,y_,regularizer,scope):
    #向前传播
    y = inference(x,regularizer)
    cross_entrypy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y,y_))

    #计算当前GPU上计算得到的正则化损失 与w相关
    regularization_loss = tf.add_n(tf.get_collection('loss',scope))
    loss = cross_entrypy + regularization_loss
    return loss

def average_gradients(tower_grads):
    average_grads = []
    #枚举所有的变量和变量在不同GPU上计算得出的梯度
    for grad_and_vars in zip(*tower_grads):
        gards = []
        for g,_ in grad_and_vars:
            expended_g = tf.expends_dims(g,0)
            grads.append(expended_g)
        grad = tf.concat(0,grads)
        grad = tf.reduce_mean(grad,0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad,v)
        #将变量和它的平均梯度对应起来
        average_grads.append(grad_and_var)
    #返回所有变量的平均梯度，用于变量更新
    return average_grads

def main(argv=None):
    #将简单运算放到cpu只有神经网络训练过程放在GPU上面
    with tf.Graph().as_default(),tf.device('/cpu:0'):
        #获取训练batch
        x,y_ = get_input()
        regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
        #定义衰减论数和指数衰减学习率
        global_step = tf.get_variable('global_step',[],initializer=tf.constant_initializer(0),trainable=False)
        learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,global_step,60000/BATCH_SIZE,LEARNING_RATE_DECAY)
        #定义优化方法
        opt = tf.train.GradientDescentOptimizer(learning_rate)
        tower_grads = []
        #将神经网络优化过程跑在不同的GPU上面
        for i in range(N_GPU):
            with tf.device('/gpu:%d' % i) as scope:
                cur_loss = get_loss(x,y_,regularizer,scope)

                #在第一次声明变量后设定reuse参数为True可以让不同的GPU同时更新同一组参数，tf.name_scope不会影响tf.get_variable的命名空间
                tf.get_variable_scope().reuse_variables()

                #计算当前变量梯度
                grads = opt.compute_gradients(cur_loss)
                tower_grads.append(grads)

        #计算变量平均梯度后输出到TensorBoard日志
        grads = average_gradients(tower_grads)
        for grad,var in grads:
            if grad is not None:
                tf.histogram_summary('gradients_on_average/%s' % var.op.name,grad)
        #使用平均梯度更新参数
        apply_gradient_op = opt.apply_gradients(grads,global_step=global_step)
        for var in tf.trainable_variables():
            tf.histogram_summary(var.op.name,var)

        #计算变量的滑动平均值
        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step=global_step)
        variable_averages_op = variable_averages.apply(tf.trainable_variables())

        #每一轮迭代需要更新变量的取值并更新变量的滑动平均值
        train_op = tf.group(apply_gradient_op,variable_averages_op)
        saver = tf.train.Saver(tf.all_variables())
        summary_op = tf.merge_all_summaries()
        init = tf.global_variables_initializer()

        #训练过程
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=-True)) as sess:
            #初始化参数启动所有队列
            init.run()
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess,coord=coord)
            summary_writer = tf.train.SummaryWriter(MODEL_SAVE_PATH,sess.graph)

            for step in range(TRAINING_STEPS):
                #执行神经网络训练操作
                start_time = time.time()

                _,loss_value = sess.run([train_op,cur_loss])
                duration=time.time() - start_time

                if step!=0 and step%10==0:
                    num_examples_per_step = BATCH_SIZE*N_GPU
                    #每秒可以处理扰动训练个数
                    examples_per_sec = num_examples_per_step/duration
                    #单个batch训练需要的时间
                    sec_per_batch = duration/N_GPU
                    #输出训练信息
                    format_str = ('step %d,loss = %.2f (%.1f examples sec;%.3f sec/batch)')
                    print(format_str % (step,loss_value,examples_per_sec,sec_per_batch))

                    #通过TensorBoard可视化训练过程
                    summary = sess.run(summary_op)
                    summary_writer.add_summary(summary,step)
                    #每隔一段保存当前的模型
                    if step%1000 == 0 or (step+1) == TRAINING_STEPS:
                        checkpoint_path = os.path.join(MODEL_SAVE_PATH,MODEL_NAME)
                        saver.save(sess,checkpoint_path,global_step)
        coord.request_stop()
        coord.join(threads)

#主程序入口调用main
if __name__ =='__main__':
    tf.app.run()
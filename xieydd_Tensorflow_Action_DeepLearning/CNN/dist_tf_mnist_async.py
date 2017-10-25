#-*- coding: utf-8 -*-
#@Description: CNN 7层网络对MNIST的训练过程实现 计算图之间的分布式模型
#@author xieydd xieydd@gmail.com  
#@date 2017-10-24 下午17:25:19
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_inference

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
DATA_PATH = "E:/tmp/data/"

#通过flags指定运行参数
FLAGS = tf.app.flags.FLAGS

#指定当前运行的是参数服务器还是计算服务器(需要在每一轮迭代进行反向传播)
tf.app.flags.DEFINE_string('job_name','worker','"ps" or "worker"')
#指定集群的参数服务器地址
tf.app.flags.DEFINE_string(
    'ps_hosts','tf-ps0:2222,tf-ps1:1111',
    'Comma-separated list of hostname:port for the parameter server jobs. e.g. "tf-ps0:2222,tf-ps1:1111"'
)
#指定集群计算服务器地址
tf.app.flags.DEFINE_string(
    'worker_hosts','tf-worker0:2222,tf-worker1:1111',
    'Comma-separated list of hostname:port for the worker jobs. e.g. "tf-worker0:2222,tf-worker1:1111"'
)

#指定当前程序的ID，Tensorflow会自动根据参数服务器和计算服务器列表中的端口来启动服务，编号一般从0开始
tf.app.flags.DEFINE_integer(
    'task_id',0,'Task ID of the worker/replica running the training.'
)


def build_model(x,y_,is_chief):
    regularizer = tf.contrib.layers.l2_regularizer()
    y = inference(x,True,regularizer)
    global_step = tf.Variable(0,trainable=False)
    
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(label=tf.argmax(y_),logits=y)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,global_step=global_step,60000/BATCH_SIZE,LEARNING_RATE_DECAY)
    triain_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
    return global_step,loss,train_op
    
def main(argv=None):
    #解析flags并配置Tensorflow集群
    ps_hosts = FLAGS.ps_hosts.split(',')
    worker_hosts = FLAGS.worker_hosts.split(',')
    cluster = tf.train.ClusterSpec({"ps":ps_hosts,"worker":worker_hosts})
    #通过ClusterSpec以及当前任务创建Server
    server = tf.train.Server(cluster,job_name=FLAGS.job_name,task_index = FLAGS.task_id)
    #参数服务器只需要管理Tensorflow中的变量，不需要执行训练过程
    if FLAGS.job_name == 'ps':
        server.join()
    
    #定义计算服务器需要运行的操作
    is_chief = (FLAGS.task_id == 0)
    mnist = input_data.read_data_sets(DATA_PATH,one_hot=True)
    #指定每一个运算的设备 replica_device_setter会将所有参数自动分配到参数服务器上，计算分配到计算服务器上
    with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % FLAGS.task_id,cluster=cluster)):
        x = tf.placeholder(tf.float32,IMPUT_NODE,name="x-input")
        y_ = tf.placeholder(tf.float32,OUT_NODE,name="y_input")
        
        global_step,loss,train_op = build_model(x,y_,is_chief)
        
        #保存模型
        saver = tf.train.Saver()
        #定义日志输出操作
        summary_op = tf.merge_all_summaries()
        #定义变量初始化操作
        init_op = tf.initialize_all_variables()
        
        #tf.train.Supervisor管理深度模型的通用功能 能管理队列、模型保存、日志输出、会话的生成
        sv = tf.train.Supervisor(
            is_chief=is_chief,#定义当前的服务器是否是主计算服务器，只有主服务器会保存模型和日志
            logdir = MODEL_SAVE_PATH,
            init_op = init_op,
            summary_op = summary_op,#指定日志生成
            saver=saver,
            global_step=global_step,#指定迭代次数，这个会体现在生成模型的文件名上
            save_model_secs=60,#指定保存模型的时间间隔
            save_summaries_secs=60#指定生成日志的时间间隔
        )
        sess_config = tf.ConfigProto(allow_soft_placement=True,log_divice_placement=False)
        #通过tf.train.Supervisor生成会话
        sess = sv.prepare_or_wait_for_session(server.target,config=sess_config)
        
        step = 0
        start_time = time.time()
        #迭代过程中tf.train.Supervisor会输出日志和保存模型
        while not sv.should_stop():
            xs,ys = mnist.train.next_batch(BATCH_SIZE)
            _,loss_value,global_step_value = sess.run([train_op,loss,global_step],feed_dict = {x:xs,y:y_})
            if global_step_value >= TRAINING_STEPS: break
            if step >0 and step % 100 ==0:
                duration = time.time() - start_time
                #不同的服务器会更新全局的训练论数 global_step_value为训练使用过batch得数量
                sec_per_batch = duration / global_step_value
                format_str = ("After %d training steps (%d global steps), loss on training batch is %g %.3f sec/batch")
                print(format_str % (step,global_step_value,loss_value,sec_per_batch))
            step += 1
        sv.stop()
        
if __name__ == "__main__":
    tf.app.run()
        
        
        
    
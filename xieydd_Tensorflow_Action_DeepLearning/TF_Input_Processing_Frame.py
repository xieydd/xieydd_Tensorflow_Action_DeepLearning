# -*- coding:utf-8 -*-
#@Description: Tensorflow输入数据处理框架 处理MNIST数据集未完成
#@author xieydd xieydd@gmail.com
#@date 2017-10-18 下午12:16:10
import tensorflow as tf
from TF_Preprocessing_Image import *
from mnist_train import *
#前提是将所有原始数据的格式统一并存储到TFRecord文件中
files = tf.train.match_filenames_once("G:/tensorflow/file/file_pattern-*")
filename_queue = tf.train.string_input_producer(files,shuffle=True)
#假设image中存储的是图像原始数据 lebel为标签 height、width、Channels为维度
reader = tf.TFRecordReader()
_,serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(serialized_example,features=
    {'image':tf.FixedLenFeature([],tf.string),
    'label':tf.FixedLenFeature([],tf.int64),
    'height':tf.FixedLenFeature([],tf.int64),
    'width':tf.FixedLenFeature([],tf.int64),
    'channels':tf.FixedLenFeature([],tf.int64)})
image,labels,height,width,channels = features['image'],features['label'],features['height'],features['width'],features['channels']
#由原始数据集解析出像素矩阵，并由图片尺寸还原图像
decode_image = tf.decode_raw(image,tf.uint8)
decode_image.set_shape([height,width,channels])
#定义神经网络输入层图片大小
image_size = 299
#使用TF_Preprocessing_Image的图像预处理方法
distorted_image = preprocess_for_train(decode_image,image_size,image_size,None)
min_after_dequeue = 10000
batch_size = 64
capacity = min_after_dequeue + 3*batch_size
image_batch,label_batch = tf.train.shuffle_batch([distorted_image,label],batch_size=batch_size,capacity=capacity,min_after_dequeue=min_after_dequeue)
#向前传播 inference calc_loss TRAINING_STEPS 可以import进来
rregularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
logit = inference(image_batch,True,regularization)
loss = calc_loss(logit,label_batch)
#将学习率优化方法以及每一轮的操作放在train_step下
with tf.name_scope("train_step"):
    train_step = tf.train.AdamsOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name='Adam').minimize(loss)
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
saver = tf.train.Saver()
with tf.Session(config) as sess:
    tf.global_variables_initializer().run()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)
    for i in range(TRAINING_STEPS):
        print(sess.run(train_step))
    writer = tf.train.SummaryWriter("MODEL_SAVE_PATH",tf.get_default_graph())
    writer.close()
    coord.request_stop()
    coord.join(threads)

#注意这是其依赖的文件
TF_Preprocessing_Image.py
# -*- coding:utf-8 -*-
#@Description: 图像预处理完整版样例
#@author xieydd xieydd@gmail.com
#@date 2017-10-17 下午14:09:23
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#给定图像和颜色 color_ordering为可选项 亮度 饱和度 色相 对比度
def distort_color(image,color_ordering=0):
    if color_ordering == 0:
        image = tf.image.random_brightness(image,max_delta=32./255.)
        image = tf.image.random_saturation(image,lower=0.5,upper=1.5)
        image = tf.image.random_hue(image,max_delta=0.2)
        image= tf.image.random_contrast(image,lower=0.5,upper=1.5)
    elif color_ordering == 1:
        image = tf.image.random_saturation(image,lower=0.5,upper=1.5)
        image = tf.image.random_brightness(image,max_delta=32./255.)
        #image= tf.image.random_constrast(image,lower=0.5,upper=1.5)
        image = tf.image.random_hue(image,max_delta=0.2)
    #elif color_ordering == 2:其他排列
    return tf.clip_by_value(image,0.0,1.0)


#给定解码后的图片，目标图像尺寸及图像的标注框  输入是图像识别问题中的原始训练数据 输出是神经网络模型输入层
def preprocess_for_train(image,height,width,bbox):
    #如果没有提供标注框就将整个图像作为标注的部分
    if bbox == None:
        bbox = tf.constant([0.0,1.0,0.0,1.0],dtype=tf.float32,shape=[1,1,4])
    #转换图像张量的类型
    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image,dtype=tf.float32)
    #随机截取图像，减小需要关注物体的大小对算法的影响
    bbox_begin,bbox_size,_ = tf.image.sample_distorted_bounding_box(tf.shape(image),bounding_boxes=bbox)
    distorted_image = tf.slice(image,bbox_begin,bbox_size)
    #将随机截取的图像调整成神经网络输入层大小
    distorted_image = tf.image.resize_images(distorted_image,[height,width],method=np.random.randint(4))
    #随机左右翻转图像
    distorted_image = tf.image.random_flip_left_right(distorted_image)
    #使用随机的顺序颜色调整图像色彩
    distorted_image = distort_color(distorted_image,np.random.randint(2))
    return distorted_image
image_raw_data = tf.gfile.FastGFile("E:/cat.jpg",'rb').read()
with tf.Session() as sess:
    img_data = tf.image.decode_jpeg(image_raw_data)
    boxes = tf.constant([[[0.05,0.5,0.9,0.7],[0.35,0.47,0.5,0.56]]])
    #运行6次获得不同的图像
    for i in range(6):
        #将图像尺寸调整成299x299
        result = preprocess_for_train(img_data,299,299,boxes)
        plt.imshow(result.eval())
        plt.show()
#inference
INPUT_NODE = 784
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
def calc_loss(logit,label_batch):
    #由于输入的是四维矩阵，所以修改格式 Batch中样例的个数 长 宽 深度 放在input名下
    with tf.name_scope('input1'):
        #x = tf.placeholder(tf.float32,[BATCH_SIZE,IMAGE_SIZE,IMAGE_SIZE,NUM_CHANNELS],name='x-input')
        #y_ = tf.placeholder(tf.float32,[None,NUM_LABELS],name='y-input')
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
        cross_entrypy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit,labels=label_batch)
        #计算当前batch的平均交叉熵
        cross_entrypy_mean = tf.reduce_mean(cross_entrypy)
        loss = cross_entrypy_mean + tf.add_n(tf.get_collection('losses'))

    #将学习率优化方法以及每一轮的操作放在train_step下
    '''
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
    '''
    return loss
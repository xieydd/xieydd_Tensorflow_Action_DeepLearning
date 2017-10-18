# -*- coding:utf-8 -*-
#@Description: Tensorflow输入数据处理框架
#@author xieydd xieydd@gmail.com  
#@date 2017-10-18 下午12:16:10
import tensorflow as tf
from TF_Preprocessing_Image import *

#前提是将所有原始数据的格式统一并存储到TFRecord文件中
files = tf.train.match_filenames_once("path/to/file_pattern-*")
filename_queue = tf.train.string_input_producer(files,shuffle=True)

#假设image中存储的是图像原始数据 lebel为标签 height、width、Channels为维度
reader = tf.TFRecoderReader()
_,serialized_example = reader.reader(filename_queue)
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
logit = inference(image_batch)
loss = calc_loss(logit,label_batch)
train_step = tf.train.AdamsOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name='Adam').minimize(loss)

with tf.Session() as sess:
	tf.global_variables_initializer().run()
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess=sess,coord=coord)

	for i in range(TRAINING_STEPS)：
		sess.run(train_step)

	coord.request_stop()
	coord.join(threads)



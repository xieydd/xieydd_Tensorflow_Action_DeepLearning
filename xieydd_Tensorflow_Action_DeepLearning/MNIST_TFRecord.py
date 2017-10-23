# -*- coding:utf-8 -*-
#@Description: 将MNIST数据转换成TFRecord格式 和 读取TFRecord格式文件
#@author xieydd xieydd@gmail.com
#@date 2017-10-16 下午16:50:50
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
#生成整数型属性
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
#生成字符串型数据
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
mnist = input_data.read_data_sets("E:/tmp/data/",dtype=tf.uint8,one_hot=True)
images = mnist.train.images
labels = mnist.train.labels
pixels = images.shape[1]
num_examples = images.shape[0]
for index in range(500):
    #输出TFRecord文件的地址
    filename = ("G:/tensorflow/output_tfrecords/data.tfrecords-%.5d-of-%.5d" % (index,num_examples))
    writer = tf.python_io.TFRecordWriter(filename)
    image_raw = images[index].tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'pixels': _int64_feature(pixels),
        'label': _int64_feature(np.argmax(labels[index])),
        'image_raw': _bytes_feature(image_raw)
    }))
    writer.write(example.SerializeToString())
writer.close()
print("TFRecord文件已保存。")



##############################################


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import numpy as np
#创建一个reader
reader = tf.TFRecordReader()
#创建一个队列维护输入文件列表
filename_queue = tf.train.string_input_producer(["G:/tensorflow/output_tfrecords"])
#从文件中读取样例,读取一个，reda_up_to可以一次性读取多个样例
_,serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(serialized_example,features={
    'image_raw':tf.FixedLenFeature([],tf.string),
    'pixels':tf.FixedLenFeature([],tf.int64),
    'label':tf.FixedLenFeature([],tf.int64)
    })
#tf.decode_raw可以将字符串解析层图像对应的像素数组
images = tf.decode_raw(features['image_raw'],tf.uint8)
labels = tf.cast(features['label'],tf.int32)
pixels = tf.cast(features['pixels'],tf.int32)
#启动多线程处理数据
sess = tf.Session()
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess,coord=coord)
for i in range(1):
    images,label,pixels = sess.run([images,labels,pixels])
sess.run(print(images))
coord.request_stop()
coord.join(threads)
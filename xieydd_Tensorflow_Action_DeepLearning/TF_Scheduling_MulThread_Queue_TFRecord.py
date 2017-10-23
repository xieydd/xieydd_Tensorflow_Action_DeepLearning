# -*- coding:utf-8 -*-
#@Description: Tensorflow对于线程的调度 多线程操作TFRecord文件
#@author xieydd xieydd@gmail.com
#@date 2017-10-17 下午17:10:20
import tensorflow as tf

#常见TFRecord文件的帮助函数
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
#模拟海量数据情况下将数据写入不同文件，num_shards定义多少文件 insatnces_per_shard定义每个文件多少数据
num_shards = 2
insatnces_per_shard = 2
for i in range(num_shards):
    filename = ('G:/tensorflow/output_tfrecords/data.tfrecords-%.5d-of-%.5d' % (i,num_shards))
    writer = tf.python_io.TFRecordWriter(filename)
    #将数据封装成Example并写入TFRecord文件
    for j in range(insatnces_per_shard):
        example = tf.train.Example(features = tf.train.Features(feature = {'i':_int64_feature(i),'j':_int64_feature(j)}))
        writer.write(example.SerializeToString())
writer.close()

#将生产的文件读取成文件列表
files = tf.train.match_filenames_once("G:/tensorflow/output_tfrecords/data.tfrecords-*")
init = (tf.global_variables_initializer(), tf.local_variables_initializer())
#创建输入队列 一般真实环境shuffle会设置为True
filename_queue = tf.train.string_input_producer(files,shuffle=False)
#解析队列中的样本
reader = tf.TFRecordReader()
_,serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(serialized_example,features={'i':tf.FixedLenFeature([],tf.int64),'j':tf.FixedLenFeature([],tf.int64)})
with tf.Session() as sess:
    #虽然这里没有定义变量，但是match_filenames_once需要初始化一些变量
    sess.run(init)
    print(sess.run(files))
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)
    #多次执行获取数据的操作
    for i in range(6):
        print(sess.run([features['i'],features['j']]))
    coord.request_stop()
    coord.join(threads)
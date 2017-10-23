# -*- coding:utf-8 -*-
#@Description: Tensorflow多线程调度使队列单或多个进，batch出作为神经网络输入
#@author xieydd xieydd@gmail.com
#@date 2017-10-18 下午13:39:10
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
#创建输入队列 一般真实环境shuffle会设置为True
filename_queue = tf.train.string_input_producer(files,shuffle=False)
#解析队列中的样本
reader = tf.TFRecordReader()
init  = (tf.global_variables_initializer(), tf.local_variables_initializer())
_,serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(serialized_example,features={'i':tf.FixedLenFeature([],tf.int64),'j':tf.FixedLenFeature([],tf.int64)})
#假设Example结构中i表示样例的特征向量 j表示标签
example,label = features['i'],features['j']
batch_size = 3
#队列大小的设置需要和batch大小相关联，避免出现队列过小导致阻塞或者过大导致占用大量内存
capacity = 1000 + 3*batch_size
#shuffle_batch函数是顺序打乱的
example_batch,label_batch = tf.train.batch([example,label],batch_size=batch_size,capacity=capacity)
with tf.Session() as sess:
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)
    #获取并答应组合后的样例，真实情况一般直接作为神经网络的输入
    for i in range(2):
        cur_example_batch,cur_label_batch = sess.run([example_batch,label_batch])
        print(cur_example_batch,cur_label_batch)
    coord.request_stop()
    coord.join(threads)
#这样实现单进Batch出
#那么如何实现多进程同时执行入队操作(数据读取和预处理)？
#设置tf.train.shuffle_batch()的num_threads参数对一个文件中的不同样例处理 避免同一文件是相同类，影响神经网络训练
#如何对不同文件中的不同样例进行处理?
#tf.train.shuffle_batch_join()可以，避免线程数大于文件数，导致文件被分割处理导致效率降低(硬盘寻址) 但是一般输入队列会使用tf.train.string_input_producer()生成真阳会保证文件平均分配
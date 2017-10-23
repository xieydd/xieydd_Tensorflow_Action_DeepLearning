# -*- coding:utf-8 -*-
#@Description: Tensorflow对于线程的调度 多线程操作一个队列数字
#@author xieydd xieydd@gmail.com  
#@date 2017-10-17 下午16:05:20
import tensorflow as tf
import numpy as np 
import time
import threading

#线程运行的程序 每隔一秒看是否需要停止并打印自己的id
def MyLoop(coord,worker_id):
	'''
	tf.Coordinator类提供三种协调工具
	should_stop:返回True当前线程退出
	request_stop:通知其他线程退出 之后should_stop的返回值会被设定为True则其他同时线程退出
	'''
	while not coord.should_stop() :
		if np.random.randn() < 0.1:
			print("Stopping from id : %d\n" % worker_id)
			coord.request_stop()
		else:
			print("Working on id : %d\n" % worker_id)
		time.sleep(1)

coord = tf.train.Coordinator()
#声明创建5个线程 list(xrange(5)) = [0,1,2,3,4]所以xrange(5)不是一个数组而是一个生成器
threads = [threading.Thread(target=MyLoop,args=(coord,i,)) for i in xrange(5)]

for t in threads: t.start()
#等待所有线程退出
coord.join(threads)



#多线程操作队列
#声明一个100个元素的FIFO队列
queue = tf.FIFOQueue(100,"float")
#定义入队操作
enqueue_op = queue.enqueue([tf.random_normal([1])])

#QueueRunner创建多个线程运行队列的入队操作 这里表示5个enqueue_op操作
qr = tf.train.QueueRunner(queue,[enqueue_op]*5)
#将定义过的QueueRunner加入Tensorflow计算图指定的集合中，这里没有指定加入默认的tf.GraphKeys.QUEUES_RUNNERS
tf.train.add_queue_runner(qr)

#定义出队列操作
out_queue = queue.dequeue()

with tf.Session as sess:
	coord = tf.Coordinator()
	'''
	使用tf.train.QueueRunner需要明确的调用tf.train.start_queue_runners启动所有线程，如果没有
	线程进行入队操作，当调用出队操作的时候程序会一直等待，tf.train.start_queue_runners会默认启动
	tf.GraphKeys.QUEUES_RUNNERS集合中国所有的QueueRunner 所以这两个需要制定一个集合或者都是默认
	'''
	threads = tf.train.start_queue_runners(sess=sess,coord=coord)
	for _ in range(3) :
		print(sess.run(out_queue)[0])
	coord.request_stop()
	coord.join(threads)

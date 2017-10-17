# -*- coding:utf-8 -*-
#@Description: Tensorflow对于队列的操作 
#@author xieydd xieydd@gmail.com  
#@date 2017-10-17 下午15:37:23
import tensorflow as tf 

#创建一个FIFO队列,指定最多保存两个元素，类型为整数
'''
Tensorflow提供两种队列:
FIFOQueue、RandomHuffleQueue：会将原元素打乱，每次出队列是随机的
'''
q = tf.FIFOQueue(2,"int32")
#与初始化变量一样，这里需要初始化队列两个数为0和10
init= q.enqueue_many(([0,10],))
#使用Dequeue函数将队列中的第一个元素出队列，这个元素存到变量x中
x = q.dequeue()
#将得到的数加1
y = x+1
#将加1后的值重新加入队列
q_inc = q.enqueue([y])

with tf.Session as sess:
	init.run()
	for _ in range(5):
		v,_ = sess.run([x,q_inc])
		print(x)

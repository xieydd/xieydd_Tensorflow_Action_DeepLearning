#-*- coding: utf-8 -*-
##-*- coding: utf-8 -*-
#@Description: CNN 7层网络对MNIST测试的实现
#@author xieydd xieydd@gmail.com  
#@date 2017-10-15 下午15:18:10
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from mnist_inference import *
from mnist_train import *

#每十秒加载一次新的饿模型，并在测试数据集上测试最新的模型的正确性
EVAL_INTERVAL_SECS = 1000

def evaluate(mnist):
    with tf.Graph().as_default() as g:
        #定义输入输出的格式
        x = tf.placeholder(tf.float32,[BATCH_SIZE,IMAGE_SIZE,IMAGE_SIZE,NUM_CHANNELS],name='x-input')
        y_ = tf.placeholder(tf.float32,[None,NUM_LABELS],name='y-input')
        

        #测试时不关注正则化损失
        y = inference(x,False,None)
        #average为 batch_size * 10数组 argmax的第二个参数表示只在第一维进行找到最大值的下标
        correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
        #一组batch的正确率
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        #通过变量名加载模型

        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True

        #每隔EVAL_INTERVAL_SECS调用一次正确率计算过程检查正确率变化
        while True:
            with tf.Session(config = config) as sess:
                #找到最新模型
                ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path :
                    #加载模型
                    saver.restore(sess,ckpt.model_checkpoint_path)
                    #通过文件名的奥模型保存迭代的论数
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    xs,ys = mnist.validation.next_batch(BATCH_SIZE)
                    reshaped_xs = np.reshape(xs,(BATCH_SIZE,IMAGE_SIZE,IMAGE_SIZE,NUM_CHANNELS))
                    validate_feed = {x:reshaped_xs,y_:ys}
                    accuracy_score = sess.run(accuracy,feed_dict=validate_feed)

                    print("After %s train step,valicdation accuracy is %g" % (global_step,accuracy_score))
                else:
                    print("No Checkpoint file found")
                    return
         time.sleep(EVAL_INTERVAL_SECS)
def main(argv=None):
    mnist = input_data.read_data_sets("E:/tmp/data/",one_hot=True)
    evaluate(mnist)
#主程序入口调用main
if __name__ =='__main__':
    tf.app.run()
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
        image= tf.image.random_constrast(image,lower=0.5,upper=1.5)
    elif color_ordering == 1:
        image = tf.image.random_saturation(image,lower=0.5,upper=1.5)
        image = tf.image.random_brightness(image,max_delta=32./255.)
        image= tf.image.random_constrast(image,lower=0.5,upper=1.5)
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
    bbox_begin,bbox_size,_ = tf.image.sample_distributed_bounding_box(tf.shape(image),bounding_boxes=bbox)
    distorted_image = tf.slice(image,bbox_begin,bbox_size)
    #将随机截取的图像调整成神经网络输入层大小
    distorted_image = tf.image.resize_images(distorted_image,height,width,method=np.random.randint(4))
    #随机左右翻转图像
    distorted_image = tf.image.random_flip_left_right(distorted_image)
    #使用随机的顺序颜色调整图像色彩
    distorted_image = distort_color(distorted_image,color_ordering=np.random.randint(2))
    return distorted_image
image_raw_data = tf.gfile.FastGFile("/path/to/picture",'r').read()
with tf.Session() as sess:
    img_data = tf.image.decode_jpeg(image_raw_data)
    boxes = tf.constant([[[0.05,0.5,0.9,0.7],[0.35,0.47,0.5,0.56]]])
    #运行6次获得不同的图像
    for i in range(6):
        #将图像尺寸调整成299x299
        result = preprocess_for_train(img_data,299,299,boxes)
        plt.imshow(result.eval())
        plt.show()
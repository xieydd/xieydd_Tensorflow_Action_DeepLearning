# -*- coding:utf-8 -*-
#@Description: Tensorflow将jpg文件编码和解码
#@author xieydd xieydd@gmail.com
#@date 2017-10-17 上午10:48:50
import matplotlib.pyplot as plt
import tensorflow as tf
#读取原始数据
image_raw_data = tf.gfile.FastGFile("E:/cat.jpg",'rb').read()
with tf.Session() as sess:
    img_data = tf.image.decode_jpeg(image_raw_data)
    #输出解码后的三维矩阵
    print(img_data.eval())
    plt.imshow(img_data.eval())
    plt.show()
    #将数据转换成实数方便下面样例程序对于图像的处理
    img_data = tf.image.convert_image_dtype(img_data,dtype=tf.uint8)
    #将三维矩阵重新按照jpeg格式编码存入文件并打开
    encoded_image = tf.image.encode_jpeg(img_data)
    with tf.gfile.GFile("E:/cat1.jpg","wb") as f:
        f.write(encoded_image.eval())
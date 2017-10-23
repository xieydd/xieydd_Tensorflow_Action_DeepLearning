# -*- coding:utf-8 -*-
#@Description: Tensorflow改变jpeg图片大小 颜色 方向 
#@author xieydd xieydd@gmail.com  
#@date 2017-10-17 上午11:09:50

#加载原始图片
import matplotlib.pyplot as plt 
import tensorflow as tf 

#读取原始数据
image_raw_data = tf.gfile.FastGFile("/path/to/picture",'r').read()

with tf.Session() as sess:
	img_data = tf.image.decode_jpeg(image_raw_data)
	#method为调整参数的算法
	'''
	method:
	0 双线性插值法
	1 最近邻法 KNN
	2 双三次插值法
	3 面积插值法
	'''
	#这里是进行压缩 实验发现只有1行2出错，3和4都是噪声
	resized = tf.image.resize_images(img_data,[300,300],method=1)
	#这里没指定深度会显示一个?
	print(resized.shape)


	#剪切
	croped = tf.image.resize_image_with_crop_or_pad(img_data,1000,1000)
	padded = tf.image.resize_image_with_crop_or_pad(img_data,3000,3000)

	#等比例调节,截取中间50%的图像
	central_cropped = tf.image.central_crop(img_data,0.5)

	#剪裁特定区域 tf.image.crop_to_bounding_box 填充特定区域 tf.image.pad_to_bunding_box注意尺寸需要满足条件
	
	#图像翻转
	flipped = tf.image.flip_up_down(img_data)#上下
	fliped = tf.image.flip_left_right(img_data)#左右翻转
	transposed = tf.image.transpose_image(img_data)#对角线翻转

	#随机翻转
	flipped = tf.image.random_flip_up_down(img_data)
	flipped = tf.image.random_flip_left_right(img_data)

	#图像色彩调整
	#图像亮度-0.5
	adjusted = tf.image.adjust_brightness(img_data,-0.5)
	#在[-max_delta,max_delta]范围内随机调整图像亮度
	adjusted = tf.image.random_brightness(img_data,max_delta)

	#对比度调节
	adjusted = tf.image.adjust_contrast(img_data,-5)
	adjusted = tf.image.adjust_contrast(img_data,5)
	#在范围内
	adjusted = tf.image.random_constrast(img_data,lower,upper)

	#调整图像色相
	adjusted = tf.image.adjust_hue(img_data,0.1)
	adjusted = tf.image.adjust_hue(img_data,0.3)
	adjusted = tf.image.adjust_hue(img_data,0.6)
	adjusted = tf.image.adjust_hue(img_data,0.9)
	#随机调整 [0.max_delta] 
	adjusted = tf.image.random_hue(img_data,max_delta)

	#调整饱和度
	adjusted = tf.image.adjust_saturation(img_data,-5)
	adjusted = tf.image.adjust_saturation(img_data,5)
	adjusted = tf.image.random_saturation(img_data,lower,upper)

	#将亮度的均值变为0方差为1 这个好像失效了
	#adjusted = tf.image.per_image_whitening(img_data)


	#处理标注框
	#将图像缩小，这样标注更加清楚
	img_data = tf.image.resize_images(img_data,180,267,method=1)
	batched = tf.expand_dims(tf.image.convert_image_dtype(img_data,tf.float32),0)
	#设置方框 这里的是比例 且设置两个方框
	boxes = tf.constant([[[0.05,0.05,0.9,0.7],[0.35,0.47,0.5,0.56]]])
	#batched要求是实数 而且是多张图片组成的四维矩阵
	result = tf.image.draw_bounding_boxing(batched,boxes)


	#加入随机标注框
	boxes = tf.constant([[[0.05,0.05,0.9,0.7],[0.35,0.47,0.5,0.56]]])
	begin,size,bbox_for_draw = tf.image.sample_distributed_bounding_box(tf.shape(img_data),bounding_boxes=boxes)
	boxes = tf.constant([[[0.05,0.05,0.9,0.7],[0.35,0.47,0.5,0.56]]])
	img_with_box = tf.image.draw_bounding_boxing(batched,bbox_for_draw)
	#每一次得到的饿结果会有所不同
	distributed_iamge=tf.slice(img_data,begin,size)

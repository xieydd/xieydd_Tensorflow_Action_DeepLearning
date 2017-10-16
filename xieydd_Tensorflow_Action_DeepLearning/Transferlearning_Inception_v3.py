# -*- coding:utf-8 -*-
import os.path
import glob
import numpy as np 
import random
import tensorflow as tf
from tensorflow.python.platform import gfile

#Inception-v3模型瓶颈层节点数
BOTTLENECK_TENSOR_SIZE = 2048

#Inception-v3模型代表瓶颈层结果张量名称
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'

#图片输入张量对应的名称
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'

#下载好的谷歌训练好的Inception-vc3模型
MODEL_DIR = '/path/to/model'
MODEL_FILE = 'classify_image_graph_def.pb'

#将原图像通过模型得到的特征向量保存在文件中
CHAHE_DIR = 'tmp/bottleneck'

#图片数据文件夹
INPUT_DATA = '/path/to/flower_data'

#验证数据的百分比
VALIDATION_PRECENTAGE = 10
#测试数据的百分比
TEST_PRECENTAGE = 10

#定义神经网络设置
LEARNING_RATE = 0.01
STEPS = 4000
BATCH = 64

#读取文件夹内的图片，按训练、验证、测试数据分开
def create_image_lists(TEST_PRECENTAGE,VALIDATION_PRECENTAGE):
	#将所有照片存到result这个字典中
	result = {}
	#获取当前目录下的子文件夹
	sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]

	#得到第一个目录是当前目录不予考虑
	is_root_dir = True
	for sub_dir in sub_dirs:
		if is_root_dir:
			is_root_dir = False
			continue

		#获取当前文件夹下的所有图片文件
		extensions = ['jpg','jpeg','JPG','JPEG']
		file_list = []
		dir_name = os.path.basename(sub_dir)
		for extension in extensions:
			file_glob = os.path.join(INPUT_DATA,dir_name,'*.'+extension)
			file_list.extend(glob.glob(file_glob))
		if not file_list:continue

		#通过目录名获取类别名称
		label_name = dir_name.lower()

		#初始化当前训练集、测试集、验证集
		training_images = [] 
		testing_images = []
		validation_images = []

		for file_name in file_list:
			base_name = os.path.basename(file_name)
			#随机将数据分配到训练集、验证集、测试集
			chance = np.random.randint(100)
			if chance < VALIDATION_PRECENTAGE:
				validation_images.append(base_name)
			if chance < (VALIDATION_PRECENTAGE + TEST_PRECENTAGE:)
				testing_images.append(base_name)
			else:
				training_images.append(base_name)

		result[label_name] = {
		'dir':dir_name,
		'training':training_images,
		'testing':testing_images,
		'valicdation':validation_images
		}
	return result


#这个函数通过类别名称、所属的数据集类型、图片编号获得一张图片的地址
def get_image_path(image_lists,image_dir,label_name,index,category):
	#获得给定类别中的所有图片信息
	label_lists = image_lists[label_name]
	#根据所属数据集类型获得集合中全部照片信息
	category_list = label_lists[category]
	mod_index = index % len(category_list)
	#获取图片名称
	base_name = category_list[mod_index]
	sub_dir = label_lists['dir']
	#最终的地址是数据根目录的地址加上类别文件夹加上图片名称
	full_path = os.path.join(image_dir,sub_dir,base_name)
	return full_path

#这个函数通过类别名称、所属的数据集和图片编号
#
#
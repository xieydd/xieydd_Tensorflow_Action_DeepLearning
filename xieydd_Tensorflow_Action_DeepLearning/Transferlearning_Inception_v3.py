# -*- coding:utf-8 -*-
#@Description: 迁移学习之Inception_v3模型实现
#@author xieydd xieydd@gmail.com  
#@date 2017-10-16 下午14:25:50 
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
MODEL_DIR = 'G:/tensorflow/inception_dec_2015'
MODEL_FILE = 'tensorflow_inception_graph.pb'

#将原图像通过模型得到的特征向量保存在文件中
CHAHE_DIR = 'G:/tensorflow/tmp/bottleneck'

#图片数据文件夹
INPUT_DATA = 'G:/tensorflow/flower_photos'

#验证数据的百分比
VALIDATION_PRECENTAGE = 10
#测试数据的百分比
TEST_PRECENTAGE = 10

#定义神经网络设置
LEARNING_RATE = 0.01
STEPS = 4000
BATCH = 100

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
			if chance < (VALIDATION_PRECENTAGE + TEST_PRECENTAGE):
				testing_images.append(base_name)
			else:
				training_images.append(base_name)

		result[label_name] = {
		'dir':dir_name,
		'training':training_images,
		'testing':testing_images,
		'validation':validation_images
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

#这个函数通过类别名称、所属的数据集和图片编号获得经过Inception_v3模型处理得到的特征向量文件地址
def get_bottleneck_path(image_lists,label_name,index,category):
	return get_image_path(image_lists,CHAHE_DIR,label_name,index,category)+'.txt'

#使用加载的训练好的Inception_v3模型处理一张照片，得到这个照片的特征向量
def run_bottleneck_on_image(sess,image_data,image_data_tensor,bottleneck_tensor):
	#将图片作为计算瓶颈张量的值，这个值就是这张图片新的特征
	bottleneck_values = sess.run(bottleneck_tensor,{image_data_tensor:image_data})
	#经过神经网络得到的是一个四层向量，这里转换成一层的特征向量
	bottleneck_values = np.squeeze(bottleneck_values)
	return bottleneck_values

#这个方法获取一张图片经过Inception_v3模型处理的特征向量，会先寻找已经计算保存下来的特征向量，如果找不到就创建
def get_or_create_bottlencek(sess,image_lists,label_name,index,category,jpeg_data_tensor,bottleneck_tensor):
	 #获取一张图片对应特征向量文件路径
    label_lists = image_lists[label_name]
    sub_dir = label_lists['dir']
    sub_dir_path = os.path.join(CHAHE_DIR,sub_dir)
    if not os.path.exists(sub_dir_path) : os.makedirs(sub_dir_path)

    bottleneck_path = get_bottleneck_path(image_lists,label_name,index,category)
    #如果这个特征向量文件不存在，通过模型计算特征向量并存入文件
    if not os.path.exists(bottleneck_path):
    #获取原始图片的路径
        image_path = get_image_path(image_lists,INPUT_DATA,label_name,index,category)
        #获取图片内容
        image_data = gfile.FastGFile(image_path,'rb').read()
        #通过模型计算得到特征
        bottleneck_values = run_bottleneck_on_image(sess,image_data,jpeg_data_tensor,bottleneck_tensor)
        #将计算的特征向量存入文件
        bottleneck_string = ','.join(str(x) for x in bottleneck_values)
        with open(bottleneck_path,'w') as bottleneck_file:
            bottleneck_file.write(bottleneck_string)

    else:
    #直接从文件中获取图片的对应的特征向量
        with open(bottleneck_path,'r') as bottleneck_file:
            bottleneck_string = bottleneck_file.read()
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
        #返回特征向量
    return bottleneck_values

#函数随机获取一个batch的图片作为训练数据
def get_random_cached_bottlenecks(sess,n_classes,image_lists,how_many,category,jpeg_data_tensor,bottleneck_tensor):
	bottlenecks = []
	ground_truths = []
	for _ in range(how_many):
		#随机一个类别和图片编号加入当前的训练数据
		label_index = random.randrange(n_classes)
		label_name = list(image_lists.keys())[label_index]
		image_index = random.randrange(65536)
		bottneck = get_or_create_bottlencek(sess,image_lists,label_name,image_index,category,jpeg_data_tensor,bottleneck_tensor)
		ground_truth = np.zeros(n_classes,dtype=np.float32)
		ground_truth[label_index] = 1.0
		bottlenecks.append(bottneck)
		ground_truths.append(ground_truth)
	return bottlenecks,ground_truths

#获得全部测试数据，最终测试的时候需要在所有测试数据上计算正确率
def get_test_bottlenecks(sess,image_lists,n_classes,jpeg_data_tensor,bottleneck_tensor):
	bottlenecks = []
	ground_truths = []
	label_name_list = list(image_lists.keys())

	#枚举所有类和类别中的测试图片
	for label_index,label_name in enumerate(label_name_list):
		category = 'testing'
		for index ,unused_base_name in enumerate(image_lists[label_name][category]):
			bottleneck = get_or_create_bottlencek(sess,image_lists,label_name,index,category,jpeg_data_tensor,bottleneck_tensor)
			ground_truth = np.zeros(n_classes,dtype=np.float32)
			ground_truth[label_index] = 1.0
			bottlenecks.append(bottlenecks)
			ground_truths.append(ground_truth)
	return bottlenecks,ground_truths 



def main(_):
    #读取所有图片
    image_lists = create_image_lists(TEST_PRECENTAGE,VALIDATION_PRECENTAGE)
    n_classes = len(image_lists.keys())
    #读取已经训练好的模型，Google将其保存在GraphDef Protocol Buffer内，里面保存每一个节点取值计算方法以及变量的取值
    with gfile.FastGFile(os.path.join(MODEL_DIR,MODEL_FILE),'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    #加载模型，返回数据输入对应的张量以及计算瓶颈结果对应的张量
    bottleneck_tensor,jpeg_data_tensor = tf.import_graph_def(graph_def,return_elements=[BOTTLENECK_TENSOR_NAME,JPEG_DATA_TENSOR_NAME])

    #定义新的神经网络输入，即进过inception_v3模型后到达瓶颈层的节点取值，即特征提取
    bottleneck_input = tf.placeholder(tf.float32,[None,BOTTLENECK_TENSOR_SIZE],name='BottleneckInputPlaceholder')
    #定义新的标准答案输入
    ground_truth_input = tf.placeholder(tf.float32,[None,n_classes],name='GroundTruthInput')

    #定义全连接层解决图片分类问题
    with tf.name_scope('final_training_ops'):
        weights = tf.Variable(tf.truncated_normal([BOTTLENECK_TENSOR_SIZE,n_classes],stddev=0.001))
        biases = tf.Variable(tf.zeros([n_classes]))
        logits = tf.matmul(bottleneck_input,weights) + biases
        final_tensor = tf.nn.softmax(logits)

    #定义交叉熵损失函数
    cross_entrypy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=tf.argmax(ground_truth_input,1))
    cross_entrypy_mean = tf.reduce_mean(cross_entrypy)
    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entrypy_mean)

    #计算正确率
    with tf.name_scope('evaluation'):
        correct_prediction = tf.equal(tf.argmax(final_tensor,1),tf.argmax(ground_truth_input,1))
        evaluation_step = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.85)
    with tf.Session(config=config) as sess:
        tf.global_variables_initializer().run()

        for i in range(STEPS):
            #每一次获得一个batch的训练数据
            train_bottlenecks,train_ground_truth = get_random_cached_bottlenecks(sess,n_classes,image_lists,BATCH,'training',jpeg_data_tensor,bottleneck_tensor)
            sess.run(train_step,feed_dict={bottleneck_input:train_bottlenecks,ground_truth_input:train_ground_truth})

            #在验证数据上测试正确率
            if i%100 == 0 or i+1 ==STEPS:
                validation_bottlenecks,valicdation_ground_truth = get_random_cached_bottlenecks(sess,n_classes,image_lists,BATCH,"validation",jpeg_data_tensor,bottleneck_tensor)
                validation_accuracy = sess.run(evaluation_step,feed_dict={bottleneck_input:validation_bottlenecks,ground_truth_input:valicdation_ground_truth})
                print('Step %d:Validation accuracy on random sample %d examples = %.1f%%' % (i,BATCH,validation_accuracy*100))

        #在测试集上测试正确率
        test_bottlenecks,test_ground_truth = get_test_bottlenecks(sess,image_lists,n_classes,jpeg_data_tensor,bottleneck_tensor)
        test_accuracy = sess.run(evaluation_step,feed_dict={bottleneck_input:test_bottlenecks,ground_truth_input:test_ground_truth})
        print('Final test accuracy  = %.1f%%' % (test_accuracy*100))

if __name__ == '__main__':
    tf.app.run()
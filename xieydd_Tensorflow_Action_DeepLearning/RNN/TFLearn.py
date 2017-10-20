# -*- coding:utf-8 -*-
#@Description: TFLearn在iris数据集上应用 和预测正弦函数
#@author xieydd xieydd@gmail.com  
#@date 2017-10-19 下午15:41:51

from sklearn import cross_validation
from sklearn import datasets
from sklearn import metrics
import tensorflow as tf 

#导入TFLearn
learn = tf.contrib.learn

#自定义模型 输入特征，正确答案 输出预测值，损失值和预测步骤
def my_model(features,targets):

	#将预测目标转换成one-hot编码模式 三个类别故向量长度为3 第一个类别为(1,0,0) 第二个为(0,1,0) ..
	target = tf.one_hot(target,3,1,0)

	logits,loss = learn.models.logistic_regression(features,target)
	train_op = tf.contrib.layers.optimize_loss(loss,tf.contrib.framework.get_global_step(),optimizer='Adagrad',learning_rate=0.1)
	return tf.arg_max(logits,1),loss,train_op

#加载iris数据集
iris = datasets.load_iris()
x_train,y_train,x_test,y_test = cross_validation.train_test_split(iris.data,iris.target,test_size=0.2,random_state =0)
#对自定义模型封装
classifier = learn.Estimator(model_fn = my_model)
#训练迭代100轮
classifier.fit(x_train,y_train,steps=100)
#将训练好的模型对结果预测
y_prediected = classifier.predict(x_test)

score = metrics.accuracy_score(y_test,y_prediected)
print("Accuracy: %.2f%%" % (score*100))



#预测正弦函数
from matplotlib import pyplot as plt

learn = tf.contrib.learn
HIDDEN_SIZE = 30#LSTM隐藏节点数目
NUM_LAYERS= 2#LSTM层数
TIMESTEPS = 10#截断长度
BATCH_SIZE = 32
TRAIN_STEPS= 10000
TRAINING_EXAMPLES = 10000#训练集个数
TEST_EXAMPLES = 1000  
SAMPLE_GAP = 0.1#采样间隔

def generate_data(seq):
	X = []
	y = []
	#i项和TIMESTEPS-1项合起来作为输入 第i+TIMESTEPS作为输出
	for i in range(len(seq) - TIMESTEPS -1):
		X.append([seq[i:i+TIMESTEPS]])
		y.append([seq[i+TIMESTEPS]])
	return np.array(X,dtype=np.float32) , np.array(y,dtype=np.float32)

def lstm_model(X,y):
	#多层lstm
	lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
	cell = tf.nn.rnn.MultiRNNCell([lstm_cell]*NUM_LAYERS)
	x_ = tf.unpack(X,axis=1)

	#
	output,_ = tf.nn.run(cell,x_,dtype=tf.float32)
	#在本问题只关注最后一个时刻输出，该结果为下一时刻的预测值
	output = output[-1]
	#对LSTM网络的输出
	prediction,loss = learn.models.logistic_regression(output,y)

	#创建模型优化器并得到优化步骤
	trian_op = tf.contrib.layers.optimize_loss(loss,tf.contrib.framework.get_global_step(),optimizer="Adagrad",learning_rate=0.1)	

	return prediction,loss,train_op

#建立深层网络模型
regressor = learn.Estimator(model_fn=lstm_model)

#用正弦函数生成训练和测试数据集合
test_start = TRAINING_EXAMPLES * SAMPLE_GAP
test_end = (TRAINING_EXAMPLES + TEST_EXAMPLES)*SAMPLE_GAP
#np.linspace创建等差数组 起 终 个数
train_X,train_y = generate_data(np.sin(np.linspace(0,test_start,TRAINING_EXAMPLES,dtype=np.float32)))
test_X,test_y = generate_data(np.sin(np.linspace(test_start,test_end,TEST_EXAMPLES,dtype=np.float32)))

#调用fit训练模型
regressor.fit(train_X,train_y,batch_size=BATCH_SIZE,steps=TRAINING_STEPS)
#将训练好的模型，对测试集进行预测
predicted = [[pred] for pred in regressor.predict(test_X)]
#计算rmse作为评价指标 均方根误差
rmse = np.sqrt(((predicted - test_y)**2).mean(axis=0))
print("Mean Square Error is: %f" % rmse[0])

fig = plt.figure()
plot_predicted = plt.plot(predicted,label='predicted')
plot_test = plt.plot(test_y,label='real_sin')
plt.legend([plot_predicted,plot_test],['predicted','real_sin'])
fig.savefig('sin.png')
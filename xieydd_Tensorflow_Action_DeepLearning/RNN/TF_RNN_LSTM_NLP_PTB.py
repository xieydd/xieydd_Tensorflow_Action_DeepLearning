# -*- coding:utf-8 -*-
#@Description: Tensorflow LSTM在PTB数据集上实现语言模型
#@author xieydd xieydd@gmail.com  
#@date 2017-10-19 上午11:47:56
import numpy as np 
import tensorflow as tf 
from tensorflow.models.rnn.ptb import reader

DATA_PATH = "/"
HIDDEN_SIZE = 200
NUM_LAYERS = 2 #深层循环神经网络层数
VOCAB_SIZE = 10000#词典规模包含结束标识符和稀有单词标识符

LEARNING_RATE = 1.0
TRAIN_BATCH_SIZE = 64
TRAIN_NUM_STEP = 80#训练截断长度

#注意测试的时候不需要使用截断，所以可以将测试数据看成一个超长的序列
EVAL_BATCH_SIZE = 1
EVAL_NUM_STEP = 1#测试截断长度
NUM_EPOCH = 2#使用训练数据的轮速
KEEP_PROB = 0.5
MAX_GRAD_NORM = 5#用于控制梯度膨胀的参数 

#使用类描述，方便维护循环神经网络的状态
class PTBModel(object)；
	def __init__(self,is_training,batch_size,num_steps):
		#记录使用的batch大小和截断大小
		self.batch_size = batch_size
		self.num_steps = num_steps

		#定义输入层 和ptb_iterator输出的训练数据batch的shape相同
		self.input_data = tf.placeholder(tf.int32,[batch_size,num_steps])

		#定义预期输出
		self.targets = tf.placeholder(tf.int32,[batch_size,num_steps])

		#定义LSTM结构为循环体结构并使用dropout的深层神经网络
		lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
		if is_training:#判断是否使用dropout,只在训练时使用dropout
			lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell,output_keep_prob=KEEP_PROB)
		cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell]*NUM_LAYERS)

		#初始化最初的状态
		self.initial_state = cell.zero_state(batch_size,tf.float32)
		#将单词ID转换成单词向量 每个单词向量的维度为HIDDEN_SIZE 更多单词向量化可以看word2vec
		embedding = tf.get_variable("embedding",[VOCAB_SIZE,HIDDEN_SIZE])
		#将原本的batch_size,num_steps的单词ID转换成单词向量[batch_size*num_steps,HIDDEN_SIZE]
		inputs = tf.nn.embedding_lookup(embedding,self.input_data)
		if is_training: inputs = tf.nn.dropout(inputs,KEEP_PROB)

		#定义输出列表，将不同时刻的LSTM结构输出收集起来，通过全连接得到最终的输出
		outputs = []
		#state存储不同batch中的LSTM的状态，初始化为0
		state = self.initial_state
		with tf.variable_scope("RNN"):
			for time_step in range(num_steps):
				if time_step > 0:tf.get_variable_scope.reuse_variables()

				cell_output,state = cell(inputs[:,time_step,:],state)
				outputs.append(cell_output)

			#将输出队列展开 [batch*numsteps,hidden_size]
			output = tf.reshape(tf.concat(1,outputs),[-1,HIDDEN_SIZE])

			#全连接
			weight = tf.get_variable("weight",[HIDDEN_SIZE,VOCAB_SIZE])
			bias = tf.get_variable("bias",[VOCAB_SIZE])
			logit = tf.matmul(outputs,bias) + bias

			#定义交叉熵损失函数
			loss = tf.nn.seq2seq.sequence_loss_by_example(
				[logit],
				[tf.reshape(self.targets,[-1])],#这里讲[batch_size,num_steps]压缩为一维数组
				[tf.ones([batch_size*num_steps],dtype=tf.float32)]#不同batch的不同时刻权值相同都是1
				)
			#计算每个batch的损失函数
			self.cost = rf.reduce_sum(loss) / batch_size
			self.final_state = state

			#只在训练中定义反向传播
			if not is_training: return 
			trainable_variables = tf.trainable_variables()
			#clip_by_gloable_norm控制梯度大小避免梯度膨胀
			grads,_ = tf.clip_by_gloable_norm(tf.gradient(self.cost,trainable_variables),MAX_GRAD_NORM)

			optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
			self.train_op = optimizer.apply_gradients(zip(grads,trainable_variables))

#在给定的模型在数据上运行train_op并返回在全部数据上的perplexity值
def run_epoch(session,model,data,train_op,output_log):
	total_cost = 0
	iters = 0
	state = session.run(model.initial_state)
	#使用当前的训练或者测试模型
	for step,(x,y) in enumerate(reader.ptb_iterator(data,model.batch_size,model.num_steps)):
		#在当前batch上计算train_op并计算损失值，交叉熵损失函数的值就是下一个单词为给定单词的概率
		cost,state,_ = sess.run([model.cost,model,final_state,train_op],{model.input_data:x,model.targets:y,model.initial_state:state})
		total_costs += cost
		iters += model.num_steps

		#只有在训练的时候输出日志
		if output_log and step%100==0:
			print("After %d steps,perplexity is %.3f" % (step,np.exp(total_costs/iters)))
		return np.exp(total_costs/iters)

def main(_):
	#获得原始数据
	train_data,test_data,valid_data = reader.ptb_raw_data(DATA_PATH)

	#初始化参数
	initializer = tf.random_uniform_initializer(-0.05,0.05)

	#定义训练用的循环神经网络模型
	with tf.variable_scope("language_model",reuse=None,initializer=initializer):
		train_model = PTBModel(True,TRAIN_BATCH_SIZE,TRAIN_NUM_STEP)

	#定义评测用的循环神经网络
	witf tf.variable_scope("language_model",reuse=None,initializer=initializer):
		eval_model = PTBModel(False,EVAL_BATCH_SIZE,EVAL_NUM_STEP)

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	with tf.Sesion(config = config) as sess:
		tf.global_variables_initializer().run()
		#使用训练数据集训练模型
		for i in range(NUM_EPOCH):
			print("Iteration: %d" % NUM_EPOCH)
			run_epoch(sess,train_model,train_data,train_op,True)

			valid_perplexity = run_epoch(sess,eval_model,valid_data,np.no_op(),False)
			
			print("Epoch:%d Validation perplexity : %.3f" % (i,valid_perplexity))

		#最后使用test
		test_perplexity = run_epoch(sess,eval_model,test_data,np.no_op(),False)
		print("Test perplexity: %.3f" % test_perplexity)

if __name__ == "__main__":
	tf.app.run()



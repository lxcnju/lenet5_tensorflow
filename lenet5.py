#-*- coding:utf-8 -*-

'''
author: lixinchun
MNIST数据集搭建LeNet5
python3.6 + thensorflow
'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from matplotlib import pyplot as plt

LEARNING_RATE = 0.001     # 学习率
EPOCHES = 20              # 训练轮数
BATCH_SIZE = 128          # 批大小

IMAGE_SIZE = 28           # mnist图片大小
IMAGE_CHANNELS_NUM = 1    # mnist图片通道数目

CLASS_NUM = 10            # 类别数目

CONV1_KERNEL_SIZE = 5     # 卷积层1的核大小
CONV1_KERNEL_NUMS = 6     # 卷积层1的核的数目

CONV2_KERNEL_SIZE = 5     # 卷积层2的核大小
CONV2_KERNEL_NUMS = 16    # 卷积层2的核的数目

FULL1_NUM = 120          # 全连接层1神经元数目
FULL2_NUM = 84           # 全连接层2神经元数目

# 加载数据集
def load_mnist_data(NORMALIZE_TYPE = None):
    mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
    if NORMALIZE_TYPE:
        if NORMALIZE_TYPE == "max_min":
            scaler = MinMaxScaler()
        elif NORMALIZE_TYPE == "standard":
            scaler = StandardScaler()
        scaler.fit_transform(mnist.test.images)
        scaler.fit_transform(mnist.test.images)
    return mnist

# 权重
def get_weight(shape, REGULARIZER_TYPE, REGULARIZER):
    w = tf.Variable(tf.truncated_normal(shape, stddev = 0.1))
    if REGULARIZER_TYPE == 'l1':
        tf.add_to_collection("losses", tf.contrib.layers.l1_regularizer(REGULARIZER)(w))
    elif REGULARIZER_TYPE == 'l2':
        tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(REGULARIZER)(w))
    return w

# 偏置
def get_bias(shape):
    b = tf.Variable(tf.zeros(shape))
    return b

def lenet5_main(mnist, ACTIVATION_FUNC = 'relu', REGULARIZER_TYPE = None, REGULARIZER = None):
    ###################### 搭建计算图 #######################
    # 占位符
    x = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS_NUM])
    y = tf.placeholder(tf.float32, [None, CLASS_NUM])
    
    # 卷积层1 + 池化层1
    conv1_w = get_weight([CONV1_KERNEL_SIZE, CONV1_KERNEL_SIZE, IMAGE_CHANNELS_NUM, CONV1_KERNEL_NUMS], REGULARIZER_TYPE, REGULARIZER)
    conv1_b = get_bias([CONV1_KERNEL_NUMS])
    conv1 = tf.nn.conv2d(x, conv1_w, strides = [1,1,1,1], padding = 'SAME')
    if ACTIVATION_FUNC == 'relu':
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_b))
    elif ACTIVATION_FUNC == 'sigmoid':
        relu1 = tf.nn.sigmoid(tf.nn.bias_add(conv1, conv1_b))
    elif ACTIVATION_FUNC == 'tanh':
        relu1 = tf.nn.tanh(tf.nn.bias_add(conv1, conv1_b))
    pool1 = tf.nn.max_pool(relu1, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID')
    
    # 卷积层2 + 池化层2
    conv2_w = get_weight([CONV2_KERNEL_SIZE, CONV2_KERNEL_SIZE, CONV1_KERNEL_NUMS, CONV2_KERNEL_NUMS], REGULARIZER_TYPE, REGULARIZER)
    conv2_b = get_bias([CONV2_KERNEL_NUMS])
    conv2 = tf.nn.conv2d(pool1, conv2_w, strides = [1,1,1,1], padding = 'VALID')
    if ACTIVATION_FUNC == 'relu':
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_b))
    elif ACTIVATION_FUNC == 'sigmoid':
        relu2 = tf.nn.sigmoid(tf.nn.bias_add(conv2, conv2_b))
    elif ACTIVATION_FUNC == 'tanh':
        relu2 = tf.nn.tanh(tf.nn.bias_add(conv2, conv2_b))
    pool2 = tf.nn.max_pool(relu2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID')
    
    # 全连接层1
    pool2_shape = pool2.get_shape().as_list()
    full_size = pool2_shape[1] * pool2_shape[2] * pool2_shape[3]
    full_x = tf.reshape(pool2, [-1, full_size])
    full1_w = get_weight([full_size, FULL1_NUM], REGULARIZER_TYPE, REGULARIZER)
    full1_b = get_bias([FULL1_NUM])
    if ACTIVATION_FUNC == 'relu':
        full1_x = tf.nn.relu(tf.matmul(full_x, full1_w) + full1_b)
    elif ACTIVATION_FUNC == 'sigmoid':
        full1_x = tf.nn.sigmoid(tf.matmul(full_x, full1_w) + full1_b)
    elif ACTIVATION_FUNC == 'tanh':
        full1_x = tf.nn.tanh(tf.matmul(full_x, full1_w) + full1_b)
    
    # 全连接层2
    full2_w = get_weight([FULL1_NUM, FULL2_NUM], REGULARIZER_TYPE, REGULARIZER)
    full2_b = get_bias([FULL2_NUM])
    if ACTIVATION_FUNC == 'relu':
        full2_x = tf.nn.relu(tf.matmul(full1_x, full2_w) + full2_b)
    elif ACTIVATION_FUNC == 'sigmoid':
        full2_x = tf.nn.sigmoid(tf.matmul(full1_x, full2_w) + full2_b)
    elif ACTIVATION_FUNC == 'tanh':
        full2_x = tf.nn.tanh(tf.matmul(full1_x, full2_w) + full2_b)
    
    # gaussian 层
    full3_w = get_weight([FULL2_NUM, CLASS_NUM], REGULARIZER_TYPE, REGULARIZER)
    full3_b = get_bias([CLASS_NUM])
    y_pre = tf.matmul(full2_x, full3_w) + full3_b
    
    # 准确率
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_pre, 1), tf.argmax(y, 1)), tf.float32))
    
    # 训练误差
    mse_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = y_pre, labels = tf.argmax(y, 1)))
    tf.add_to_collection('losses', mse_loss)
    loss = tf.add_n(tf.get_collection('losses'))
    
    # 训练
    train_op = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)
    
    ############## 开始训练测试 #############
    # 记录训练过程中训练集和测试集上的损失和准确率变化
    train_losses = []      # 训练集loss，取训练集前20%进行预测
    test_losses = []       # 测试集loss, 50%
    train_accus = []       # 训练集准确率, 20%
    test_accus = []        # 测试集准确率, 50%
    # 训练集和测试集(部分),为了加快计算速度
    part_train_num = int(mnist.train.images.shape[0] * 0.2)
    part_test_num = int(mnist.test.images.shape[0] * 0.5)
    part_train_x, part_train_y = mnist.train.images[0:part_train_num, :], mnist.train.labels[0:part_train_num, :]
    part_test_x, part_test_y = mnist.test.images[0:part_test_num, :], mnist.test.labels[0:part_test_num, :]
    part_train_x = np.reshape(part_train_x, [-1, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS_NUM])
    part_test_x = np.reshape(part_test_x, [-1, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS_NUM])
    steps_list = []
    
    with tf.Session() as sess:
        # 训练
        sess.run(tf.global_variables_initializer())    # 初始化
        all_steps = int(mnist.train.num_examples/BATCH_SIZE) * EPOCHES
        print(all_steps)
        for i in range(all_steps):
            batch_x, batch_y = mnist.train.next_batch(BATCH_SIZE)
            batch_x = np.reshape(batch_x, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS_NUM])
            _, loss_value = sess.run([train_op, loss], feed_dict = {x : batch_x, y : batch_y})
            if i % 200 == 0:
                print("Step i = ", i, "Loss value = ", loss_value)
                train_loss, train_accu = sess.run([loss, accuracy], feed_dict = {x : part_train_x, y : part_train_y})
                test_loss, test_accu = sess.run([loss, accuracy], feed_dict = {x : part_test_x, y : part_test_y})
                train_losses.append(train_loss)
                test_losses.append(test_loss)
                train_accus.append(train_accu)
                test_accus.append(test_accu)
                steps_list.append(i)
        # 绘制训练测试损失和准确率变化
        plt.figure()
        plt.plot(steps_list, train_losses, steps_list, test_losses)
        plt.legend(["Train Loss", "Test Loss"])
        plt.xlabel("Train Step")
        plt.ylabel("Loss")
        plt.show()
        
        plt.figure()
        plt.plot(steps_list, train_accus, steps_list, test_accus)
        plt.legend(["Train Accuracy", "Test Accuracy"])
        plt.xlabel("Train Step")
        plt.ylabel("Accuracy")
        plt.show()
        
        # 测试
        test_x, test_y = mnist.test.images, mnist.test.labels
        test_x = np.reshape(test_x, [-1, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS_NUM])
        accuracy = sess.run([accuracy], feed_dict = {x : test_x, y : test_y})
        print("Test accuracy = ", accuracy)
    print("Done!")

print("##############################")
print('正常运行...')
mnist = load_mnist_data()
lenet5_main(mnist)
print("##############################")

#### 尝试不同的数据标准化方式
for NORMALIZE_TYPE in ["max_min", "standard"]:
    tf.reset_default_graph()
    print("##############################")
    print("数据归一化方式 = ", NORMALIZE_TYPE)
    mnist = load_mnist_data(NORMALIZE_TYPE = NORMALIZE_TYPE)
    lenet5_main(mnist)
    print("##############################")

#### 尝试不同的激活函数
for ACTIVATION_FUNC in ["sigmoid", "tanh"]:
    tf.reset_default_graph()
    print("##############################")
    print("激活函数 = ", ACTIVATION_FUNC)
    mnist = load_mnist_data()
    lenet5_main(mnist, ACTIVATION_FUNC = ACTIVATION_FUNC)
    print("##############################")


#### 尝试不同的正则化
for REGULARIZER_TYPE in ["l2", "l1"]:
    for REGULARIZER in [0.0005, 0.001, 0.01]:
        tf.reset_default_graph()
        print("##############################")
        print("正则化方式 = ", REGULARIZER_TYPE, "正则化系数 = ", REGULARIZER)
        mnist = load_mnist_data()
        lenet5_main(mnist, REGULARIZER_TYPE = REGULARIZER_TYPE, REGULARIZER = REGULARIZER)
        print("##############################")

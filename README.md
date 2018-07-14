# lenet5_tensorflow
Using tensorflow to build the lenet-5 network proposed by Yann LeCun.

利用神经网络搭建LeNet-5，由Yann LeCun于1998年在论文Gradient-based learning applied to document recognition中提出的，它是第一个成功应用于数字识别问题的卷积神经网络。神经网络的搭建采用Python3.6 + Tensorflow1.3实现。

* 代码架构
* 原理解析
  * LeNet-5
 

## 代码架构
 * lenet5.py 搭建网络，采用MNIST数据集进行训练
 
## 原理解析
 * LeNet-5 <br>
 LeNet-5有七层，包括卷积层1，池化层1，卷积层2，池化层2，全连接层1，全连接层2，输出层。其网络结构图如下：<br>
 ![Lenet5](https://github.com/lxcnju/lenet5_tensorflow/blob/master/lenet5.png) <br>
 lenet5.py里面给出了不同设置的训练，比如对数据进行标准化、采用不同激活函数、采用不同正则化方式和正则化系数等等。下面给出默认设置：不对数据集标准化，采用ReLu激活函数，不采用正则化项，其余参数设置见代码。下面是训练集上训练20遍过程中模型在训练集和测试集上的准确率和损失：<br>
 ![Lenet5 Loss](https://github.com/lxcnju/lenet5_tensorflow/blob/master/loss.png)
 ![Lenet5 Accuracy](https://github.com/lxcnju/lenet5_tensorflow/blob/master/accuracy.png) <br>
 

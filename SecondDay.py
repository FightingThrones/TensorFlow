"""
 目的：
 Tensorflow学习代码练习：
 1.相关模块的基本操作
 2.神经网络逼近股票价格小例子的实现
 @Author:  Thrones
 GitHub:   https://github.com/FightingThrones/TensorFlow
 Email:    yang18885672964@gmail.com
"""
# import tensorflow as tf
#
# data1=tf.placeholder(tf.float32)
# data2=tf.placeholder(tf.float32)
# dataAdd=tf.add(data1,data2)
# with tf.Session() as sess:
#     #1 dataAdd 2 data (feed_dict={1:6,2:2})
#     print(sess.run(dataAdd,feed_dict={data1:6,data2:2}))
# print("End")

#矩阵操作
#一行两列
#[[6,7]]

# import tensorflow as tf
# #定义一个一行两列的矩阵
# data1=tf.constant([[6,7]])
# #定义一个两行一列的矩阵
# data2=tf.constant([[2],
#                    [3]])
# data3=tf.constant([3,3])
# data4=tf.constant([[3,4],
#                    [5,6],
#                    [7,8]])
# #打印矩阵的维度
# print(data4.shape)
# with tf.Session() as sess:
#     #打印矩阵全部
#     print(sess.run(data4))
#     #打印某一行
#     print(sess.run(data4[0]))
#     #打印某一列
#     print(sess.run(data4[:,0]))
#     #打印具体位置
#     print(sess.run(data4[0,0]))
#
#     matMul=tf.matmul(data1,data2)
#     matMul2=tf.multiply(data1,data2)
#     matAdd=tf.add(data1,data3)
#     with tf.Session() as sess:
#         print(sess.run(matMul))
#         print(sess.run(matAdd))
#         print("End")
#         print(sess.run(matMul2))
#         print("End")
#         #一次打印多个内容
#         print(sess.run([matMul,matAdd]))



# import tensorflow as tf
# #定义一些空矩阵
# mat0=tf.constant([[0,0,0],
#                   [0,0,0]])
# #定义一个2行3列的空矩阵
# mat1=tf.zeros([2,3])
# #定义一个3行2列的1矩阵
# mat2=tf.ones([3,2])
# #定义一个填充矩阵
# mat3=tf.fill([2,3],15)
# mat4=tf.zeros_like(mat3)
# #平均分割
# mat5=tf.linspace(0.0,2.0,11)
# #定义一个随机矩阵
# mat6=tf.random_uniform([2,3],-3,8)
# with tf.Session() as sess:
#     #print(sess.run(mat0))
#      print(sess.run(mat1))
#      print(sess.run(mat2))
#      print(sess.run(mat3))
#      print(sess.run(mat4))
#      print("######")
#      print(sess.run(mat5))
#      print("#######")
#      print(sess.run(mat6))

#numpy模块的使用
# import numpy as np
# data1=np.array([1,2,3,4,5])
# print(data1)
# data2=np.array([[2,3],
#                [5,3]])
# print(data2)
# #打印出data2的维度
# print(data1.shape,data2.shape)
# #打印一个0矩阵和单位矩阵
# print(np.zeros([2,3]),np.ones([2,3]))
# #修改矩阵
# data2[1,0]=5
# print(data2)
# print(data2[1,1])
#
# #基本运算
# data3=np.ones([2,3])
# print(data3*2)
# print(data3/3)
# print(data3+8)
# print(data3-1)
#
# #矩阵+ *
# data4=np.array([[2,3,4],
#                [3,5,8]])
# print(data3+data4)
# print(data3*data4)

#matplotlib库的使用
# import numpy as np
# import matplotlib.pyplot as plt
# x=np.array([1,2,3,4,5,6])
# y=np.array([3,5,8,8,2,8])
# #绘制折线图
# #参数1 x 参数2 y 参数3 颜色
# plt.plot(x,y,'r')
# #plt.show()
# #第四个参数 线的宽度
# plt.plot(x,y,'g',lw=10)
# #plt.show()
#
# #折线  饼状 柱状
# x=np.array([1,2,3,4,5,6,7,8])
# y=np.array([23,25,42,21,23,21,23,52])
# #第三个参数 柱状图占用宽的比例
# plt.bar(x,y,0.2,alpha=1,color='b')
# plt.show()

#神经网络逼近股票价格例子
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#时间
date=np.linspace(1,15,15)
#股票收盘价
endPrice = np.array([2511.90,2538.68,2591.66,2732.98,2701.69,2701.29,2678.67,2726.50,2681.50,2739.17,2715.07,2823.53,2864.90,2019.68,2526.32])
#股票开盘价
beginPrice= np.array([2438.71,2500.88,2534.95,2512.52,2594.04,2743.26,2697.47,2695.24,2678.23,2722.13,2674.93,2744.13,2717.46,2832.73,2653.20])
print(date)
#绘图
plt.figure()
for i in range(0,15):
    #柱状图
    dateOne=np.zeros([2])
    #当天的开盘价和收盘价日期都一样，所以都赋值为i
    #存放开盘价日期
    dateOne[0]=i;
    #存放收盘价日期
    dateOne[1]=i;

    #用来存放当天的开盘价和收盘价
    priceOne=np.zeros([2])
    #用来存放当天的开盘价
    priceOne[0]=beginPrice[i]
    #用来存放当天的收盘价
    priceOne[1]=endPrice[i]

    #如果当天收盘价大于开盘价，绘制图形，用红色表示
    if endPrice[i]>beginPrice[i]:
        plt.plot(dateOne,priceOne,'r',lw=8)

    #当天的收盘价小于开盘价，用绿色图形绘制
    else:
        plt.plot(dateOne,priceOne,'g',lw=8)
#图形显示
#plt.show()

#神经网络相关知识：
#输入层  隐藏层  输出层
#隐藏层可以有多层

#构建的神经网络结构：
#输入层矩阵  隐藏层矩阵 输出层矩阵
#15x1       1x10           15x1
#输入:天数  A*w1+b1=B
#A:输入矩阵 w1:权重  b1:偏置矩阵
#输出:每天股价  B*w2+b2=C
#B:隐藏矩阵  w2：权重 b2:偏置矩阵  C:输出矩阵

#第一层->>第二层 A*w1+b1=B
#第二层->>第三层 B*w2+b2=C
# A:输入层 B:隐藏层  C:输出层
# w1 1x10  w2 10x1
# b1 1x10  b2 15x1
#A(15x1)*w1(1x10)+b1(1*10)=B(15x10)
#B(15x10)*w2(10x1)+b2(15x1)=C(15x1)


#A 输入层处理
#将时间和价格进行归一化处理
dateNormal=np.zeros([15,1])
priceNormal=np.zeros([15,1])
for i in range(0,15):
    dateNormal[i,0]=i/14.0;
    priceNormal[i,0]=endPrice[i]/3000.0;
x=tf.placeholder(tf.float32,[None,1])
y=tf.placeholder(tf.float32,[None,1])


#B隐藏层处理
#定义一个一行10列的随机矩阵，最小值为0 最大值为1
#要涉及到反向传播,所以w1为变量
w1=tf.Variable(tf.random_uniform([1,10],0,1))
b1=tf.Variable(tf.zeros([1,10]))
wb1=tf.matmul(x,w1)+b1
#rele激励函数
layer1=tf.nn.relu(wb1)

#C输出层处理
#权重矩阵
w2=tf.Variable(tf.random_uniform([10,1],0,1))
#偏置矩阵
b2=tf.Variable(tf.zeros([15,1]))
wb2=tf.matmul(layer1,w2)+b2
layer2=tf.nn.relu(wb2)
#标准差计算
loss=tf.reduce_mean(tf.square(y-layer2))

#梯队下降法
train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(0,10000):
        sess.run(train_step,feed_dict={x:dateNormal,y:priceNormal})

    #w1 w2 b1 b2 A+wb->>layer2
    pred=sess.run(layer2,feed_dict={x:dateNormal})
    predPrice=np.zeros([15,1])
    for i in range(0,15):
        predPrice[i,0]=(pred*3000)[i,0]
    plt.plot(date,predPrice,'b',lw=1)
plt.show()
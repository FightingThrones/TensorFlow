"""
 目的：    Tensorflow学习代码练习
 @Author:  Thrones
 GitHub:
 Email:
"""

#tensorflow安装是否成功测试代码
# import tensorflow as tf
# hello=tf.constant('hello tf:')
# sess=tf.Session()
# print(sess.run(hello))

#opencv安装是否成功测试代码
# import cv2
# print("hello opencv")

#图片的读取和展示
#包括以下步骤：
# 1:文件的读取
# 2:封装格式解析
# 3:数据解码
# 4:数据加载
# import cv2
# #cv2.imread()传入的两个参数，第一个为图片名称
# #第二个参数为0时为灰度图片，为1时为彩色图片
# img=cv2.imread('./images/6.jpg',1)
# #参数1：窗体名称  参数2：传入的图片
# cv2.imshow('image',img)
# cv2.waitKey(0)

#图片写入操作
# import cv2
# #读入图片并进行灰度变换
# img=cv2.imread('./images/6.jpg',0)
# #将图片写入下列文件中并命名为ImageTest.jpg
# cv2.imwrite('./images/ImageTest.jpg',img)

# #不同图片质量保存
# import cv2
# img=cv2.imread('.images/6.jpg',1)
# #jpg进行图片压缩，最后一个参数范围：0-100 有损压缩 数字越小压缩比越高
# cv2.imwrite('./images/ImgTest.jpg',img,[cv2.IMWRITE_JPEG_QUALITY,0])
# #png进行图片压缩，最后一个参数范围：0-9 无损压缩，数字越小压缩比越低
# cv2.imwrite('./images/ImgTest.png',img,[cv2.IMWRITE_JPEG_QUALITY,0])

#像素操作基础知识笔记
# 1.像素
# 2.RGB
# 3.颜色深度 8bit 0-255
# 4.图片宽高 w h 640*480
# 5.1.4M=720*547*3*8bit/8  (B) =1.14M
# 6.RGB 阿尔法通道描述透明度信息
# 7.RGB bgr
# 8.bgr b g

#像素读取写入操作
import cv2
img=cv2.imread('images/9.jpg',1)
#像素的读取返回存入一个元组中
#b g r 三种颜色通道：蓝色 绿色 红色
(b,g,r)=img[100,100]
#打印图片的[100,100]这个坐标点的b g r 值
print(b,g,r)

#在图片上绘制一条直线
#[10 ,100]--[110,100]
for i in range(1,100):
    img[10+i,100]=[0,0,255]
#显示图片
cv2.imshow('image',img)
cv2.waitKey(0)

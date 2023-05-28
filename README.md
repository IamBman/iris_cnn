# iris_cnn

#### 介绍
虹膜图像分类，同时测试神经网络压缩相关技术

#### 软件架构
软件架构说明


#### 安装教程

能跑pytorch，有cuda就行

#### 使用说明

1.  enrollment_data是用于训练的数据集，test_data是测试用的数据集，他们都来自CASIA-Iris-Thousand数据集，取自其中的第750~999号受试者。预处理使用基于hought变换的虹膜定位方案。通过极坐标展开将图像展开为256*64像素。再通过图像增强，拼接成256*256像素，得到最终的训练图像。训练集取7张，测试集取3张。图像预处理的效果如下：
![预处理](S5750L00.png "Magic Gardens")
2.  resnet18是标准的残差神经网络形式，224*224像素,三通道
3.  resnet8是简化版的残差神经网络，只保留了通道倍增的层，32*32像素,单通道
4.  后续会添加使用ShuffleNet的版本
5.  quant1是resnet8的量化版
6.  quant2是ShufflenNet的量化版
7.  tool.py是一些工具函数
8.  data是制作数据集的函数，resnet18不用这个公共函数

#### 致谢

1.  感谢学长提供的数据集，他的预处理程序抄谁的我忘了
2.  时间太久远，resnet第一版的代码是用一个1000类32*32图片分类的resnet源码改过来的，具体是哪个大佬开源的我也忘了，总之感谢这个大佬。
3.  感谢指导老师和三个学弟的帮助，没有你们我毕不了业



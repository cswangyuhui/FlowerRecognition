# FlowerRecognition
```
本项目先通过网络爬虫在百度图库上抓取十种不同的花的图片作为数据集，然后通过迁移学习重新训练MobileNet
神经网络模型，最终在验证数据集上的准确率为百分之89.6，最后将训练好的模型转化成可以在手机上运行的模型
(TensorFlow Lite Format)，最后完成物体识别APP的设计与实现。
```
## 数据集
[百度网盘链接](https://pan.baidu.com/s/1HA9M2h2JpKDY8uQHCbvxxQ) 提取码:ti4m

# 文件介绍

|文件名|作用|
|:---|:---|
|模型文件文件夹|内含迁移学习得到的模型文件mobile_graph.pb和TensorFlow Lite Format格式的模型文件graph.tflite|
|download.py|从百度图库下载图片的网络爬虫程序|
|input.py|图片预处理，包括随机翻转图片，随机调整图片的亮度、色相、对比度和饱和度，随机截取图像的一部分|
|retrain.py|重新训练MobileNet神经网络的迁移学习程序|
|Introduction.pptx|项目介绍PPT|
|MobileNet.pdf|介绍MobileNet神经网络的论文|
|tflite|安卓APP程序源码( 用Android Studio打开)|

# 数据预处理结果

<img src="https://github.com/cswanngyuhui/FlowerRecognition/blob/master/result/PreProcess1.png" width="550" height="400"/>
<br>
<img src="https://github.com/cswanngyuhui/FlowerRecognition/blob/master/result/PreProcess2.png" width="550" height="400"/>

# 安卓APP截图

<img src="https://github.com/tianYingDao/FlowerRecognition/blob/master/result/result.png" width="550" height="400"/>

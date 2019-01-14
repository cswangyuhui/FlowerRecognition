#coding=utf-8
import tensorflow as tf
import numpy as np
import os.path
import glob
import random
from tensorflow.python.platform import gfile
import matplotlib.pyplot as plt

INPUT_DATA = '/Users/wangyuhui/Desktop/flower_photos'#数据集路径
CREATED_IMAGES = '/Users/wangyuhui/Desktop/created_flower_photos'#存储图片数据预处理程序的结果的目录
FILE_COUNT = 0

def delFile(path):
    for root, dirs, files in os.walk(path):
        for file in files:
            os.remove(os.path.join(root,file))
        for dir in dirs:
            delFile(os.path.join(root,dir))
            os.rmdir(os.path.join(root,dir))


def distort_color(image,color_ordering=0):
    if color_ordering == 0:
        image = tf.image.random_brightness(image,max_delta=32./255.)
        image = tf.image.random_saturation(image,lower=0.5,upper=1.5)
        image = tf.image.random_hue(image,max_delta=0.2)
        image = tf.image.random_contrast(image,lower=0.5,upper=1.5)
    elif color_ordering == 1:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image,max_delta=32./255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image,max_delta=0.2)
    elif color_ordering == 2:
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    elif color_ordering == 3:
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
    return tf.clip_by_value(image,0.0,1.0)

def preprocess_for_train(image,box):
    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image,dtype=tf.float32)
    if box is None:
        distorted_image = tf.image.random_flip_left_right(image)
        distorted_image = distort_color(distorted_image,np.random.randint(4))
        return  distorted_image
    else:
        bbox_begin,bbox_size,_ = tf.image.sample_distorted_bounding_box(tf.shape(image),bounding_boxes=box)
        distorted_image = tf.slice(image,bbox_begin,bbox_size)
        distorted_image = tf.image.random_flip_left_right(distorted_image)
        distorted_image = distort_color(distorted_image, np.random.randint(4))
        return distorted_image

def ensure_dir_exists(dir_name):
  """如果某个文件夹不存在，新建文件夹"""
  if not os.path.exists(dir_name):
    os.makedirs(dir_name)

def preprocessImages(sess):
    global FILE_COUNT
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]
    is_roor_dir = True
    for sub_dir in sub_dirs:
        if is_roor_dir:
            is_roor_dir = False
            continue
        dir_name = os.path.basename(sub_dir)
        filename = os.path.join(INPUT_DATA, dir_name, '*')
        file_list = []
        file_list.extend(glob.glob(filename))
        FILE_COUNT += len(file_list)
        created_images_path = os.path.join(CREATED_IMAGES,dir_name)
        ensure_dir_exists(created_images_path)
        delFile(created_images_path)
        fileCount = 0
        for fn in file_list:
            print "正在预处理",fn
            image_data = gfile.FastGFile(fn, 'rb').read()
            with tf.gfile.GFile(created_images_path + "/" + str(fileCount) + ".jpg", 'wb') as f:
                f.write(image_data)
            fileCount += 1
            example = tf.image.decode_jpeg(image_data)
            # 若要随机截取图像的一部分则取消下面一行注释
            bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
            if example.dtype != tf.float32:
                example = tf.image.convert_image_dtype(example, dtype=tf.float32)
                #plt.imshow(example.eval())
                #plt.show()
            #preprocess_images  = []
            for t in range(6):
                # 若要随机截取图像的一部分则取消下面一行注释
                example = preprocess_for_train(example, bbox)
                #example = preprocess_for_train(example, None)
                with tf.gfile.FastGFile(created_images_path + "/" + str(fileCount) + ".jpg", 'wb') as f:
                    f.write(sess.run(tf.image.encode_jpeg(tf.image.convert_image_dtype(example, dtype=tf.uint8))))
                #preprocess_images.extend(sess.run(tf.image.encode_jpeg(tf.image.convert_image_dtype(example, dtype=tf.uint8))))
                #若要显示预处理的图片，取消下面的注释
                # plt.imshow(example.eval())
                # plt.show()
                fileCount += 1

if __name__=='__main__':
    with tf.Session() as sess:
        preprocessImages(sess)
        print "预处理的图片总数为：", FILE_COUNT

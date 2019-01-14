#coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import os.path
import random
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile

CACHE_DIR = '/Users/wangyuhui/Desktop/training_workingspace/bottleneck'
INPUT_DATA = '/Users/wangyuhui/Desktop/flower_photos'
VALIDATION_PERCENTAGE = 10
TEST_PERCENTAGE = 10
MODEL_DIR = '/Users/wangyuhui/Desktop/training_workingspace/models/mobilenet_0.50_224/mobilenet_v1_0.50_224'
MODEL_FILE = 'frozen_graph.pb'
BOTTLENECK_TENSOR_SIZE = 1001
BOTTLENECK_TENSOR_NAME = 'MobilenetV1/Predictions/Reshape:0'
RESIZED_INPUT_TENSOR_NAME = 'input:0'
LEARNING_RATE = 0.01
STEPS = 100
BATCH = 100
FINAL_TENSOR_NAME='final_result'
OUTPUT_LABELS='/Users/wangyuhui/Desktop/training_workingspace/labels.txt'
OUTPUT_GRAPH='/Users/wangyuhui/Desktop/training_workingspace/mobile_graph.pb'
SUMMARIES_DIR='/Users/wangyuhui/Desktop/training_workingspace/summaries'
EVAL_STEP_INTERVAL=100
INPUT_MEAN = 127.5
INPUT_STD = 127.5
GRAPH_WIDTH = 224
GRAPH_HEIGHT = 224
GRAPH_DEPTH = 3

def create_image_lists(image_dir,testing_percent,validation_percent):
    """从输入图片建立训练数据集、测试数据集和验证数据集"""
    if not gfile.Exists(image_dir):
        tf.logging.error("未找到图片路径" + image_dir)
        return None
    result = collections.OrderedDict()
    sub_dirs = [os.path.join(image_dir,item) for item in gfile.ListDirectory(image_dir)]
    sub_dirs = sorted(item for item in sub_dirs if gfile.IsDirectory(item))
    for sub_dir in sub_dirs:
        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
        file_list = []
        dir_name = os.path.basename(sub_dir)
        # print dir_name
        if dir_name == image_dir:
            continue
        for extension in extensions:
            file_glob = os.path.join(INPUT_DATA, dir_name, '*.' + extension)
            # print file_glob
            file_list.extend(gfile.Glob(file_glob))
        if not file: continue
        label_name = dir_name.lower()
        training_images = []
        testing_images = []
        validation_images = []
        validation_images_len = int(len(file_list) * validation_percent / 100)
        testing_images_len = int(len(file_list) * validation_percent / 100) + \
                             int(len(file_list) * testing_percent / 100)
        addCount = 1
        random.shuffle(file_list)
        for file_name in file_list:
            base_name = os.path.basename(file_name)
            if addCount <= validation_images_len:
                addCount = addCount + 1
                validation_images.append(base_name)
            elif addCount <= testing_images_len:
                addCount = addCount + 1
                testing_images.append(base_name)
            else:
                addCount = addCount + 1
                training_images.append(base_name)
        #print(len(training_images),len(testing_images),len(validation_images))
        result[label_name] = {'dir': dir_name,
                              'training':training_images,
                              'testing':testing_images,
                              'validation':validation_images}
    return result


def get_image_path(image_lists, label_name, index, image_dir, category):
  """返回一个指定label和指定index下的图片路径"""
  if label_name not in image_lists:
    tf.logging.fatal('Label %s 不存在.', label_name)
  label_lists = image_lists[label_name]
  if category not in label_lists:
    tf.logging.fatal('Category %s 数据集不存在.', category)
  category_list = label_lists[category]
  if not category_list:
    tf.logging.fatal('Label %s 在category %s 数据集中没有图片.',
                     label_name, category)
  mod_index = index % len(category_list)
  base_name = category_list[mod_index]
  sub_dir = label_lists['dir']
  full_path = os.path.join(image_dir, sub_dir, base_name)
  return full_path


def get_bottleneck_path(image_lists, label_name, index, bottleneck_dir,
                        category):
  """这个函数通过类别名称、所属数据集和图片编号获取经过模型处理之后的特征向量的地址"""
  return get_image_path(image_lists, label_name, index, bottleneck_dir,
                        category) + '.txt'


def create_model_graph():
  """从一个被保存的GraphDef文件返回一个Graph对象"""
  with tf.Graph().as_default() as graph:
    model_path = os.path.join(MODEL_DIR, MODEL_FILE)
    with gfile.FastGFile(model_path, 'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      bottleneck_tensor, resized_input_tensor = (tf.import_graph_def(
          graph_def,
          name='',
          return_elements=[
              BOTTLENECK_TENSOR_NAME,
              RESIZED_INPUT_TENSOR_NAME
          ]))
  return graph, bottleneck_tensor, resized_input_tensor


def run_bottleneck_on_image(sess, image_data, image_data_tensor,
                            decoded_image_tensor, resized_input_tensor,
                            bottleneck_tensor):
  """得到特征向量,其中image_data为图片，image_data_tensor为重定义大小操作的输入,decoded_image_tensor
      为重定义大小的输出，resized_input_tensor为神经网络的输入"""
  resized_input_values = sess.run(decoded_image_tensor,
                                  {image_data_tensor: image_data})
  bottleneck_values = sess.run(bottleneck_tensor,
                               {resized_input_tensor: resized_input_values})
  bottleneck_values = np.squeeze(bottleneck_values)
  return bottleneck_values

def ensure_dir_exists(dir_name):
  """如果某个文件夹不存在，新建文件夹"""
  if not os.path.exists(dir_name):
    os.makedirs(dir_name)

def create_bottleneck_file(bottleneck_path, image_lists, label_name, index,
                           image_dir, category, sess, jpeg_data_tensor,
                           decoded_image_tensor, resized_input_tensor,
                           bottleneck_tensor):
  """建立bottleneck file"""
  tf.logging.info('提取特征向量： ' + bottleneck_path)
  image_path = get_image_path(image_lists, label_name, index,
                              image_dir, category)
  if not gfile.Exists(image_path):
    tf.logging.fatal('数据文件 %s 不存在', image_path)
  image_data = gfile.FastGFile(image_path, 'rb').read()
  try:
    bottleneck_values = run_bottleneck_on_image(
        sess, image_data, jpeg_data_tensor, decoded_image_tensor,
        resized_input_tensor, bottleneck_tensor)
  except Exception as e:
    raise RuntimeError('处理文件 %s 发生错误 %s' % (image_path,str(e)))
  bottleneck_string = ','.join(str(x) for x in bottleneck_values)
  with open(bottleneck_path, 'w') as bottleneck_file:
    bottleneck_file.write(bottleneck_string)


def get_or_create_bottleneck(sess, image_lists, label_name, index, image_dir,
                             category, bottleneck_dir, jpeg_data_tensor,
                             decoded_image_tensor, resized_input_tensor,
                             bottleneck_tensor):
  """计算或获取一幅图片的bottleneck值"""
  label_lists = image_lists[label_name]
  sub_dir = label_lists['dir']
  sub_dir_path = os.path.join(bottleneck_dir, sub_dir)
  ensure_dir_exists(sub_dir_path)
  bottleneck_path = get_bottleneck_path(image_lists, label_name, index,
                                        bottleneck_dir, category)
  if not os.path.exists(bottleneck_path):
    create_bottleneck_file(bottleneck_path, image_lists, label_name, index,
                           image_dir, category, sess, jpeg_data_tensor,
                           decoded_image_tensor, resized_input_tensor,
                           bottleneck_tensor)
  with open(bottleneck_path, 'r') as bottleneck_file:
    bottleneck_string = bottleneck_file.read()
  did_hit_error = False
  try:
    bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
  except ValueError:
    tf.logging.warning('从文件'+bottleneck_path+'获取特征向量出错，重新建立瓶颈层向量文件。')
    did_hit_error = True
  if did_hit_error:
    create_bottleneck_file(bottleneck_path, image_lists, label_name, index,
                           image_dir, category, sess, jpeg_data_tensor,
                           decoded_image_tensor, resized_input_tensor,
                           bottleneck_tensor)
    with open(bottleneck_path, 'r') as bottleneck_file:
      bottleneck_string = bottleneck_file.read()
    bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
  return bottleneck_values


def cache_bottlenecks(sess, image_lists, image_dir, bottleneck_dir,
                      jpeg_data_tensor, decoded_image_tensor,
                      resized_input_tensor, bottleneck_tensor):
  """预处理得到数据集的特征向量并保存"""
  how_many_bottlenecks = 0
  ensure_dir_exists(bottleneck_dir)
  for label_name, label_lists in image_lists.items():
    for category in ['training', 'testing', 'validation']:
      category_list = label_lists[category]
      for index, unused_base_name in enumerate(category_list):
        get_or_create_bottleneck(
            sess, image_lists, label_name, index, image_dir, category,
            bottleneck_dir, jpeg_data_tensor, decoded_image_tensor,
            resized_input_tensor, bottleneck_tensor)
        how_many_bottlenecks += 1
        if how_many_bottlenecks % 100 == 0:
          tf.logging.info(
              str(how_many_bottlenecks) + ' 瓶颈层向量（特征向量）文件被建立.')


def get_random_cached_bottlenecks(sess, image_lists, how_many, category,
                                  bottleneck_dir, image_dir, jpeg_data_tensor,
                                  decoded_image_tensor, resized_input_tensor,
                                  bottleneck_tensor):
  """从一个特定数据集中的bottleneck_file中获取一部分图片的瓶颈向量"""
  class_count = len(image_lists.keys())
  bottlenecks = []
  ground_truths = []
  filenames = []
  if how_many >= 0:
    # Retrieve a random sample of bottlenecks.
    for unused_i in range(how_many):
      label_index = random.randrange(class_count)
      label_name = list(image_lists.keys())[label_index]
      max_num = len(image_lists[label_name][category])
      image_index = random.randrange(max_num + 1)
      image_name = get_image_path(image_lists, label_name, image_index,
                                  image_dir, category)
      bottleneck = get_or_create_bottleneck(
          sess, image_lists, label_name, image_index, image_dir, category,
          bottleneck_dir, jpeg_data_tensor, decoded_image_tensor,
          resized_input_tensor, bottleneck_tensor)
      ground_truth = np.zeros(class_count, dtype=np.float32)
      ground_truth[label_index] = 1.0
      bottlenecks.append(bottleneck)
      ground_truths.append(ground_truth)
      filenames.append(image_name)
  else:
    # Retrieve all bottlenecks.
    for label_index, label_name in enumerate(image_lists.keys()):
      for image_index, image_name in enumerate(
          image_lists[label_name][category]):
        image_name = get_image_path(image_lists, label_name, image_index,
                                    image_dir, category)
        bottleneck = get_or_create_bottleneck(
            sess, image_lists, label_name, image_index, image_dir, category,
            bottleneck_dir, jpeg_data_tensor, decoded_image_tensor,
            resized_input_tensor, bottleneck_tensor)
        ground_truth = np.zeros(class_count, dtype=np.float32)
        ground_truth[label_index] = 1.0
        bottlenecks.append(bottleneck)
        ground_truths.append(ground_truth)
        filenames.append(image_name)
  return bottlenecks, ground_truths, filenames



def variable_summaries(var):
  """给一个Tensor附加一些summaries用来给TensorBoard可视化"""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)


def add_final_training_ops(class_count, final_tensor_name, bottleneck_tensor,
                           bottleneck_tensor_size):
  """为retrain增添一层新的全连接层和sofrmax层"""
  with tf.name_scope('input'):
    bottleneck_input = tf.placeholder_with_default(
        bottleneck_tensor,
        shape=[None, bottleneck_tensor_size],
        name='BottleneckInputPlaceholder')

    ground_truth_input = tf.placeholder(tf.float32,
                                        [None, class_count],
                                        name='GroundTruthInput')

  layer_name = 'final_training_ops'
  with tf.name_scope(layer_name):
    with tf.name_scope('weights'):
      initial_value = tf.truncated_normal(
          [bottleneck_tensor_size, class_count], stddev=0.001)

      layer_weights = tf.Variable(initial_value, name='final_weights')

      variable_summaries(layer_weights)
    with tf.name_scope('biases'):
      layer_biases = tf.Variable(tf.zeros([class_count]), name='final_biases')
      variable_summaries(layer_biases)
    with tf.name_scope('Wx_plus_b'):
      logits = tf.matmul(bottleneck_input, layer_weights) + layer_biases
      tf.summary.histogram('pre_activations', logits)

  final_tensor = tf.nn.softmax(logits, name=final_tensor_name)
  tf.summary.histogram('activations', final_tensor)

  with tf.name_scope('cross_entropy'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        labels=ground_truth_input, logits=logits)
    with tf.name_scope('total'):
      cross_entropy_mean = tf.reduce_mean(cross_entropy)
  tf.summary.scalar('cross_entropy', cross_entropy_mean)

  with tf.name_scope('train'):
    optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
    train_step = optimizer.minimize(cross_entropy_mean)

  return (train_step, cross_entropy_mean, bottleneck_input, ground_truth_input,
          final_tensor)


def add_evaluation_step(result_tensor, ground_truth_tensor):
  """增加判断预测结果准确度的操作"""
  with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
      prediction = tf.argmax(result_tensor, 1)
      correct_prediction = tf.equal(
          prediction, tf.argmax(ground_truth_tensor, 1))
    with tf.name_scope('accuracy'):
      evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.summary.scalar('accuracy', evaluation_step)
  return evaluation_step, prediction


def save_graph_to_file(sess, graph, graph_file_name):
  """将重新训练的模型保存"""
  output_graph_def = graph_util.convert_variables_to_constants(
      sess, graph.as_graph_def(), [FINAL_TENSOR_NAME])
  with gfile.FastGFile(graph_file_name, 'wb') as f:
    f.write(output_graph_def.SerializeToString())
  return


def prepare_file_system():
  """为TensorBoard可视化建立所需要的目录文件"""
  if tf.gfile.Exists(SUMMARIES_DIR):
    tf.gfile.DeleteRecursively(SUMMARIES_DIR)
  tf.gfile.MakeDirs(SUMMARIES_DIR)


def add_jpeg_decoding(input_width, input_height, input_depth, input_mean,
                      input_std):
  """建立将一个图片decode并重定义大小的操作"""
  jpeg_data = tf.placeholder(tf.string, name='DecodeJPGInput')
  decoded_image = tf.image.decode_jpeg(jpeg_data, channels=input_depth)
  decoded_image_as_float = tf.cast(decoded_image, dtype=tf.float32)
  decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
  resize_shape = tf.stack([input_height, input_width])
  resize_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)
  resized_image = tf.image.resize_bilinear(decoded_image_4d,
                                           resize_shape_as_int)
  offset_image = tf.subtract(resized_image, input_mean)
  mul_image = tf.multiply(offset_image, 1.0 / input_std)
  return jpeg_data, mul_image


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)
  prepare_file_system()
  graph, bottleneck_tensor, resized_image_tensor = (
      create_model_graph())
  image_lists = create_image_lists(INPUT_DATA,TEST_PERCENTAGE,VALIDATION_PERCENTAGE)
  class_count = len(image_lists.keys())
  if class_count == 0:
    tf.logging.error('未在 ' + INPUT_DATA + ' 发现文件夹.')
    return -1
  if class_count == 1:
    tf.logging.error('在 ' + INPUT_DATA + ' 只发现一个文件夹，分类操作需要多个类别文件夹.')
    return -1

  with tf.Session(graph=graph) as sess:
    jpeg_data_tensor, decoded_image_tensor = add_jpeg_decoding(
        GRAPH_WIDTH,GRAPH_HEIGHT,GRAPH_DEPTH,INPUT_MEAN,INPUT_MEAN)
    cache_bottlenecks(sess, image_lists, INPUT_DATA,
                        CACHE_DIR, jpeg_data_tensor,
                        decoded_image_tensor, resized_image_tensor,
                        bottleneck_tensor)

    (train_step, cross_entropy, bottleneck_input, ground_truth_input,
     final_tensor) = add_final_training_ops(
         len(image_lists.keys()), FINAL_TENSOR_NAME, bottleneck_tensor,
         BOTTLENECK_TENSOR_SIZE)

    evaluation_step, prediction = add_evaluation_step(
        final_tensor, ground_truth_input)

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(SUMMARIES_DIR + '/train',
                                         sess.graph)

    validation_writer = tf.summary.FileWriter(
        SUMMARIES_DIR + '/validation')

    init = tf.global_variables_initializer()
    sess.run(init)

    for i in range(STEPS):
      (train_bottlenecks,train_ground_truth, _) = get_random_cached_bottlenecks(
             sess, image_lists, BATCH, 'training',
             CACHE_DIR, INPUT_DATA, jpeg_data_tensor,
             decoded_image_tensor, resized_image_tensor, bottleneck_tensor)

      train_summary, _ = sess.run([merged, train_step],feed_dict={bottleneck_input: train_bottlenecks,
                     ground_truth_input: train_ground_truth})
      train_writer.add_summary(train_summary, i)

      is_last_step = (i + 1 == STEPS)
      if (i % 100) == 0 or is_last_step:
        train_accuracy, cross_entropy_value = sess.run(
            [evaluation_step, cross_entropy],
            feed_dict={bottleneck_input: train_bottlenecks,
                       ground_truth_input: train_ground_truth})
        tf.logging.info('%s: 第 %d 步: 训练准确率 = %.1f%%' %
                        (datetime.now(), i, train_accuracy * 100))
        tf.logging.info('%s: 第 %d 步: 交叉熵 = %f' %
                        (datetime.now(), i, cross_entropy_value))
        validation_bottlenecks, validation_ground_truth, _ = (
            get_random_cached_bottlenecks(
                sess, image_lists, -1, 'validation',
                CACHE_DIR, INPUT_DATA, jpeg_data_tensor,
                decoded_image_tensor, resized_image_tensor, bottleneck_tensor))

        validation_summary, validation_accuracy = sess.run(
            [merged, evaluation_step],
            feed_dict={bottleneck_input: validation_bottlenecks,
                       ground_truth_input: validation_ground_truth})
        validation_writer.add_summary(validation_summary, i)
        tf.logging.info('%s: 第 %d 步: 验证数据集上的准确率  = %.1f%% (N=%d)' %
                        (datetime.now(), i, validation_accuracy * 100,
                         len(validation_bottlenecks)))

    test_bottlenecks, test_ground_truth, test_filenames = (
        get_random_cached_bottlenecks(
            sess, image_lists, -1, 'testing',
            CACHE_DIR, INPUT_DATA, jpeg_data_tensor,
            decoded_image_tensor, resized_image_tensor, bottleneck_tensor))
    test_accuracy, predictions = sess.run(
        [evaluation_step, prediction],
        feed_dict={bottleneck_input: test_bottlenecks,
                   ground_truth_input: test_ground_truth})
    tf.logging.info('最终测试数据集准确率 = %.1f%% (N=%d)' %
                    (test_accuracy * 100, len(test_bottlenecks)))

    save_graph_to_file(sess, graph, OUTPUT_GRAPH)
    with gfile.FastGFile(OUTPUT_LABELS, 'w') as f:
      f.write('\n'.join(image_lists.keys()) + '\n')

if __name__ == '__main__':
    tf.app.run(main=main)
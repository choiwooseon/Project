from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2 as cv
import tensorflow as tf  # tensorflow module
import numpy as np
import os
import matplotlib.pyplot as plt
DATA_dir = 'D:\\total\\' #데이터경로



from datasets import download_and_convert_cifar10
from datasets import download_and_convert_flowers
from datasets import download_and_convert_mnist
from datasets import download_and_convert_parking
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'dataset_name',
    None,
    'The name of the dataset to convert, one of "cifar10", "flowers", "mnist".')

tf.app.flags.DEFINE_string(
    'dataset_dir',
    None,
    'The directory where the output TFRecords and temporary files are saved.')

''' image resize'''

def change():
    for i in range (1,2):
        img = cv.imread(DATA_dir +"test\\" + str(i) + ".jpg", cv.IMREAD_ANYCOLOR)
        new_img = cv.resize(img, (224, 224))
        cv.imwrite(DATA_dir + "resize\\" + str(i) + ".jpg", new_img)
        cv.waitKey(0)
        print("image resize finish")

''' tfrecord 변환'''

def main(_):
  if not FLAGS.dataset_name:
    raise ValueError('You must supply the dataset name with --dataset_name')
  if not FLAGS.dataset_dir:
    raise ValueError('You must supply the dataset directory with --dataset_dir')

  if FLAGS.dataset_name == 'cifar10':
    download_and_convert_cifar10.run(FLAGS.dataset_dir)
  elif FLAGS.dataset_name == 'flowers':
    download_and_convert_flowers.run(FLAGS.dataset_dir)
  elif FLAGS.dataset_name == 'mnist':
    download_and_convert_mnist.run(FLAGS.dataset_dir)
  elif FLAGS.dataset_name == 'parking':
      download_and_convert_parking.run(FLAGS.dataset_dir)
  else:
    raise ValueError(
        'dataset_name [%s] was not recognized.' % FLAGS.dataset_name)

''' cnn 모델링'''


DATA_DIR = "D:\\mirror\\"
#DATA_DIR = "./car/"
TRAINING_SET_SIZE = 536
BATCH_SIZE = 10
IMAGE_SIZE = 224


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# image object from protobuf
class _image_object:
    def __init__(self):
        self.image = tf.Variable([], dtype=tf.string)
        self.height = tf.Variable([], dtype=tf.int64)
        self.width = tf.Variable([], dtype=tf.int64)
        self.label = tf.Variable([], dtype=tf.int32)


def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example, features={
        "image/encoded": tf.FixedLenFeature([], tf.string),
        'image/format': tf.FixedLenFeature([], tf.string),
        "image/height": tf.FixedLenFeature([], tf.int64),
        "image/width": tf.FixedLenFeature([], tf.int64),
        "image/class/label": tf.FixedLenFeature([], tf.int64), })
    image_encoded = tf.cast(features['image/encoded'], tf.string)
    image_raw = tf.image.decode_jpeg(image_encoded, channels=3)
    image_object = _image_object()
    image_object.image = tf.image.resize_image_with_crop_or_pad(image_raw, IMAGE_SIZE, IMAGE_SIZE)
    image_object.height = features["image/height"]
    image_object.width = features["image/width"]
    image_object.label = tf.cast(features["image/class/label"], tf.int64)

    image = tf.image.resize_image_with_crop_or_pad(image_raw, IMAGE_SIZE, IMAGE_SIZE)
    label = tf.cast(features['image/class/label'], tf.int64)

    # return image_object
    return image, label


def flower_input(if_random=True, if_training=True):
    if (if_training):
        filenames = [os.path.join(DATA_DIR, "parking_train_00000-of-0000%d.tfrecord" % i) for i in range(0, 4)]
    else:
        filenames = [DATA_DIR + "parking_train_00000-of-00001.tfrecord"]

    for flower in filenames:
        if not tf.gfile.Exists:
            raise ValueError("Failed to find file: " + flower)
    filename_queue = tf.train.string_input_producer(filenames)
    image, label = read_and_decode(filename_queue)
    image = tf.image.per_image_standardization(image)
    #    image = image_object.image
    #    image = tf.image.adjust_gamma(tf.cast(image_object.image, tf.float32), gamma=1, gain=1) # Scale image to (0, 1)

    if (if_random):
        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int(TRAINING_SET_SIZE * min_fraction_of_examples_in_queue)
        print(
            "Filling queue with %d images before starting to train. " "This will take a few minutes." % min_queue_examples)
        num_preprocess_threads = 1
        image_batch, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=BATCH_SIZE,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * BATCH_SIZE,
            min_after_dequeue=min_queue_examples)
        return image_batch, label_batch
    else:
        image_batch, label_batch = tf.train.batch(
            [image, label],
            batch_size=BATCH_SIZE,
            num_threads=1)
        return image_batch, label_batch


def weight_variable(shape, fan_in, fan_out):
    xaiver = np.random.randn(fan_in / fan_out) / np.sqrt(fan_in / 2.0)
    # initial = tf.truncated_normal(shape, stddev=0.05)
    initial = tf.fill(shape, xaiver)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.02, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def flower_inference(image_batch):
    # W_conv1 = weight_variable([5, 5, 3, 32], 3, 32)
    W_conv1 = tf.get_variable("W_conv1", shape=[5, 5, 3, 32], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    b_conv1 = bias_variable([32])

    x_image = tf.reshape(image_batch, [-1, IMAGE_SIZE, IMAGE_SIZE, 3])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)  # 112

    # W_conv2 = weight_variable([5, 5, 32, 64], 32, 64)
    W_conv2 = tf.get_variable("W_conv2", shape=[5, 5, 32, 64], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)  # 56

    # W_conv3 = weight_variable([5, 5, 64, 128], 64, 128)
    W_conv3 = tf.get_variable("W_conv3", shape=[5, 5, 64, 128],
                              initializer=tf.contrib.layers.xavier_initializer(seed=0))

    b_conv3 = bias_variable([128])

    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3)  # 28

    # W_conv4 = weight_variable([5, 5, 128, 256], 128, 256)
    W_conv4 = tf.get_variable("W_conv4", shape=[5, 5, 128, 256],
                              initializer=tf.contrib.layers.xavier_initializer(seed=0))
    b_conv4 = bias_variable([256])

    h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
    h_pool4 = max_pool_2x2(h_conv4)  # 14

    # W_conv5 = weight_variable([5, 5, 256, 256], 256, 256)
    W_conv5 = tf.get_variable("W_conv5", shape=[5, 5, 256, 256],
                              initializer=tf.contrib.layers.xavier_initializer(seed=0))
    b_conv5 = bias_variable([256])

    h_conv5 = tf.nn.relu(conv2d(h_pool4, W_conv5) + b_conv5)
    h_pool5 = max_pool_2x2(h_conv5)  # 7

    # W_fc1 = weight_variable([7*7*256, 2048], 7*7*256, 2048)
    W_fc1 = tf.get_variable("W_fc1", shape=[7 * 7 * 256, 2048],
                            initializer=tf.contrib.layers.xavier_initializer(seed=0))
    b_fc1 = bias_variable([2048])

    h_pool5_flat = tf.reshape(h_pool5, [-1, 7 * 7 * 256])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool5_flat, W_fc1) + b_fc1)

    h_fc1_drop = tf.nn.dropout(h_fc1, 1.0)

    # W_fc2 = weight_variable([2048, 256], 2048, 256)
    W_fc2 = tf.get_variable("W_fc2", shape=[2048, 256], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    b_fc2 = bias_variable([256])

    h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    # W_fc3 = weight_variable([256, 64], 256, 64)
    W_fc3 = tf.get_variable("W_fc3", shape=[256, 64], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    b_fc3 = bias_variable([64])

    h_fc3 = tf.nn.relu(tf.matmul(h_fc2, W_fc3) + b_fc3)

    # W_fc4 = weight_variable([64, 6], 64, 6)
    W_fc4 = tf.get_variable("W_fc4", shape=[64, 3], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    b_fc4 = bias_variable([3])

    y_conv = tf.nn.softmax(tf.matmul(h_fc3, W_fc4) + b_fc4)
    #    y_conv = tf.matmul(h_fc3, W_fc4) + b_fc4

    return y_conv

def flower_eval():
    image_batch_out, label_batch_out = flower_input(if_random=False, if_training=False)

    image_batch_placeholder = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3])
    image_batch = tf.reshape(image_batch_out, (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3))

    label_tensor_placeholder = tf.placeholder(tf.int64, shape=[BATCH_SIZE])
    label_batch = label_batch_out

    logits_out = tf.reshape(flower_inference(image_batch_placeholder), [BATCH_SIZE, 3])
    logits_batch = tf.to_int64(tf.arg_max(logits_out, dimension=1))

    correct_prediction = tf.equal(logits_batch, label_tensor_placeholder)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        try:
            saver.restore(sess, DATA_DIR + "tmp\\checkpoint.ckpt")
            print("\n--------model restored--------\n")
        except:
            print("\n--------model Not restored--------\n")
            pass
        saver.restore(sess, DATA_DIR + "tmp\\checkpoint.ckpt")
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        accuracy_accu = 0

        for i in range(1, 2):
            image_out, label_out = sess.run([image_batch, label_batch])

            accuracy_out, logits_batch_out = sess.run([accuracy, logits_batch],
                                                      feed_dict={image_batch_placeholder: image_out,
                                                                 label_tensor_placeholder: label_out})
            accuracy_accu += accuracy_out

            print(i)
            print(image_out.shape)
            print("label_out: ")
            # print(filename_out)
            print(label_out)
            print(logits_batch_out)

        print("cnn finish")
        print(accuracy_accu / 1)

        coord.request_stop()
        coord.join(threads)
        sess.close()

change()
if __name__ == '__main__':
  tf.app.run()
flower_eval()
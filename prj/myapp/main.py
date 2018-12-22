#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import cv2
import random
import numpy as np
import tensorflow as tf
import tensorflow.python.platform

# 識別ラベルの数(今回は3つ)
NUM_CLASSES = 3
# 学習する時の画像のサイズ(px)
IMAGE_SIZE = 28
# 画像の次元数(28px*28px*3(カラー))
IMAGE_PIXELS = IMAGE_SIZE*IMAGE_SIZE*3

# Flagはデフォルト値やヘルプ画面の説明文を定数っぽく登録できるTensorFlow組み込み関数
flags = tf.app.flags
FLAGS = flags.FLAGS
# 学習用データ
flags.DEFINE_string('train', '/Users/yuhi/tensor/data/train/data.txt', 'File name of train data')
# 検証用データ
flags.DEFINE_string('test', '/Users/yuhi/tensor/data/test/data.txt', 'File name of train data')
# TensorBoardのデータ保存先フォルダ
flags.DEFINE_string('train_dir', '/Users/yuhi/django/alcohol/soft/', 'Directory to put the training data.')
# 学習訓練の試行回数
flags.DEFINE_integer('max_steps', 100, 'Number of steps to run trainer.')
# 1回の学習で何枚の画像
flags.DEFINE_integer('batch_size', 20, 'Batch size Must divide evenly into the dataset sizes.')
# 学習率
flags.DEFINE_float('learning_rate', 1e-4, 'Initial learning rate.')


def inference(images_placeholder, keep_prob):
# 重みを標準偏差0.1の正規分布で初期化する
    def weight_variable(shape):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial)

    # バイアスを標準偏差0.1の正規分布で初期化する
    def bias_variable(shape):
      initial = tf.constant(0.1, shape=shape)
      return tf.Variable(initial)

    # 畳み込み層を作成する
    def conv2d(x, W):
      return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    # プーリング層を作成する
    def max_pool_2x2(x):
      return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')

    # ベクトル形式で入力されてきた画像データを28px*28pxの画像に戻す(多分)。
    # 今回はカラー画像なので3(モノクロだと1)
    x_image = tf.reshape(images_placeholder, [-1, IMAGE_SIZE, IMAGE_SIZE, 3])

    # 畳み込み層第1レイヤーを作成
    with tf.name_scope('conv1') as scope:

      W_conv1 = weight_variable([5, 5, 3, 32])
      # バイアスの数値を代入
      b_conv1 = bias_variable([32])

      h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)


    with tf.name_scope('pool1') as scope:
      h_pool1 = max_pool_2x2(h_conv1)

    # 畳み込み層第2レイヤーの作成
    with tf.name_scope('conv2') as scope:
      W_conv2 = weight_variable([5, 5, 32, 64])
      # バイアスの数値を代入(第一レイヤーと同じ)
      b_conv2 = bias_variable([64])
      # 検出した特徴の整理(第一レイヤーと同じ)
      h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    # プーリング層2の作成(ブーリング層1と同じ)
    with tf.name_scope('pool2') as scope:
      h_pool2 = max_pool_2x2(h_conv2)

    # 全結合層1の作成
    with tf.name_scope('fc1') as scope:
      W_fc1 = weight_variable([7*7*64, 1024])
      b_fc1 = bias_variable([1024])
      # 画像の解析を結果をベクトルへ変換
      h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
      # 第一、第二と同じく、検出した特徴を活性化させている
      h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

      h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # 全結合層2の作成(読み出しレイヤー)
    with tf.name_scope('fc2') as scope:
      W_fc2 = weight_variable([1024, NUM_CLASSES])
      b_fc2 = bias_variable([NUM_CLASSES])

    # ソフトマックス関数による正規化
    # ここまでのニューラルネットワークの出力を各ラベルの確率へ変換する
    with tf.name_scope('softmax') as scope:
      y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    # 各ラベルの確率(のようなもの?)を返す。算出された各ラベルの確率を全て足すと1になる
    return y_conv

def loss(logits, labels):
  # 交差エントロピーの計算
  cross_entropy = -tf.reduce_sum(labels*tf.log(logits))
  # TensorBoardで表示するよう指定
  tf.summary.scalar("cross_entropy", cross_entropy)
  # 誤差の率の値(cross_entropy)を返す
  return cross_entropy

def training(loss, learning_rate):
  #この関数がその当たりの全てをやってくれる様
  train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
  return train_step

# inferenceで学習モデルが出した予測結果の正解率を算出する
def accuracy(logits, labels):
  # 予測ラベルと正解ラベルが等しいか比べる。同じ値であればTrueが返される
  # argmaxは配列の中で一番値の大きい箇所のindex(=一番正解だと思われるラベルの番号)を返す
  correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
  # booleanのcorrect_predictionをfloatに直して正解率の算出
  # false:0,true:1に変換して計算する
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
  # TensorBoardで表示する様設定
  tf.summary.scalar("accuracy", accuracy)
  return accuracy

if __name__ == '__main__':
  # 学習用画像をTensorFlowで読み込めるようTensor形式(行列)に変換
  # ファイルを開く
  f = open(FLAGS.train, 'r')
  # データを入れる配列
  train_image = []
  train_label = []
  for line in f:
    line = line.rstrip()
    l = line.split()

    #print(l[0])
    img = cv2.imread(l[0])
    #print(img.shape)

    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    train_image.append(img.flatten().astype(np.float32)/255.0)

    tmp = np.zeros(NUM_CLASSES)
    tmp[int(l[1])] = 1
    train_label.append(tmp)

  train_image = np.asarray(train_image)
  train_label = np.asarray(train_label)
  f.close()

  # 検証用画像をTensorFlowで読み込めるようTensor形式(行列)に変換
  f = open(FLAGS.test, 'r')
  test_image = []
  test_label = []
  for line in f:
    line = line.rstrip()
    l = line.split()
    img = cv2.imread(l[0])
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    test_image.append(img.flatten().astype(np.float32)/255.0)
    tmp = np.zeros(NUM_CLASSES)
    tmp[int(l[1])] = 1
    test_label.append(tmp)
  test_image = np.asarray(test_image)
  test_label = np.asarray(test_label)
  f.close()

  #TensorBoardのグラフに出力するスコープを指定
  with tf.Graph().as_default():
    images_placeholder = tf.placeholder("float", shape=(None, IMAGE_PIXELS))
    labels_placeholder = tf.placeholder("float", shape=(None, NUM_CLASSES))
    keep_prob = tf.placeholder("float")

    logits = inference(images_placeholder, keep_prob)
    loss_value = loss(logits, labels_placeholder)
    train_op = training(loss_value, FLAGS.learning_rate)
    acc = accuracy(logits, labels_placeholder)

    # 保存の準備
    saver = tf.train.Saver()
    # Sessionの作成
    sess = tf.Session()
    # 変数の初期化
    sess.run(tf.initialize_all_variables())
    # TensorBoard表示
    summary_op = tf.summary.merge_all()
    # train_dirでTensorBoardログを出力するpathを指定
    summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph_def)

    # 実際にmax_stepの回数だけ訓練の実行していく
    for step in range(FLAGS.max_steps):
      for i in range(int(len(train_image)/FLAGS.batch_size)):

        batch = FLAGS.batch_size*i
        sess.run(train_op, feed_dict={
          images_placeholder: train_image[batch:batch+FLAGS.batch_size],
          labels_placeholder: train_label[batch:batch+FLAGS.batch_size],
          keep_prob: 0.5})

      train_accuracy = sess.run(acc, feed_dict={
        images_placeholder: train_image,
        labels_placeholder: train_label,
        keep_prob: 1.0})
      print ("step %d, training accuracy %g"%(step, train_accuracy))

      summary_str = sess.run(summary_op, feed_dict={
        images_placeholder: train_image,
        labels_placeholder: train_label,
        keep_prob: 1.0})
      summary_writer.add_summary(summary_str, step)

  print("test accuracy %g"%sess.run(acc, feed_dict={
    images_placeholder: test_image,
    labels_placeholder: test_label,
    keep_prob: 1.0}))

  save_path = saver.save(sess, "./model.ckpt")
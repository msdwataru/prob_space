import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import random
from sklearn.model_selection import cross_val_score
from IPython import embed

from cnn import CNN, conv2d, max_pool_2x2

if __name__ == "__main__":
   #img_file = "./kmnist-train-imgs.npz"
   #label_file = "./kmnist-train-labels.npz"
   test_img_file = "./kmnist-test-imgs.npz"

   imgs = np.load(test_img_file)["arr_0"]
   #labels = np.load(label_file)["arr_0"]
   #labels_onehot = np.zeros([len(labels), 10])
   #for i, l in enumerate(labels):
   #   labels_onehot[i, l] = 1
      
   #num_valid = int(0.2 * len(imgs))
   
   #train_data = imgs[:-num_valid]
   #train_data = np.expand_dims(train_data,axis=3)
   train_data = imgs / 255.
   #valid_data = imgs[-num_valid:]
   #valid_data = np.expand_dims(valid_data,axis=3)
   #valid_data = valid_data / 255.
   #train_label = labels_onehot[:-num_valid]
   #valid_label = labels_onehot[-num_valid:]

   in_ph = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
   target_ph = tf.placeholder(tf.float32, shape=[None, 10])
   keep_prob_ph = tf.placeholder(tf.float32)
   
   channel_list = [1, 16, 16, 32, 32]
   model = CNN(3, 3, ch_list=channel_list)
   
   output = model(in_ph, keep_prob_ph)
   #loss = tf.reduce_sum(tf.square(output - target_ph))
   #loss = tf.reduce_sum(-target_ph*tf.log(output + 1e-7) - (1 - target_ph) * tf.log(1. - output + 1e-7))
   loss = tf.reduce_mean(-output * tf.log(target_ph + 1e-7))
   train_op = tf.train.AdamOptimizer(learning_rate=0.002).minimize(loss)
   
   saver = tf.train.Saver(tf.global_variables())
   
   epoch = 100
   batch_size = 100
   total_batch = int(len(train_data) / batch_size)

   sess = tf.InteractiveSession()
   sess.run(tf.global_variables_initializer())

   #feed_dict = {in_ph: train_data,
   #             target_ph: train_label}
   train_idx = list(range(len(train_data)))
   for epc in range(1, epoch + 1):
      random.shuffle(train_idx)
      for i in range(total_batch):
         mini_batch = train_data[train_idx[i*batch_size:(i+1)*batch_size]]
         #mini_batch = train_data[:100]
         mini_batch_y = train_label[train_idx[i*batch_size:(i+1)*batch_size]]
         #mini_batch_y = train_label[:100]
         feed_dict = {in_ph: mini_batch,
                      target_ph: mini_batch_y,
                      keep_prob_ph: 0.7}
         
         result = sess.run([loss, train_op], feed_dict=feed_dict)
         
      train_res = sess.run(output, feed_dict=feed_dict)
      pred_label = [np.argmax(train_res[i]) for i in range(len(train_res))]

      #accuracy_train = sum(pred_label == labels[:100]) / len(train_res)
      
      valid_res = sess.run(output, feed_dict={in_ph: valid_data, keep_prob_ph: 1.0})
      pred_label = [np.argmax(valid_res[i]) for i in range(len(valid_res))]
      accuracy_valid = sum(pred_label == labels[-num_valid:]) / len(valid_label)
      #print("epoch: {}, loss: {}, accuracy_train: {}, accuracy_valid: {}".format(epc, result[0], accuracy_train, accuracy_valid))
      print("epoch: {}, loss: {}, accuracy_valid: {}".format(epc, result[0], accuracy_valid))
      
      
      
   valid_res = sess.run(output, feed_dict={in_ph: valid_data,keep_prob_ph: 1.0})
   saver.save(sess, "./result/model", global_step=epoch)




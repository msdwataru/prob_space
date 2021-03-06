import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import random
from sklearn.model_selection import cross_val_score
from IPython import embed

from cnn import CNN, conv2d, max_pool_2x2

if __name__ == "__main__":

   test_img_file = "./kmnist-test-imgs.npz"

   imgs = np.load(test_img_file)["arr_0"]

   test_data = imgs / 255.
   test_data = np.expand_dims(test_data,axis=3)

   in_ph = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
   target_ph = tf.placeholder(tf.float32, shape=[None, 10])
   keep_prob_ph = tf.placeholder(tf.float32)
   
   channel_list = [1, 16, 16, 32, 32]
   model = CNN(3, 3, ch_list=channel_list)
   
   output = model(in_ph, keep_prob_ph)
   loss = tf.reduce_mean(-output * tf.log(target_ph + 1e-7))
   train_op = tf.train.AdamOptimizer(learning_rate=0.002).minimize(loss)
   
   saver = tf.train.Saver(tf.global_variables())
   
   sess = tf.InteractiveSession()
   sess.run(tf.global_variables_initializer())
   checkpoint = tf.train.get_checkpoint_state("./result/")
   if checkpoint and checkpoint.model_checkpoint_path:
      print(checkpoint.model_checkpoint_path)
      saver.restore(sess, checkpoint.model_checkpoint_path)
   feed_dict = {in_ph: test_data,
                #target_ph: mini_batch_y,
                keep_prob_ph: 1.0}
         
   test_res = sess.run(output, feed_dict=feed_dict)
   pred_label = [np.argmax(test_res[i]) for i in range(len(test_res))]

   embed()
   sess.close()

import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import random
from sklearn.model_selection import cross_val_score
from IPython import embed

from learn_resnet import CNN, conv2d, max_pool_2x2, batch_norm_wrapper

if __name__ == "__main__":

   test_img_file = "./kmnist-test-imgs.npz"

   imgs = np.load(test_img_file)["arr_0"]

   test_data = imgs / 255.
   test_data = np.expand_dims(test_data,axis=3)

   in_ph = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
   target_ph = tf.placeholder(tf.float32, shape=[None, 10])
   keep_prob_ph = tf.placeholder(tf.float32)
   
   #channel_list = [1, 8, 16, 16, 16, 16]
   channel_list = [1, 16, 32, 32, 64, 64, 128, 128]
   model = CNN(3, 3, ch_list=channel_list)
   
   output = model(in_ph, keep_prob_ph)
   #loss = tf.reduce_mean(-output * tf.log(target_ph + 1e-7))
   #train_op = tf.train.AdamOptimizer(learning_rate=0.002).minimize(loss)
   
   saver = tf.train.Saver(tf.global_variables())

   cv = 5
   all_pred = []
   all_pred_label = []
   for i in range(5):
      cv += 1
      sess = tf.InteractiveSession()
      sess.run(tf.global_variables_initializer())
      checkpoint = tf.train.get_checkpoint_state("./result/cv{}/".format(cv))
      if checkpoint and checkpoint.model_checkpoint_path:
         print(checkpoint.model_checkpoint_path)
      saver.restore(sess, checkpoint.model_checkpoint_path)
      feed_dict = {in_ph: test_data,
                   #target_ph: mini_batch_y,
                   keep_prob_ph: 1.0}
      
      pred = sess.run(output, feed_dict=feed_dict)
      pred_label = [np.argmax(pred[i]) for i in range(len(pred))]
      all_pred.append(pred)
      all_pred_label.append(pred_label)
      sess.close()
   all_pred = np.array(all_pred).transpose(1, 0, 2)
   all_pred_label = np.array(all_pred_label).transpose(1, 0)

   mean_pred = np.empty([len(pred), 10])
   for i in range(len(pred)):
      mean_pred[i, :] = np.mean(all_pred[i], axis=0)
   ans = [np.argmax(mean_pred[i]) for i in range(len(pred))]
   np.savetxt("ans.txt", ans)


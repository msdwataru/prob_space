import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import random
from sklearn.model_selection import cross_val_score
from IPython import embed

class CNN:
   def __init__(self, k_h, k_w, ch_list=[3,32,32,16,8], stddev=0.1):
        self.ch_list = ch_list
        #define learnable parameter
        with tf.variable_scope("cnn"):
            #conv
            self.w_conv1 = tf.Variable(tf.truncated_normal([k_h, k_w, self.ch_list[0], self.ch_list[1]], stddev=stddev))
            
            self.w_conv2 = tf.Variable(tf.truncated_normal([k_h, k_w, self.ch_list[1], self.ch_list[2]], stddev=stddev))
            
            self.w_conv3 = tf.Variable(tf.truncated_normal([k_h, k_w, self.ch_list[2], self.ch_list[3]], stddev=stddev))

            self.w_conv4 = tf.Variable(tf.truncated_normal([k_h, k_w, self.ch_list[3], self.ch_list[4]], stddev=stddev))
            
            self.w_fc1 = tf.Variable(tf.truncated_normal([7 * 7 * self.ch_list[4], 10], stddev=stddev))
            self.b_fc1 = tf.Variable(tf.zeros([10]))
            """
            #deconv
            self.w_fc2 = tf.Variable(tf.truncated_normal([5, 128]))
            self.b_fc2 = tf.Variable(tf.zeros([128]))

            self.w_deconv1 = tf.Variable(tf.truncated_normal([k_h, k_w, self.ch_list[2], self.ch_list[3]], stddev=stddev))

            self.w_deconv2 = tf.Variable(tf.truncated_normal([k_h, k_w, self.ch_list[1], self.ch_list[2]], stddev=stddev))
            
            self.w_deconv3 = tf.Variable(tf.truncated_normal([k_h, k_w,self.ch_list[0], self.ch_list[1]], stddev=stddev))
            """
            
   def __call__(self, x, keep_prob, train=True):
        #Conv1(28*28*16)
        h_conv1 = conv2d(x, self.w_conv1, train=train)
        
        #Conv2(28*28*16)
        h_conv2 = conv2d(h_conv1, self.w_conv2, train=train)

        # max pooling(14*14*16)
        h_pool2 = max_pool_2x2(h_conv2) 

        #Conv3(14*14*32)
        h_conv3 = conv2d(h_pool2, self.w_conv3, train=train)

        #Conv4(14*14*32)
        h_conv4 = conv2d(h_conv3, self.w_conv4, train=train)

        # max pooling(7*7*32)
        h_pool4 = max_pool_2x2(h_conv4) 

        #Full connection1(10)
        h_pool4 = tf.reshape(h_pool4, [-1, 7 * 7 * self.ch_list[4]])
        h_fc1 = tf.matmul(h_pool4, self.w_fc1) + self.b_fc1
        h_drop = tf.nn.dropout(h_fc1, keep_prob)
        #h_fc1 = tf.nn.tanh(h_fc1)
        #h_fc1 = tf.nn.sigmoid(h_fc1)
        out_softmax = tf.nn.softmax(h_drop)
        """
        #Full connection2(128)
        h_fc2 = tf.matmul(h_fc1, self.w_fc2) + self.b_fc2
        h_fc2 = tf.nn.tanh(h_fc2)
        h_fc2 = tf.reshape(h_fc2, [-1, 4, 4, self.ch_list[3]])

        #Deconv1(7*7*16)
        h_deconv1 = deconv2d(h_fc2, self.w_deconv1, [batch_size, 7, 7, self.ch_list[2]], train=train)
        
        #Deconv2(14*14*16)
        h_deconv2 = deconv2d(h_deconv1, self.w_deconv2, [batch_size, 14, 14, self.ch_list[1]], train=train)
        
        #Deconv3(28*28*1)
        h_deconv3 = deconv2d(h_deconv2, self.w_deconv3, [batch_size, 28, 28, self.ch_list[0]], train=train)
        """
        #return h_deconv3, h_fc1
        return out_softmax

def conv2d(x, weight, batch_norm=None, train=True, activation=tf.nn.relu, stride=1):
    h_conv = tf.nn.conv2d(x, weight, strides=[1,stride,stride,1], padding="SAME")
    if batch_norm != None:
        h_conv = batch_norm(h_conv, train=train)
    h_conv = activation(h_conv)
    return h_conv

def max_pool_2x2(x):
   return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                         strides=[1, 2, 2, 1], padding='SAME')
 
def deconv2d(x, weight, output_shape, batch_norm=None, train=True, activation=tf.nn.tanh):
    h_deconv = tf.nn.conv2d_transpose(x, weight, output_shape=output_shape, strides=[1, 2, 2, 1], padding="SAME")
    if batch_norm != None:
        h_deconv = batch_norm(h_deconv, train=train)
    h_deconv = activation(h_deconv)
    return h_deconv
        



if __name__ == "__main__":
   img_file = "./kmnist-train-imgs.npz"
   label_file = "./kmnist-train-labels.npz"
   test_img_file = "./kmnist-test-imgs.npz"

   imgs = np.load(img_file)["arr_0"]
   labels = np.load(label_file)["arr_0"]
   labels_onehot = np.zeros([len(labels), 10])
   for i, l in enumerate(labels):
      labels_onehot[i, l] = 1
      
   num_valid = int(0.2 * len(imgs))
   
   train_data = imgs[:-num_valid]
   train_data = np.expand_dims(train_data,axis=3)
   train_data = train_data / 255.
   valid_data = imgs[-num_valid:]
   valid_data = np.expand_dims(valid_data,axis=3)
   valid_data = valid_data / 255.
   train_label = labels_onehot[:-num_valid]
   valid_label = labels_onehot[-num_valid:]

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




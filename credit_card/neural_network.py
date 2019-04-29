import numpy as np
import pandas as pd
import tensorflow as tf
from IPython import embed

class NN:
    def __init__(self, in_dim, num_hidden=30):
        self.w_i2h = tf.Variable(tf.truncated_normal([in_dim, num_hidden], stddev=0.01))
        self.b_h = tf.Variable(tf.zeros([num_hidden]))
        self.w_h2o = tf.Variable(tf.truncated_normal([num_hidden, 1], stddev=0.01))
        self.b_o = tf.Variable(tf.zeros([1]))                               
                               
    def __call__(self, x):
        inter_states= tf.matmul(x, self.w_i2h) + self.b_h
        hidden_states = tf.nn.tanh(inter_states)
        out_states = tf.matmul(hidden_states, self.w_h2o) + self.b_o
        out = tf.nn.sigmoid(out_states)
        return out
                               

file_name = "./train_data.csv"
data = np.array(pd.read_csv(file_name))
num_train = int(0.8 * len(data))
train_data = data[:num_train]
validation_data = data[num_train:]
x = train_data[:, 1:-1].astype(np.float32)
y = train_data[:, -1].astype(np.float32)
y = np.expand_dims(y, axis=1)
x = x[:,5:10]
in_ph = tf.placeholder(tf.float32, shape=[None, x.shape[1]])
target_ph = tf.placeholder(tf.float32, shape=[None, 1])
model = NN(x.shape[1])
out_put = model(in_ph)
loss = tf.reduce_mean(tf.square(out_put - target_ph))
train_op = tf.train.AdamOptimizer().minimize(loss)
saver = tf.train.Saver(tf.global_variables())

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

epoch = 1000
for epc in range(1, epoch):
    feed_dict = {in_ph: x,
                 target_ph: y}
    result = sess.run([loss, train_op], feed_dict=feed_dict)
    print(result[0])
    if epc % 100 == 0:
        saver.save(sess, ".result/model", global_step = epc)
    
x_test = validation_data[:, 1:-1].astype(np.float32)
y_test = validation_data[:, -1].astype(np.float32)
y_test = np.expand_dims(y_test, axis=1)
x_test = x_test[:,5:10]
feed_dict = {in_ph: x_test,
             target_ph: y_test}
    
prediction = sess.run(out_put, feed_dict=feed_dict)
res = []
for i in range(len(x_test)):
    if prediction[i] > 0.5:
        res.append(1)
    else:
        res.append(0)
        
embed()


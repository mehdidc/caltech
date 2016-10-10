import numpy as np
import tensorflow as tf

# source : https://richardstechnotes.wordpress.com/2016/08/09/encouraging-tensorflow-to-use-more-cores/
config = tf.ConfigProto(
   device_count={"CPU":12},
   inter_op_parallelism_threads=0,
   intra_op_parallelism_threads=0)
session = tf.Session(config=config)

# linreg
inputs = tf.placeholder('float', name='X')
outputs = tf.placeholder('float', name='y')
w = (tf.matmul(tf.transpose(inputs), inputs)) # x.T * x
w = tf.matrix_inverse(w, name='W') # (x.T * x)^{-1}
w = tf.matmul(w, tf.transpose(inputs)) # ((x.T * x)^{-1}) X^T
w = tf.matmul(w, tf.reshape(outputs, (-1, 1))) #((x.T * x)^{-1}) X^Ty

def linreg_np(X, y):
    return np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)

def linreg_tf(X, y):
    W, = session.run([w], {inputs:X, outputs: y})
    return W[:,0]

def linreg_sgd_tf(X, y, lr=0.1, nb_iter=1000, verbose=0, batch_size=1, tol=1e-5):
    inputs = tf.placeholder('float', name='X')
    outputs = tf.placeholder('float', name='y')
    w = tf.Variable(tf.random_normal([X.shape[1], 1]), name='W')
    loss = tf.reduce_mean(tf.square(tf.matmul(inputs, w) - outputs))
    train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss)
    init = tf.initialize_all_variables()
    sess.run(init)
    prev_loss_val = np.inf
    for cur_iter in range(nb_iter):
        for i in range(0, len(X), batch_size):
            x_cur = X[i:i + batch_size]
            y_cur = y[i:i + batch_size]
            sess.run(train_step, feed_dict={inputs: x_cur, outputs: y_cur})
        loss_val = (sess.run(loss, feed_dict={inputs: X, outputs:y}))
        if np.abs(loss_val - prev_loss_val) < tol:
            print('quit early at iter {}'.format(cur_iter))
            break
        prev_loss_val = loss_val
    W = (session.run(w))
    return W

linreg = linreg_tf
     
def run_perceptron(x, y, w=None, max_iter=100):
    if w is None:
        w = np.zeros((3,))
    nb_updates = 0
    for cur_iter in range(max_iter):
        y_pred = (np.dot(x, w) > 0) * 2 - 1
        ind = np.arange(len(y))[y_pred != y]
        if len(ind) == 0:
            break
        i = np.random.choice(ind)
        w += x[i] * y[i]
        nb_updates += 1
    return w, nb_updates

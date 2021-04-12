import tensorflow as tf

# Functions used for the layers in CNN

def conv2dx(inp, filt, stride_s) :
  conx = tf.nn.conv2d(inp, filters = filt, strides = [1, stride_s, stride_s, 1], padding = "SAME")
  return tf.nn.relu(conx)

def maxpool(inp, k, stride_s) :
  poolx = tf.nn.max_pool2d(inp,ksize=[1, k, k, 1], strides=[1, stride_s, stride_s, 1],padding = "VALID")
  return poolx

def dense(inp, weights, dropout_rate=0.5) :
  densex = tf.nn.relu(tf.matmul(inp, weights))
  return tf.nn.dropout(densex, dropout_rate)





import tensorflow as tf
import layers as ll

########## Initializing random values for the weights of filters and pool and dense layers ########

initializer = tf.initializers.glorot_uniform()

def rng_weight( shape_weight , name ):
  return tf.Variable( initializer( shape_weight ) , name=name , trainable=True , dtype=tf.float32 )

###################################################################################################

class cdmodel(tf.Module) :
  def __init__(self) :
    super(cdmodel, self).__init__()
    self.model_weights = []
    self.output_class = 2
    self.dropout_rate = 0.5
    #initialize model weights
    #Referring to the documentation of tf.nn.conv2d, the size of the filter argument
    #conforms to [filter_height, filter_width, in_channels, out_channels]
    # 1. Input channels for the first convolution layer = 3 (R, G, B values)
    # 2. For first use, 3x3 filter would suffice
    # 3. Used working model for output channels which is 16
    self.shape_weight = [
      [3, 3, 3, 16],            #1st convo 
      [3, 3, 16, 16],
      [3, 3, 16, 32],           #2nd convo layer
      [3, 3, 32, 32],
      [3, 3, 32, 64],           #3rd convo
      [3, 3, 64, 64],
      [3, 3, 64, 64],
      [3, 3, 64, 128],          #4th convo
      [3, 3, 128, 128],
      [3, 3, 128, 128],
      [3, 3, 128, 256],         #5th convo
      [3, 3, 256, 256],
      [3, 3, 256, 256],
      [3, 3, 256, 512],         #6th convo
      [3, 3, 512, 512],
      [3, 3, 512, 512],
      [8192, 3600],
      [3600, 2400],
      [2400, 1600],
      [1600, 800],
      [800, 64],
      [64, self.output_class]
    ]

    #randomize values
    for i in range( len( self.shape_weight ) ):
      self.model_weights.append( rng_weight( self.shape_weight[ i ] , 'weight{}'.format( i ) ) )
  
  def __call__(self, x) :
    x = tf.cast(x, dtype=tf.float32)
    self.conv1 = ll.conv2dx(x, self.model_weights[0], 1)           # 1st Layer
    self.conv1 = ll.conv2dx(self.conv1, self.model_weights[1], 1)
    self.pool1 = ll.maxpool(self.conv1, 2, 2)

    self.conv2 = ll.conv2dx(self.pool1, self.model_weights[2], 1)  # 2nd Layer
    self.conv2 = ll.conv2dx(self.conv2, self.model_weights[3], 1)
    self.pool2 = ll.maxpool(self.conv2, 2, 2)

    self.conv3 = ll.conv2dx(self.pool2, self.model_weights[4], 1)  # 3rd Layer
    self.conv3 = ll.conv2dx(self.conv3, self.model_weights[5], 1)
    self.conv3 = ll.conv2dx(self.conv3, self.model_weights[6], 1)
    self.pool3 = ll.maxpool(self.conv3, 2, 2)

    self.conv4 = ll.conv2dx(self.pool3, self.model_weights[7], 1)  # 4th Layer
    self.conv4 = ll.conv2dx(self.conv4, self.model_weights[8], 1)
    self.conv4 = ll.conv2dx(self.conv4, self.model_weights[9], 1)
    self.pool4 = ll.maxpool(self.conv4, 2, 2)

    self.conv5 = ll.conv2dx(self.pool4, self.model_weights[10], 1)                       # 5th Layer
    self.conv5 = ll.conv2dx(self.conv5, self.model_weights[11], 1)
    self.conv5 = ll.conv2dx(self.conv5, self.model_weights[12], 1)
    self.pool5 = ll.maxpool(self.conv5, 2, 2)

    self.conv6 = ll.conv2dx(self.pool5, self.model_weights[13], 1)                       # 6th Layer
    self.conv6 = ll.conv2dx(self.conv6, self.model_weights[14], 1)
    self.conv6 = ll.conv2dx(self.conv6, self.model_weights[15], 1)
    self.pool6 = ll.maxpool(self.conv6, 2, 2)

    self.flatten_layer = tf.reshape(self.pool6, shape=(tf.shape(self.pool6)[0], -1))  # flatten

    self.dense1 = ll.dense(self.flatten_layer, self.model_weights[16], self.dropout_rate)
    self.dense2 = ll.dense(self.dense1, self.model_weights[17], self.dropout_rate)
    self.dense3 = ll.dense(self.dense2, self.model_weights[18], self.dropout_rate)
    self.dense4 = ll.dense(self.dense3, self.model_weights[19], self.dropout_rate)
    self.dense5 = ll.dense(self.dense4, self.model_weights[20], self.dropout_rate)
    self.dense6 = tf.matmul(self.dense5, self.model_weights[21])

    return tf.nn.softmax(self.dense6)
  
  def update_weights(self,weights) :
    self.model_weights = weights
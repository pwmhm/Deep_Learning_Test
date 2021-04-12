import tensorflow as tf
import main as m
import pickle
from datetime import date


#initialize some parameters
learning_rate = 0.0008   #standard learning rate2
epoch         = 20      #massive data
batch_size    = 2
dropout_rate  = 0.7
dataset_name = 'cats_vs_dogs' 

#import cats vs dogs dataset
train_dataset = tf.data.experimental.load("processed_dataset/train/", (tf.TensorSpec(shape=(300, 300, 3), dtype=tf.float32, name=None),
 tf.TensorSpec(shape=(), dtype=tf.int64, name=None)), compression="GZIP")

train_dataset = train_dataset.batch(batch_size)




def loss(pred, target) :
    return tf.losses.categorical_crossentropy( target , pred )

optimizer = tf.optimizers.Adam( learning_rate ) 

training_model = m.cdmodel()
str_date = "logs/training-" + str(date.today()) +".txt"
text_file = open(str_date, "wt")
text_file.write("Parameters : \nLearning_rate = {0} ,Epoch : {1} ,Batch Size : {2},Data Length : {3}\n\n".format(learning_rate, epoch, batch_size, len(train_dataset)))
text_file.write("Current Epoch,Current Loss\n")

for i in range(epoch) :
    j=0
    for inputs in train_dataset:
        if j%4 == 0 :
            with open('saved_weights.pickle', 'wb') as handle:
                pickle.dump(training_model.model_weights, handle)
        #extract the inputs
        image, label = inputs[0], inputs[1]

        #resize value to 0..100
        image_input_mapped = image/255.0
        
        #the gradient descent
        with tf.GradientTape() as Tape :
            closs = loss(training_model(image_input_mapped), tf.one_hot(label, depth=2))
        g = Tape.gradient(closs, training_model.model_weights)
        optimizer.apply_gradients(zip(g, training_model.model_weights))
        j += 1
        text_file.write("{0},{1}\n".format(i, tf.reduce_mean(closs)))
        tf.print(i, tf.reduce_mean(closs))

with open('saved_weights.pickle', 'wb') as handle:
    pickle.dump(training_model.model_weights, handle)
    print("SAVED!")
text_file.close()
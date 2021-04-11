import tensorflow as tf
import tensorflow_datasets as tfds
import main as m
import dataset as ds


#import cats vs dogs dataset

dataset_name = 'cats_vs_dogs' 

ds_raw = tfds.load(
    name=dataset_name,
    split = 'train',
    with_info =False,
    shuffle_files=False)

ds_raw = ds_raw.take(5000)

batch_size = 10
divider = len(ds_raw)/2

train_dataset, test_dataset = ds.data_preprocessing(ds_raw, batch_size, divider)

#initialize some training functions
learning_rate = 1e-03   #standard learning rate2
epoch         = 20      #massive data


def loss(pred, target) :
    return tf.losses.categorical_crossentropy( target , pred )

optimizer = tf.optimizers.Adam( learning_rate ) 

training_model = m.cdmodel()

for i in range(epoch) :
    print("Epoch : %d" % epoch)
    print("Current Epoch : %d" % i)
    for inputs in train_dataset:
        #extract the inputs
        image, label = inputs[0], inputs[1]

        #resize value to 0..100
        image_input_mapped = image/255.0
        
        #the gradient descent
        with tf.GradientTape() as Tape :
            closs = loss(training_model(image_input_mapped), tf.one_hot(label, depth=2))
        g = Tape.gradient(closs, training_model.model_weights)
        optimizer.apply_gradients(zip(g, training_model.model_weights))
        tf.print(tf.reduce_mean(closs))
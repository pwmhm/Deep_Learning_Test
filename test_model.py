import tensorflow as tf
import main as m
import pickle
import matplotlib.pyplot as plt

output_label = ["Cat", "Dog"]

#import cats vs dogs dataset
test_dataset = tf.data.experimental.load("processed_dataset/test/", (tf.TensorSpec(shape=(300, 300, 3), dtype=tf.float32, name=None),
 tf.TensorSpec(shape=(), dtype=tf.int64, name=None)), compression="GZIP")

test_dataset = test_dataset.batch(1)

#initialize a random model
dropout_rate  = 0.7
test_model = m.cdmodel()
test_model.dropout_rate = dropout_rate

#retrieve weights saved from training
with open('saved_weights.pickle', 'rb') as handle :
    new_w = pickle.load(handle)

#update the weights of our model
test_model.update_weights(new_w)

#test the model
for ele in test_dataset :
    res = test_model(ele[0])
    #find index of maximum values
    p_init = tf.where(tf.equal(res[0], max(res[0])))

    #Compare with target. If target == output, output = 1. Thus using Prediction XNOR Target
    prediction = tf.cast(p_init[0], "int64")
    target = tf.cast(ele[1], "int64")
    output = ~(prediction ^ target)+2

    print("Data : {0} , Pred : {1}\n".format(output_label[ele[1][0].numpy()], output_label[prediction[0].numpy()]))


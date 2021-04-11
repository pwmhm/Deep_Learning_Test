import tensorflow as tf
import tensorflow_datasets as tfds
import main as m
import dataset as ds
import pickle
import matplotlib.pyplot as plt

#import cats vs dogs dataset

dataset_name = 'cats_vs_dogs'
output_label = ['cat', 'dog']
ds_raw = tfds.load(
    name=dataset_name,
    split = 'train',
    with_info =False,
    shuffle_files=False)

ds_raw = ds_raw.take(10)

divider = len(ds_raw)/2

train_dataset, test_dataset = ds.data_preprocessing(dataset=ds_raw, train_test_separator=divider)

#initialize a random model
test_model = m.cdmodel()

#retrieve weights saved from training
with open('saved_weights.pickle', 'rb') as handle :
    new_w = pickle.load(handle)

#update the weights of our model
test_model.update_weights(new_w)

#to show image
plt.figure(figsize=(10,10))

#test the model
for ele in test_dataset :
    res = test_model(ele[0])
    #find index of maximum values
    p_init = tf.where(tf.equal(res[0], max(res[0])))

    #Compare with target. If target == output, output = 1. Thus using Prediction XNOR Target
    prediction = tf.cast(p_init[0], "int64")
    target = tf.cast(ele[1], "int64")
    output = ~(prediction ^ target)+2

    label_im = "Data : {0} , Pred : {1}".format(output_label[ele[1][0].numpy()], output_label[prediction[0].numpy()])

    plt.subplot(1,1,1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(ele[0][0] / 255)
    plt.xlabel(label_im)
    plt.show()

#show the test image


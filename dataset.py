import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import pickle

#set up new empty list
temp_image = []
temp_label = []
batch_size = 1


dataset_name = 'cats_vs_dogs'
output_label = ['cat', 'dog']
ds_raw = tfds.load(
    name=dataset_name,
    split = 'train',
    with_info =False,
    shuffle_files=False)

ds_raw = ds_raw.take(5000)
train_test_separator = int(len(ds_raw)/2)

#resize all the images to fit our model
for ele in ds_raw :
    image = tf.image.resize(ele['image'], [300, 300], method='bilinear', antialias=False)
    temp_image.append(image)
    temp_label.append(ele['label'])

#turn temps into datasets
d1 = tf.data.Dataset.from_tensors(temp_image[:])
d2 = tf.data.Dataset.from_tensors(temp_label[:])

#merge dataset
merge_ds = tf.data.Dataset.zip((d1,d2))

#Because of temp variables, the tensors are batched into one single batch. so we need to unbatch first
merge_ds = merge_ds.unbatch()

#now we divide the datasets and batch accordingly
out_1 = merge_ds.take(int(train_test_separator))
out_2 = merge_ds.skip(int(train_test_separator))

#shuffle to ensure randomness
out_1 = out_1.shuffle(len(out_1))
out_2 = out_2.shuffle(len(out_2))

# plt.figure(figsize=(10,10))
#
# i = 1
# for ele in out_2 :
#     plt.subplot(5, 5, i+5)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(ele[0] / 255)
#     i+=1
# plt.show()

tf.data.experimental.save(out_1, "processed_dataset/train/", compression="GZIP")
tf.data.experimental.save(out_2, "processed_dataset/test/", compression="GZIP")




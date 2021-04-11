import tensorflow as tf
import tensorflow_datasets as tfds

def data_preprocessing(dataset, train_test_separator, batch_size=1) :

    #set up new empty list
    temp_image = []
    temp_label = []

    #resize all the images to fit our model
    for ele in dataset :
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
    out_1 = out_1.shuffle(len(out_1)).batch(batch_size)
    out_2 = out_2.shuffle(len(out_2)).batch(1)

    return out_1, out_2
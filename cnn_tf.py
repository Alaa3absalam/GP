import tensorflow as tf

tf.enable_eager_execution()
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

import pathlib
data_root = "E:\GP\DataSet"
data_root = pathlib.Path(data_root)
print(data_root)

for item in data_root.iterdir():
  print(item)

import random
all_image_paths = list(data_root.glob('*/*'))
all_image_paths = [str(path) for path in all_image_paths]
random.shuffle(all_image_paths)

image_count = len(all_image_paths)
image_count
print (image_count)

all_image_paths[:10]

attributions = (data_root/"LICENSE.txt").read_text(encoding="utf8").splitlines()[4:]
attributions = [line.split(' CC-BY') for line in attributions]
attributions = dict(attributions)

import IPython.display as display
#
def caption_image(image_path):
    return "Image (CC BY 2.0) "

#
for n in range(3):
  image_path = random.choice(all_image_paths)
  display.display(display.Image(image_path))
  print(caption_image(image_path))
  print()

label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
label_names

label_to_index = dict((name, index) for index,name in enumerate(label_names))
label_to_index

all_image_labels = [label_to_index[pathlib.Path(path).parent.name]
                    for path in all_image_paths]

print("First 10 labels indices: ", all_image_labels[:10])

img_path = all_image_paths[0]
img_path

img_raw = tf.read_file(img_path)
print(repr(img_raw)[:100]+"...")

img_tensor = tf.image.decode_image(img_raw)

print(img_tensor.shape)
print(img_tensor.dtype)

def preprocess_image(image):
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.resize_images(image, [192, 192])
  image /= 255.0  # normalize to [0,1] range

  return image

def load_and_preprocess_image(path):
  image = tf.read_file(path)
  return preprocess_image(image)

import matplotlib.pyplot as plt

image_path = all_image_paths[0]
label = all_image_labels[0]


path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
print('shape: ', repr(path_ds.output_shapes))
print('type: ', path_ds.output_types)
print()
print(path_ds)


AUTOTUNE = tf.data.experimental.AUTOTUNE
image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)




label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))

for label in label_ds.take(10):
  print(label_names[label.numpy()])


image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))

print('image shape: ', image_label_ds.output_shapes[0])
print('label shape: ', image_label_ds.output_shapes[1])
print('types: ', image_label_ds.output_types)
print()
print(image_label_ds)


BATCH_SIZE = 32


ds = image_label_ds.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))
ds = ds.batch(BATCH_SIZE)
ds = ds.prefetch(buffer_size=AUTOTUNE)
ds


mobile_net = tf.keras.applications.MobileNetV2(input_shape=(192, 192, 3), include_top=False)
mobile_net.trainable=False

def change_range(image,label):
  return 2*image-1, label

keras_ds = ds.map(change_range)

# The dataset may take a few seconds to start, as it fills its shuffle buffer.
image_batch, label_batch = next(iter(keras_ds))

feature_map_batch = mobile_net(image_batch)
print(feature_map_batch.shape)

model = tf.keras.Sequential([
  mobile_net,
  tf.keras.layers.GlobalAveragePooling2D(),
  tf.keras.layers.Dense(len(label_names))])

logit_batch = model(image_batch).numpy()

print("min logit:", logit_batch.min())
print("max logit:", logit_batch.max())
print()

print("Shape:", logit_batch.shape)

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=["accuracy"])

len(model.trainable_variables)


model.summary()

steps_per_epoch=tf.ceil(len(all_image_paths)/BATCH_SIZE).numpy()
steps_per_epoch

model.fit(ds, epochs=1, steps_per_epoch=3)


#import time
#
#def timeit(ds, batches=2*steps_per_epoch+1):
#  overall_start = time.time()
#  # Fetch a single batch to prime the pipeline (fill the shuffle buffer),
#  # before starting the timer
#  it = iter(ds.take(batches+1))
#  next(it)
#
#  start = time.time()
#  for i,(images,labels) in enumerate(it):
#    if i%10 == 0:
#      print('.',end='')
#  print()
#  end = time.time()
#
#  duration = end-start
#  print("{} batches: {} s".format(batches, duration))
#  print("{:0.5f} Images/s".format(BATCH_SIZE*batches/duration))
#  print("Total time: {}s".format(end-overall_start))
#
#
#ds = image_label_ds.apply(
#  tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))
#ds = ds.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
#ds


#timeit(ds)

#ds = image_label_ds.cache()
#ds = ds.apply(
#  tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))
#ds = ds.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
#ds
#
#timeit(ds)
#ds = image_label_ds.cache(filename='./cache.tf-data')
#ds = ds.apply(
#  tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))
#ds = ds.batch(BATCH_SIZE).prefetch(1)
#ds
#
#timeit(ds)
#
#
#image_ds = tf.data.Dataset.from_tensor_slices(all_image_paths).map(tf.read_file)
#tfrec = tf.data.experimental.TFRecordWriter('images.tfrec')
#tfrec.write(image_ds)
#
#image_ds = tf.data.TFRecordDataset('images.tfrec').map(preprocess_image)
#
#ds = tf.data.Dataset.zip((image_ds, label_ds))
#ds = ds.apply(
#  tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))
#ds=ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
#ds
#
#
#timeit(ds)
#
#paths_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
#image_ds = paths_ds.map(load_and_preprocess_image)
#image_ds
#
#ds = image_ds.map(tf.serialize_tensor)
#ds

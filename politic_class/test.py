import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras

import utils

IMG_SIZE = 256
BATCH_SIZE = 32
CHANNELS = 3
EPOCHS = 50
TRAIN_SIZE = 0.8
MODEL_VERSION = 5.0
INPUT_SIZE = (BATCH_SIZE, IMG_SIZE, IMG_SIZE, CHANNELS)
N_CLASSES = 2

dataset = keras.preprocessing.image_dataset_from_directory(
    directory='downloads',
    shuffle=False,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)
class_names = dataset.class_names
train_ds, val_ds, test_ds = utils.get_dataset_partitions_tf(dataset)

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)

model = keras.models.load_model(f'../politic_class/model/{MODEL_VERSION}')
scores = model.evaluate(test_ds)
# print(scores)

img_path = 'C:/Michal/Programming//Repositories_MG//repo-for-practice//politic_class//Data//test4.jpg'

pred, conf, p = utils.predict(model, img_path, class_names)

print(f' probablities: {class_names[0]}: {round(100 * p[0][0], 2)}'
      f' {class_names[1]}: {round(100 * p[0][1], 2)}'
      f' {class_names[2]}: {round(100 * p[0][2], 2)}'
      f' {class_names[3]}: {round(100 * p[0][3], 2)}'
      f' {class_names[4]}: {round(100 * p[0][4], 2)}'
      f' {class_names[5]}: {round(100 * p[0][5], 2)}')

print(f'Predicted: Predicted:class: {class_names[np.argmax(p[0])]}')

# for img_batch, lbl_batch in test_ds.take(1):
#     first_img = img_batch[0].numpy().astype('uint8')
#     first_lbl = lbl_batch[0].numpy()
#
#     print(f'Actual class: {class_names[first_lbl]}')
#     print(f'Predicted:class: {class_names[np.argmax(model.predict(img_batch)[0])]}')
#     print(f' probablities: {class_names[0]}: {int(100*model.predict(img_batch)[0][0])}'
#           f' {class_names[1]}: {int(100*model.predict(img_batch)[0][1])}'
#           f' {class_names[2]}: {int(100*model.predict(img_batch)[0][2])}'
#           f' {class_names[3]}: {int(100*model.predict(img_batch)[0][3])}'
#           f' {class_names[4]}: {int(100*model.predict(img_batch)[0][4])}'
#           f' {class_names[5]}: {int(100*model.predict(img_batch)[0][5])}')
#     plt.imshow(first_img)
#     plt.title(model.predict(img_batch)[0])
#     plt.show()

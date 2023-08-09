import keras
import os
import shutil
import tensorflow as tf
import numpy as np


def get_dataset_partitions_tf(ds, train_split=0.7, val_split=0.15, shuffle=True, shuffle_size=10000):
    ds_length = len(ds)

    train_length = int(train_split * ds_length)
    val_length = int(val_split * ds_length)

    if shuffle:
        ds = ds.shuffle(shuffle_size, seed=12)

    train_ds = ds.take(train_length)
    val_ds = ds.skip(train_length).take(val_length)
    test_ds = ds.skip(train_length).skip(val_length)

    return train_ds, val_ds, test_ds


def move_images_up(subfolder_path):
    # Get a list of all files in the subfolder
    files = os.listdir(subfolder_path)

    # Filter out only image files (you can extend the list of valid extensions)
    image_extensions = ['.jpeg']
    image_files = [f for f in files if os.path.splitext(f)[1].lower() in image_extensions]

    if not image_files:
        print("No image files found in the subfolder.")
        return

    parent_folder = os.path.dirname(subfolder_path)

    for image_file in image_files:
        source_path = os.path.join(subfolder_path, image_file)
        destination_path = os.path.join(parent_folder, image_file)

        # Move the image file
        shutil.move(source_path, destination_path)
        print(f"Moved {image_file} to {parent_folder}")


from PIL import Image


def predict(model, img, class_names):
    # img_array = keras.preprocessing.image.img_to_array(img.numpy())
    image = Image.open(img)
    resized_image = image.resize((256, 256))

    img_array = np.array(resized_image)
    img_array = tf.expand_dims(img_array, 0)
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction[0])]
    confidence = round(100 * (np.max(prediction[0])), 2)
    return predicted_class, confidence, prediction

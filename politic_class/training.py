import tensorflow as tf
import keras

import utils
import mlflow.keras

IMG_SIZE = 256
BATCH_SIZE = 2
CHANNELS = 3
EPOCHS = 150
TRAIN_SIZE = 0.8
MODEL_VERSION = 6.0
INPUT_SIZE = (BATCH_SIZE, IMG_SIZE, IMG_SIZE, CHANNELS)
N_CLASSES = 6

dataset = keras.preprocessing.image_dataset_from_directory(
    directory='downloads',
    shuffle=True,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)
class_names = dataset.class_names
print(class_names)
train_ds, val_ds, test_ds = utils.get_dataset_partitions_tf(dataset)

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)

resize_and_rescale = keras.Sequential([
    keras.layers.experimental.preprocessing.Resizing(IMG_SIZE, IMG_SIZE),
    keras.layers.experimental.preprocessing.Rescaling(1.0 / 255)
])

data_augmentation = keras.Sequential([
    keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    keras.layers.experimental.preprocessing.RandomRotation(0.2)
])

with mlflow.start_run() as run:
    model = keras.models.Sequential([
        resize_and_rescale,
        data_augmentation,
        keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=1, activation='relu', input_shape=INPUT_SIZE),
        keras.layers.MaxPool2D((2, 2)),
        keras.layers.Dropout(0.1),
        keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=1, activation='relu',
                            kernel_regularizer=keras.regularizers.l2(0.01)),
        keras.layers.Dropout(0.2),
        keras.layers.MaxPool2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(N_CLASSES, activation='softmax')
        # keras.layers.Dense(N_CLASSES, activation='sigmoid')

    ])

    model.build(input_shape=INPUT_SIZE)
    # model.summary()

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',  # Metric to monitor (usually validation loss)
        patience=3,  # Number of epochs with no improvement after which training will be stopped
        restore_best_weights=True  # Restore model weights from the epoch with the best value of the monitored metric
    )

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath='best_model.h5',  # Filepath to save the best model
        monitor='val_loss',  # Metric to monitor (e.g., validation loss)
        save_best_only=True,  # Save only the best model
        save_weights_only=False,  # Save entire model including architecture
        mode='auto',  # 'auto' will choose the direction to monitor based on the metric
        verbose=1  # Print information about saving the model
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )


    class LogMetricsCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            mlflow.log_metric("train_accuracy", logs["accuracy"], step=epoch)
            mlflow.log_metric("val_accuracy", logs["val_accuracy"], step=epoch)
            mlflow.log_metric("loss", logs["loss"], step=epoch)
            mlflow.log_metric("val_loss", logs["val_loss"], step=epoch)


    # Create an instance of the custom callback
    log_metrics_callback = LogMetricsCallback()
    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
        validation_data=val_ds,
        callbacks=[log_metrics_callback, early_stopping]
    )
    final_train_accuracy = history.history['accuracy'][-1]
    final_val_accuracy = history.history['val_accuracy'][-1]
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]

    mlflow.log_metric("final_train_accuracy", final_train_accuracy)
    mlflow.log_metric("final_val_accuracy", final_val_accuracy)
    mlflow.log_metric("final_train_loss", final_train_loss)
    mlflow.log_metric("final_val_loss", final_val_loss)

    mlflow.keras.log_model(model, "model")

    # scores = model.evaluate(test_ds)
    mlflow.log_param("img_size", IMG_SIZE)
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("model_version", MODEL_VERSION)
    mlflow.log_param("n_classes", N_CLASSES)

    model.save(f'../politic_class/model/{MODEL_VERSION}')
    mlflow.log_artifacts(f'../politic_class/model/{MODEL_VERSION}')
    mlflow.end_run()

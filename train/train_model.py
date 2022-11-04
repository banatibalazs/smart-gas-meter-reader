import os

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf


CHECKPOINT_PATH = "/models/model_{epoch:02d}_{val_accuracy:02f}"
BATCH_SIZE = 128


resize_and_rescale = tf.keras.Sequential([
    # tf.keras.layers.Resizing(IMG_WIDTH, IMG_HEIGHT),
    tf.keras.layers.Rescaling(1. / 255)
])

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.15)
    # tf.keras.layers.RandomContrast(0.5)
])

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint('./models/' + '_{epoch:03d}-{val_loss:.2f}_.h5', monitor='val_accuracy')

def create_model():
    model = tf.keras.models.Sequential()
    model.add(resize_and_rescale)
    model.add(data_augmentation)
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    return model


# Load datasets
train_x = np.load("datasets/train_x.npy")
train_y = np.load("datasets/train_y.npy")
test_x = np.load("datasets/test_x.npy")
test_y = np.load("datasets/test_y.npy")


# Create new model
model = create_model()
optimizer = tf.keras.optimizers.Adam(learning_rate=10e-4)
model.compile(optimizer=optimizer,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])


history = model.fit(
    train_x, train_y,
    validation_data=(test_x, test_y),
    epochs=30,
    batch_size=BATCH_SIZE,
    shuffle=True,
    callbacks=[model_checkpoint_callback]
)

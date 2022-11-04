import os
import numpy as np
import tensorflow as tf
from helper_functions import create_model

PATH = "./models/"
BATCH_SIZE = 128

def scheduler(epoch, lr):
  if epoch < 20:
    return lr
  else:
    return lr * tf.math.exp(-0.1)

CALLBACKS = [tf.keras.callbacks.ModelCheckpoint(PATH + '_{epoch:03d}-{val_loss:.3f}_.h5', monitor='val_accuracy', save_best_only=True),
             tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1),
             tf.keras.callbacks.CSVLogger(PATH + "training.csv", separator=",")]

# Load datasets
train_x = np.load("datasets/train_x.npy")
train_y = np.load("datasets/train_y.npy")
test_x = np.load("datasets/test_x.npy")
test_y = np.load("datasets/test_y.npy")


# Create new model
model = create_model()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=10e-4),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])


history = model.fit(
    train_x, train_y,
    validation_data=(test_x, test_y),
    epochs=40,
    batch_size=BATCH_SIZE,
    shuffle=True,
    callbacks=[CALLBACKS]
)

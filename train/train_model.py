import numpy as np
import tensorflow as tf
from helper_functions import create_model

MODEL_PATH = "./models/"
DATA_PATH = './datasets/'
BATCH_SIZE = 128
EPOCHS = 20
LEARNING_RATE = 1e-4

def scheduler(epoch, lr):
  if epoch < 20:
    return lr
  else:
    return lr * tf.math.exp(-0.1)

CALLBACKS = [tf.keras.callbacks.ModelCheckpoint(MODEL_PATH + '_{epoch:03d}-{val_loss:.3f}_.h5', monitor='val_accuracy', save_best_only=True),
             tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1),
             tf.keras.callbacks.CSVLogger(MODEL_PATH + "training.csv", separator=",")]

# Load datasets
train_x = np.load(DATA_PATH + "train_x.npy")
train_y = np.load(DATA_PATH + "train_y.npy")
test_x = np.load(DATA_PATH + "test_x.npy")
test_y = np.load(DATA_PATH + "test_y.npy")


# Create new model
model = create_model()

# Load pretrained model
model = tf.keras.models.load_model('models/_023-0.026_.h5')

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])


history = model.fit(
    train_x, train_y,
    validation_data=(test_x, test_y),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=True,
    callbacks=[CALLBACKS]
)

import os
import numpy as np
import math
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

tf.get_logger().setLevel('ERROR')

DATA_DIR = "train_images/"
BATCH_SIZE = None
IMG_HEIGHT = 28
IMG_WIDTH = 28
TRAIN_VAL_RATIO = 0.01


# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

x_train = np.array(x_train, np.uint8)
x_test = np.array(x_test, np.uint8)
y_train = np.array(y_train, np.int64)
y_test = np.array(y_test, np.int64)

# Load local dataset (analog gas meter numbers)
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=TRAIN_VAL_RATIO,
    subset="training",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    color_mode='grayscale')

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=TRAIN_VAL_RATIO,
    subset="validation",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    color_mode='grayscale')

t_data, t_label = zip(*[(im.numpy(), lb.numpy()) for im, lb in iter(train_ds)])
v_data, v_label = zip(*[(im.numpy(), lb.numpy()) for im, lb in iter(val_ds)])

train_x = np.array(t_data, np.uint8)
test_x = np.array(v_data, np.uint8)
train_y = np.array(t_label, np.int64)
test_y = np.array(v_label, np.int64)

# Concatenate the two datasets
conc_train_x = np.concatenate((train_x, x_train), axis=0)
conc_train_y = np.concatenate((train_y, y_train), axis=0)
conc_test_x = np.concatenate((test_x, x_test), axis=0)
conc_test_y = np.concatenate((test_y, y_test), axis=0)

#Save datasets
np.save("datasets/train_x", conc_train_x)
np.save("datasets/train_y", conc_train_y)
np.save("datasets/test_x", conc_test_x)
np.save("datasets/test_y", conc_test_y)

#Plot train_images from concatenated dataset
IM_NUMBER = 64

idx = np.random.permutation(len(conc_train_x))
idx = idx[0:IM_NUMBER]
x, y = conc_train_x[idx], conc_train_y[idx]

dim = math.ceil(math.sqrt(len(x)))
rows = dim
cols = dim
fig = plt.figure(figsize=[15, 18])

for i, img in enumerate(x):
    # print(lbl)
    # print(img)
    ax = plt.subplot(rows, cols, (i + 1))
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img)
    plt.title("Label: " + str(y[i]))
plt.show()





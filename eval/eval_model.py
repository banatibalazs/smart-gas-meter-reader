import os
import numpy as np
import  matplotlib.pyplot as plt
import math

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

MODEL_PATH = '../train/models/model.h5'
DATA_PATH = '../train/datasets/'
PREDICT_ON_TEST = False

# Load data
train_x, train_y = np.load(DATA_PATH + "train_x.npy"), np.load(DATA_PATH + "train_y.npy")
test_x, test_y = np.load(DATA_PATH + "test_x.npy"), np.load(DATA_PATH + "test_y.npy")

# Load model
model = tf.keras.models.load_model(MODEL_PATH)
optimizer = tf.keras.optimizers.Adam(learning_rate=10e-5)
model.compile(optimizer=optimizer,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])


# Evaluate model on train and test
# model.evaluate(train_x, train_y)
model.evaluate(test_x, test_y)

# List mispredicted images
images, labels = (test_x, test_y) if PREDICT_ON_TEST else (train_x, train_y)
prediction = model.predict(images)


all_class_predictions = prediction
predictions = np.argmax(prediction, axis=1)
print(predictions.shape, type(prediction))

wrongs = np.where(np.equal(predictions, labels) == False)[0]
print(wrongs)


# Plot misclassified images
tup = np.asarray(list(zip(all_class_predictions, predictions, labels, images)), dtype=object)
tup = tup[wrongs]

dim = math.ceil(math.sqrt(len(tup)))
rows = dim
cols = dim
fig = plt.figure(figsize=[15, 18])

for i, (all_class_prediction, prediction, label, img) in enumerate(tup):
    ax = plt.subplot(rows, cols, (i + 1))
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img/255.)

    plt.title("Pred: " + str(prediction) + "\n" + "Label: " + str(label) + "\n" +
             str(np.argsort(all_class_prediction)[-1]) + ": " +
             str(round(sorted(all_class_prediction * 100)[-1], 2)) + " %" + "\n" +
             str(np.argsort(all_class_prediction)[-2]) + ": " +
             str(round(sorted(all_class_prediction * 100)[-2], 2))  + " %", fontsize=10)
plt.show()
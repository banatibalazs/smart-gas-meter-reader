from helper_functions import *
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import pandas as pd
import time
import os
import math
import matplotlib.pyplot as plt
import glob
import datetime
from tflite_support.task import core
from tflite_support.task import processor
from helper_functions import *
import cv2
from tqdm import tqdm

# Image paths
SOURCE_IMG_PATH = "./images/"
RESULT_IMG_PATH = "./results/predictions/"
INTERMETIATE_IMG_PATH = "./results/intermediate_results/"

# Parameters of object detection
WIDTH = 1000
HEIGHT = 140
DIM = (WIDTH, HEIGHT)
NUM_THREADS = 10
# DETECTOR_PATH = './numberplate_detector.tflite'
DETECTOR_PATH = './object_detector.tflite'

# Parameters of balancing algorithm (Hugh Lines)
balancing_cycles = 3

# Parameters of sharpener algorithm
tb_w = 50
tb_th = 0
tb_blur_size = 10
tb_blur_sigma = 50

# Adaptive threshold and blur
blockSize = 65
k = 0.3

# Contour constraints
h_min = 50
w_min = 25
w_max = 120
x_min = 25
x_max = 400
h_w_ratio_max = 3
h_w_ratio_min = 1.25


def analyze(image, object_detector, classification_model, filename):
    greyscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Balance tilted images
    balanced_image = balancing_tilted_image(image, greyscale_image, balancing_cycles)
    cv2.imwrite(f'{INTERMETIATE_IMG_PATH}0_balanced_images/' + filename, balanced_image)

    # Object detector detects the number plates and crops out detected area.
    detected_image = detect_numberplate(object_detector, balanced_image)
    cv2.imwrite(f'{INTERMETIATE_IMG_PATH}1_detected_images/' + filename, detected_image)

    # Resize and sharpen image
    resized_sharp_image = resize_and_sharpen_image(detected_image, DIM, tb_w, tb_th, tb_blur_size, tb_blur_sigma)
    cv2.imwrite(f'{INTERMETIATE_IMG_PATH}2_resized_and_sharp_images/' + filename, resized_sharp_image)

    # Niblack threshold and median-blur
    threshold_image = adaptive_threshold_and_median_blur(resized_sharp_image, blockSize, k)

    IMG_WIDTH = classification_model.layers[0].input_shape[1]
    IMG_HEIGHT = classification_model.layers[0].input_shape[2]

    # Contours and clip image into 8 pieces
    image_list, threshold_im = find_contours(threshold_image, resized_sharp_image, IMG_WIDTH, IMG_HEIGHT, h_min=h_min,
                                             w_min=w_min, w_max=w_max, x_min=x_min, x_max=x_max,
                                             h_w_ratio_max=h_w_ratio_max, h_w_ratio_min=h_w_ratio_min)
    cv2.imwrite(f'{INTERMETIATE_IMG_PATH}/3_threshold_images/' + filename, threshold_im)

    tensor = tf.image.rgb_to_grayscale(image_list)
    prediction_array = np.argmax(classification_model.predict(tensor, verbose=False), axis=1)
    prediction_str = ''.join([str(num) for num in prediction_array])

    for i, im in enumerate(image_list):
        cv2.imwrite(f'{RESULT_IMG_PATH}{prediction_array[i]}/' + '_' + filename, im)

    return prediction_str, resized_sharp_image, image_list

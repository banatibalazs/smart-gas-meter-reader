from mqtt_server_and_analyzer.helper_functions import *
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
from mqtt_server_and_analyzer.helper_functions import *
import cv2
from tqdm import tqdm

# Model paths
OBJECT_DETECTOR_PATH = '../../setup/setup_server/models/object_detector.tflite'
CLASSIFICATION_MODEL_PATH = '../../setup/setup_server/models/classifier.h5'


# Image paths
SOURCE_IMG_PATH = "./images/"
RESULT_IMG_PATH = "./results/predictions/"
INTERMEDIATE_IMG_PATH = "./results/intermediate_results/"
UNPROCESSED_IMG_PATH = "./results/unprocessed_images/"

# Parameters of object detection
WIDTH = 1000
HEIGHT = 140
DIM = (WIDTH, HEIGHT)
NUM_THREADS = 10


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
h_min = 60
w_min = 30
w_max = 120
x_min = 25
x_max = 400
h_w_ratio_max = 3
h_w_ratio_min = 1.25


def analyze(cur_image, object_detector, cl_model, fname):
    greyscale_image = cv2.cvtColor(cur_image, cv2.COLOR_BGR2GRAY)

    # Balance tilted images
    balanced_image = balancing_tilted_image(cur_image, greyscale_image, balancing_cycles)
    cv2.imwrite(f'{INTERMEDIATE_IMG_PATH}0_balanced_images/' + fname, balanced_image)

    # Object detector detects the number plates and crops out detected area.
    detected_image = detect_numberplate(object_detector, balanced_image)
    cv2.imwrite(f'{INTERMEDIATE_IMG_PATH}1_detected_images/' + fname, detected_image)

    # Resize and sharpen image
    resized_sharp_image = resize_and_sharpen_image(detected_image, DIM, tb_w, tb_th, tb_blur_size, tb_blur_sigma)
    cv2.imwrite(f'{INTERMEDIATE_IMG_PATH}2_resized_and_sharp_images/' + fname, resized_sharp_image)

    # Niblack threshold and median-blur
    threshold_image = adaptive_threshold_and_median_blur(resized_sharp_image, blockSize, k)

    IMG_WIDTH = cl_model.layers[0].input_shape[1]
    IMG_HEIGHT = cl_model.layers[0].input_shape[2]

    # Contours and clip image into 8 pieces
    image_list, threshold_im = find_contours(threshold_image, resized_sharp_image, IMG_WIDTH, IMG_HEIGHT, h_min=h_min,
                                             w_min=w_min, w_max=w_max, x_min=x_min, x_max=x_max,
                                             h_w_ratio_max=h_w_ratio_max, h_w_ratio_min=h_w_ratio_min)
    cv2.imwrite(f'{INTERMEDIATE_IMG_PATH}/3_threshold_images/' + fname, threshold_im)

    tensor = tf.image.rgb_to_grayscale(image_list)
    prediction_array = np.argmax(cl_model.predict(tensor, verbose=False), axis=1)
    prediction_str = ''.join([str(num) for num in prediction_array])

    for j, im in enumerate(image_list):
        cv2.imwrite(f'{RESULT_IMG_PATH}{prediction_array[j]}/' + '_' + fname, im)

    return prediction_str, resized_sharp_image, image_list


if __name__ == '__main__':
    # Clear the folders
    root = RESULT_IMG_PATH
    paths = [di for di in glob.glob(root + "*") if os.path.isfile(di)]
    for filename in paths:
        os.remove(filename)

    root = RESULT_IMG_PATH
    paths = [di + "/" for di in glob.glob(root + "*") if os.path.isdir(di)]
    for folder in paths:
        filenames = glob.glob(folder + "*.jpg")
        for filename in filenames:
            os.remove(filename)

    root = INTERMEDIATE_IMG_PATH
    paths = [di + "/" for di in glob.glob(root + "*") if os.path.isdir(di)]
    for folder in paths:
        filenames = glob.glob(folder + "*.jpg")
        for filename in filenames:
            os.remove(filename)


    # Load object detection model
    base_options = core.BaseOptions(
        file_name=OBJECT_DETECTOR_PATH, use_coral=False, num_threads=NUM_THREADS)
    detection_options = processor.DetectionOptions(
        max_results=3, score_threshold=0.1)
    options = vision.ObjectDetectorOptions(
        base_options=base_options, detection_options=detection_options)
    detector = vision.ObjectDetector.create_from_options(options)

    # Load image classification model
    classification_model = tf.keras.models.load_model(CLASSIFICATION_MODEL_PATH)

    # Load images
    path_list = [SOURCE_IMG_PATH + file for file in os.listdir(SOURCE_IMG_PATH) if '.jpg' in file]
    filename_list = [file for file in os.listdir(SOURCE_IMG_PATH) if '.jpg' in file]

    skipped_images = []
    for i, img_path in tqdm(enumerate(path_list), total=len(path_list)):
        try:
            image = cv2.imread(img_path, cv2.IMREAD_COLOR)
            pred, img, image_list = analyze(image, detector, classification_model, filename_list[i])
            cv2.imwrite(RESULT_IMG_PATH + f"{i}_" + pred[:-3] + "_" + pred[-3:] + ".jpg", img)

            with open(RESULT_IMG_PATH + "result_images.csv", 'a') as f:
                # f.write(f"{datetime.datetime.now()}, {pred}\n")
                f.write(f"{filename_list[i]}, {pred}\n")
        except Exception as e:
            # print(img_path, e)
            skipped_images.append(img_path)
            try:
                cv2.imwrite(UNPROCESSED_IMG_PATH + f"{filename_list[i]}_" + ".jpg", image)
            except Exception as exc:
                print(exc, "--------------")


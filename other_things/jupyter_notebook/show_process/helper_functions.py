import matplotlib.pyplot as plt
import pandas as pd
import math
import cv2
import numpy as np
# from tflite_support.task import vision
import traceback
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf


def plot_csv_metrics(path):
    dataframe = pd.read_csv(path, sep=',')
    epoch = dataframe['epoch'].to_numpy()
    accuracy = dataframe['accuracy'].to_numpy()
    loss = dataframe['loss'].to_numpy()
    val_accuracy = dataframe['val_accuracy'].to_numpy()
    val_loss = dataframe['val_loss'].to_numpy()

    plt.figure(figsize=(20, 7))
    plt.subplot(1, 2, 1)
    plt.plot(epoch, accuracy, label="accuracy")
    plt.plot(epoch, val_accuracy, label="val_accuracy")
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epoch, loss, label="loss")
    plt.plot(epoch, val_loss, label="val_loss")
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()


def get_augmented_ds(data_dir="./eval_data/", train_test_ratio=0.01, seed=125, img_h=28, img_w=28,
                     colormode='grayscale', batch_size=128, rotation=0.15):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=train_test_ratio,
        subset="training",
        seed=seed,
        image_size=(img_h, img_w),
        batch_size=None,
        color_mode=colormode)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=train_test_ratio,
        subset="validation",
        seed=seed,
        image_size=(img_h, img_w),
        batch_size=None,
        color_mode=colormode)

    train_x, train_y = zip(*[(a.numpy(), b.numpy()) for a, b in iter(train_ds)])
    test_x, test_y = zip(*[(a.numpy(), b.numpy()) for a, b in iter(val_ds)])

    train_x = np.array(np.squeeze(train_x) / 255., np.float32)
    test_x = np.array(np.squeeze(test_x) / 255., np.float32)

    train_x = train_x.reshape((train_x.shape[0], train_x.shape[1], train_x.shape[2], 1))
    test_x = test_x.reshape((test_x.shape[0], test_x.shape[1], test_x.shape[2], 1))

    train_y = np.array(train_y, np.int64)
    test_y = np.array(test_y, np.int64)

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(rotation)
        # tf.keras.layers.RandomContrast(0.5)
    ])

    train_ds = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    test_ds = tf.data.Dataset.from_tensor_slices((test_x, test_y))

    train_ds = train_ds.shuffle(len(train_ds), reshuffle_each_iteration=True)
    test_ds = test_ds.shuffle(len(test_ds), reshuffle_each_iteration=True)

    # Batch all datasets.
    train_ds = train_ds.batch(batch_size)
    test_ds = test_ds.batch(batch_size)

    train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y),
                            num_parallel_calls=tf.data.AUTOTUNE)

    test_ds = test_ds.map(lambda x, y: (data_augmentation(x, training=True), y),
                          num_parallel_calls=tf.data.AUTOTUNE)

    augmented_train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    augmented_test_ds = test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    # train_ds = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    # test_ds = tf.data.Dataset.from_tensor_slices((test_x, test_y))

    return augmented_train_ds, augmented_test_ds


class Balancer:

    def __init__(self,
                 threshold_1=50,
                 threshold_2=200,
                 aperture_size=3,
                 hough_rho=1,
                 hough_theta=(np.pi / 180),
                 hough_threshold=110,
                 hough_min_theta=0,
                 hough_max_theta=0,
                 max_angle=15):

        self.lined_image = None
        self.filtered_lines = None
        self.max_angle = max_angle

        self.canny_threshold_1 = threshold_1
        self.canny_threshold_2 = threshold_2
        self.canny_aperture_size = aperture_size
        # cv2.Canny(greyscale_image, threshold_1, threshold_2, None, aperture_size)

        self.hough_rho = hough_rho
        self.hough_theta = hough_theta
        self.hough_threshold = hough_threshold
        self.hough_min_theta = hough_min_theta
        self.hough_max_theta = hough_max_theta
        # cv2.HoughLines(dst, hough_rho, hough_theta, hough_threshold, None, hough_min_theta, hough_max_theta)

    def balance_image(self, image, balancing_cycles=3):

        greyscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        for _ in range(balancing_cycles):
            dst = cv2.Canny(greyscale_image,
                            self.canny_threshold_1,
                            self.canny_threshold_2,
                            None,
                            self.canny_aperture_size)
            cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)

            lines = cv2.HoughLines(dst,
                                   self.hough_rho,
                                   self.hough_theta,
                                   self.hough_threshold,
                                   None,
                                   self.hough_min_theta,
                                   self.hough_max_theta)
            sizemax = math.sqrt(cdst.shape[0] ** 2 + cdst.shape[1] ** 2)

            # Draw the lines
            if lines is not None:
                average_angle = 0
                line_num = 0
                filtered_lines = []
                for i in range(0, len(lines)):
                    rho = lines[i][0][0]
                    theta = lines[i][0][1]

                    cur_angle = theta * 180 / math.pi
                    # print("Theta value: " + str(cur_angle))
                    if abs((90 - cur_angle)) < self.max_angle:
                        average_angle += cur_angle
                        line_num += 1
                        # print(str(cur_angle))

                        a = math.cos(theta)
                        b = math.sin(theta)
                        x0 = a * rho
                        y0 = b * rho
                        # Computing line endpoints outside of image matrix
                        pt1 = (int(x0 + sizemax * (-b)), int(y0 + sizemax * a))
                        pt2 = (int(x0 - sizemax * (-b)), int(y0 - sizemax * a))
                        cv2.line(cdst, pt1, pt2, (255, 0, 0), 3, cv2.LINE_AA)
                        filtered_lines.append(lines[i])

                self.filtered_lines = filtered_lines
                self.lined_image = cdst
                rows, cols = greyscale_image.shape[:2]
                M = cv2.getRotationMatrix2D((cols / 2, rows / 2), ((average_angle / line_num) - 90), 1)
                greyscale_image = cv2.warpAffine(greyscale_image, M, (cols, rows))
                image = cv2.warpAffine(image, M, (cols, rows))

        return image

    def get_lined_image(self):
        return self.lined_image

    def get_lines(self):
        return self.filtered_lines


def balancing_tilted_image(image, greyscale_image, balancing_cycles):
    # Rotating the image
    for _ in range(balancing_cycles):
        dst = cv2.Canny(greyscale_image, 50, 200, None, 3)
        cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
        lines = cv2.HoughLines(dst, 1, np.pi / 180, 110, None, 0, 0)
        sizemax = math.sqrt(cdst.shape[0] ** 2 + cdst.shape[1] ** 2)

        # Draw the lines
        if lines is not None:
            average_angle = 0
            line_num = 0
            for i in range(0, len(lines)):
                rho = lines[i][0][0]
                theta = lines[i][0][1]

                cur_angle = theta * 180 / math.pi
                # print("Theta value: " + str(cur_angle))
                if abs((90 - cur_angle)) < 15:
                    average_angle += cur_angle
                    line_num += 1
                    # print(str(cur_angle))

                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                # Computing line endpoints outside of image matrix
                pt1 = (int(x0 + sizemax * (-b)), int(y0 + sizemax * a))
                pt2 = (int(x0 - sizemax * (-b)), int(y0 - sizemax * a))
                cv2.line(cdst, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)

            # print(str("Average line angle: " + str((average_angle / line_num) - 90)))
            try:
                # assert line_num == 0
                rows, cols = greyscale_image.shape[:2]
                M = cv2.getRotationMatrix2D((cols / 2, rows / 2), ((average_angle / line_num) - 90), 1)
                greyscale_image = cv2.warpAffine(greyscale_image, M, (cols, rows))
                image = cv2.warpAffine(image, M, (cols, rows))

            except Exception as e:
                tb = traceback.format_exc()
                raise Exception(tb)

    return image


def detect_numberplate(detector, image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Create a TensorImage object from the RGB image.
    input_tensor = vision.TensorImage.create_from_array(rgb_image)
    # Run object detection estimation using the model.
    try:
        detection_result = detector.detect(input_tensor)
        bbox = detection_result.detections[0].bounding_box
    except Exception as e:
        tb = traceback.format_exc()
        raise Exception(tb)

    return image[bbox.origin_y: bbox.origin_y + bbox.height, bbox.origin_x: bbox.origin_x + bbox.width, :]


def resize_and_sharpen_image(image, DIM, tb_w, tb_th, tb_blur_size, tb_blur_sigma):
    image = cv2.resize(image, DIM, interpolation=cv2.INTER_LANCZOS4)
    # Sharpening, not always produces good result_images, especially when the original image is faint or dark.
    imLab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    imL, ima, imb = cv2.split(imLab)
    w = tb_w / 10.0
    th = tb_th
    blur_size = tb_blur_size * 2 + 3
    blur_sigma = tb_blur_sigma / 10.0
    im_blur = cv2.GaussianBlur(imL, (blur_size, blur_size), blur_sigma)
    im_diff = cv2.subtract(imL, im_blur, dtype=cv2.CV_16S)
    im_abs_diff = cv2.absdiff(imL, im_blur)
    im_diff_masked = im_diff.copy()
    im_diff_masked[im_abs_diff < th] = 0
    imL_sharpen = cv2.add(imL, w * im_diff_masked, dtype=cv2.CV_8UC1)

    res_Lab = cv2.merge([imL_sharpen, ima, imb])
    res_bgr = cv2.cvtColor(res_Lab, cv2.COLOR_Lab2BGR)

    return res_bgr


def adaptive_threshold_and_median_blur(image, blockSize, k):
    res_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    res_grey = cv2.ximgproc.niBlackThreshold(res_grey, 255, cv2.THRESH_BINARY, blockSize, k,
                                             binarizationMethod=cv2.ximgproc.BINARIZATION_NIBLACK, r=106)
    res_grey = cv2.medianBlur(res_grey, 3)
    res_grey = cv2.medianBlur(res_grey, 5)
    res_grey = cv2.medianBlur(res_grey, 7)
    res_grey = cv2.medianBlur(res_grey, 9)

    return res_grey


def find_contours(threshold_im, original_im, img_width, img_height, h_min=50, w_min=25, w_max=120, x_min=25, x_max=400, h_w_ratio_max=3, h_w_ratio_min=1, y_min=25, y_max=125):
    # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv2.findContours(threshold_im, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)

    try:
        # h_min = 50
        # w_min = 25
        # w_max = 120
        # x_min = 25
        # x_max = 400
        # h_w_ratio_max = 3
        # h_w_ratio_min = 1

        threshold_im = cv2.cvtColor(threshold_im, cv2.COLOR_GRAY2BGR)

        filtered_contours = []
        for cntrIdx in range(0, len(contours)):
            x, y, w, h = cv2.boundingRect(contours[cntrIdx])

            if h < h_min or w < w_min or w > w_max or x < x_min or x > x_max or h / w > h_w_ratio_max or h / w < h_w_ratio_min or y < y_min or (y+h) > y_max:
                continue

            # if h < 80:
            #     continue

            start_point = x, y
            end_point = x + w, y + h
            threshold_im = cv2.rectangle(threshold_im, start_point, end_point, (0, 0, 255), 3)

            filtered_contours.append({
                "x": x,
                "y": y,
                "w": w,
                "h": h
            })
        filtered_contours = sorted(filtered_contours, key=lambda d: d['x'])

        if len(filtered_contours) <= 2:
            clipping_1 = round((filtered_contours[1]["x"] - (filtered_contours[0]["x"] + filtered_contours[0]["w"])) / 2 + (
                    filtered_contours[0]["x"] + filtered_contours[0]["w"]))
            clipping_offset = filtered_contours[1]["x"] - (filtered_contours[0]["x"])
        else:
            clipping_1 = round(
                (filtered_contours[1]["x"] - (filtered_contours[0]["x"] + filtered_contours[0]["w"])) / 2 + (
                        filtered_contours[0]["x"] + filtered_contours[0]["w"]))
            clipping_2 = round((filtered_contours[2]["x"] - (filtered_contours[1]["x"] + filtered_contours[1]["w"])) / 2 + (
                        filtered_contours[1]["x"] + filtered_contours[1]["w"]))
            clipping_offset = (clipping_2 - clipping_1)

        image_list = []
        for i in range(8):
            image = original_im[:,
                    ((clipping_1 + i * clipping_offset) - clipping_offset) : (clipping_1 + i * clipping_offset),
                    :]
            image = cv2.resize(image, (img_width, img_height), interpolation=cv2.INTER_LANCZOS4)
            image_list.append(image)

    except Exception as e:
        tb = traceback.format_exc()
        raise Exception(tb)

    return image_list, threshold_im


# resize_and_rescale = tf.keras.Sequential([
#     # tf.keras.layers.Resizing(IMG_WIDTH, IMG_HEIGHT),
#     tf.keras.layers.Rescaling(1. / 255)
# ])
#
# data_augmentation = tf.keras.Sequential([
#     tf.keras.layers.RandomRotation(0.15)
#     # tf.keras.layers.RandomContrast(0.5)
# ])

# def create_model():
#     model = tf.keras.models.Sequential([
#         tf.keras.layers.Rescaling(1. / 255),
#         tf.keras.layers.RandomRotation(0.15),
#         tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
#         tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
#         # tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
#         # tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.MaxPooling2D((2, 2)),
#         tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
#         # tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.MaxPooling2D((2, 2)),
#         tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
#         # tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.Flatten(),
#         tf.keras.layers.Dense(128, activation='relu'),
#         tf.keras.layers.Dense(10, activation='softmax')
#     ])
#
#     return model

def create_model():
    resize_and_rescale = tf.keras.Sequential([
        # tf.keras.layers.Resizing(IMG_WIDTH, IMG_HEIGHT),
        tf.keras.layers.Rescaling(1. / 255)
    ])

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.15)
        # tf.keras.layers.RandomContrast(0.5)
    ])

    model = tf.keras.models.Sequential()
    model.add(resize_and_rescale)
    model.add(data_augmentation)
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    return model


def normalize_image(image, img_width, img_height):
    # image = image / 255.
    image = cv2.resize(image, (img_width, img_height), interpolation=cv2.INTER_LANCZOS4)
    return image

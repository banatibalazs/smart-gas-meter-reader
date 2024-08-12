# python3.6
import datetime
import io
import random
import sys
import traceback
import PIL.Image as Image
import cv2
import numpy as np
from paho.mqtt import client as mqtt_client
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
from credentials import PASSWORD
import tensorflow as tf
from setup.setup_server.mqtt_and_analyze.mqtt_and_analyze import *

password = PASSWORD

broker = '192.168.0.33'  # Set the broker address
port = 1883 # Set the broker port
sub_esp_1_photo = "esp_cam_1/from_esp"
pub_esp_1_photo = "esp_cam_1/photo"
pub_esp_1_sleep = "esp_cam_1/sleep"
pub_hello_esp_1 = "esp_cam_1/hello"
sub_hello_esp_1 = "esp_cam_1/hello_pub"
# generate client ID with pub prefix randomly
client_id = f'python-mqtt-{random.randint(0, 100)}'
# username = 'emqx'
# password = 'public'

TF_LITE_MODEL_PATH = 'models/object_detector.tflite'
TF_MODEL_PATH = 'models/classifier.h5'
# tf_model = create_model()
# tf_model.load_weights(TF_MODEL_PATH)

tf_model = tf.keras.models.load_model(TF_MODEL_PATH)

optimizer = tf.keras.optimizers.Adam(learning_rate=10e-5)
tf_model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])

num_threads = 10
enable_edgetpu = False

base_options = core.BaseOptions(
    file_name=TFLITE_MODEL_PATH, use_coral=enable_edgetpu, num_threads=num_threads)
detection_options = processor.DetectionOptions(
    max_results=3, score_threshold=0.1)
options = vision.ObjectDetectorOptions(
    base_options=base_options, detection_options=detection_options)
detector = vision.ObjectDetector.create_from_options(options)


#Image paths
RESULT_IMG_PATH = "result/"

#Parameters of object detection
WIDTH = 1000
HEIGHT = 140
DIM = (WIDTH, HEIGHT)
NUM_THREADS = 10
DETECTOR_PATH = './models/object_detector.tflite'

#Parameters of balancing algorithm (Hugh Lines)
balancing_cycles = 3

#Parameters of sharpener algorithm
tb_w = 70
tb_th = 0
tb_blur_size = 10
tb_blur_sigma = 50

#Adaptive threshold and blur
blockSize = 65
k = 0.5

#Contours
h_min = 60
w_min = 25
w_max = 120
x_min = 30
x_max = 400
y_min = 15
y_max = 135
h_w_ratio_max = 3.99
h_w_ratio_min = 1.0


def analyze(image, object_detector_model, number_classifier_model):
    greyscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Balance image
    balanced_image = balancing_tilted_image(image, greyscale_image, balancing_cycles)
    print("balance: OK")
    # Object detection and crop detected area
    detected_image = detect_numberplate(object_detector_model, balanced_image)
    print("detection: OK")
    # Resize and sharpen image
    resized_sharp_image = resize_and_sharpen_image(detected_image, DIM, tb_w, tb_th, tb_blur_size, tb_blur_sigma)
    print("Sharpening: OK")

    # Niblack threshold and medianblur
    threshold_image = adaptive_threshold_and_median_blur(resized_sharp_image, blockSize, k)
    print("Threshold: OK")

    IMG_WIDTH = number_classifier_model.layers[0].input_shape[1]
    IMG_HEIGHT = number_classifier_model.layers[0].input_shape[2]
    # Contours and clip image into 8 pieces
    image_list, threshold_im = find_contours(threshold_image, resized_sharp_image, 28, 28, h_min=h_min,
                                             w_min=w_min, w_max=w_max, x_min=x_min, x_max=x_max,
                                             h_w_ratio_max=h_w_ratio_max, h_w_ratio_min=h_w_ratio_min, y_min=y_min,
                                             y_max=y_max)

    print("Contour: OK")
    tensor = tf.image.rgb_to_grayscale(image_list)

    print(len(image_list), tensor.shape)
    prediction_array = np.argmax(number_classifier_model.predict(tensor, verbose=False), axis=1)
    prediction_str = ''.join([str(num) for num in prediction_array])

    return prediction_str, resized_sharp_image, image_list

def connect_mqtt() -> mqtt_client:
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print("Failed to connect, return code %d\n", rc)

    client = mqtt_client.Client(client_id)
    # client.username_pw_set(username, password)
    client.on_connect = on_connect
    client.connect(broker, port)

    filename = datetime.datetime.timestamp(datetime.datetime.now())
    print("date and time:", filename)
    return client


def subscribe(client: mqtt_client):
    def on_message(client, userdata, message):
        print("received message: ", )
        if message.topic == sub_hello_esp_1:
            if 'Setup is ready' in str(message.payload):
                print(str(message.payload))
                client.publish(pub_esp_1_photo, 'Photo')

        if message.topic == sub_esp_1_photo:
            try:
                bytes = bytearray(message.payload)

                pil_image = Image.open(io.BytesIO(bytes)).convert('RGB')
                open_cv_image = np.array(pil_image)
                # Convert RGB to BGR
                image = open_cv_image[:, :, ::-1].copy()

                filename = round(datetime.datetime.timestamp(datetime.datetime.now()))
                print("date and time:", filename)

                pred, img, image_list = analyze(image, detector, tf_model)

                idx = len(os.listdir(RESULT_IMG_PATH))
                cv2.imwrite(RESULT_IMG_PATH + f"{idx}_" + pred[:-3] + "_" + pred[-3:] + ".jpg", img)

                with open(RESULT_IMG_PATH + "result_images.csv", 'a') as f:
                    # f.write(f"{datetime.datetime.now()}, {pred}\n")
                    f.write(f"{filename}, {pred}\n")

                print(pred)


            except Exception as e:
                tb = traceback.format_exc()
                print(tb)

            sys.exit()

    client.subscribe(sub_esp_1_photo)
    client.subscribe(sub_hello_esp_1)
    client.on_message = on_message


def run():
    client = connect_mqtt()
    subscribe(client)
    client.publish(pub_hello_esp_1, "Hello")
    client.publish(pub_esp_1_sleep, 1) # Sleep ESP32 camera, message represents the time in seconds
    client.loop_forever()


if __name__ == '__main__':
    run()


# python3.6

import random
from paho.mqtt import client as mqtt_client
import time
import sys
import math
import cv2
import numpy as np
import os
import io
import PIL.Image as Image
import requests
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
import utils

broker   = '192.168.0.33'
port     = 1883
topic    = "esp_cam_1/from_esp"
# generate client ID with pub prefix randomly
client_id  = f'python-mqtt-{random.randint(0, 100)}'
# username = 'emqx'
# password = 'public'

dir_path       = "/home/balazs/Asztal/gas_pics/"
model          = '/home/balazs/Asztal/object_detection/gas_number.tflite'
num_threads    = 2
enable_edgetpu = False

base_options = core.BaseOptions(
    file_name=model, use_coral=enable_edgetpu, num_threads=num_threads)
detection_options = processor.DetectionOptions(
    max_results=3, score_threshold=0.1)
options = vision.ObjectDetectorOptions(
    base_options=base_options, detection_options=detection_options)
detector = vision.ObjectDetector.create_from_options(options)

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
    return client


def subscribe(client: mqtt_client):
    # def on_message(client, userdata, msg):
    #     print(f"Received `{msg.payload.decode()}` from `{msg.topic}` topic")
    def on_message(client, userdata, message):
        print("received message: ", )
        if message.topic == "esp_cam_2/from_esp":
            bytes = bytearray(message.payload)

            pil_image = Image.open(io.BytesIO(bytes)).convert('RGB')
            open_cv_image = np.array(pil_image)
            # Convert RGB to BGR
            image = open_cv_image[:, :, ::-1].copy()
            cv2.imshow("Opencv image", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            try:
                # Convert the image from BGR to RGB as required by the TFLite model.
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # Create a TensorImage object from the RGB image.
                input_tensor = vision.TensorImage.create_from_array(rgb_image)

                # Run object detection estimation using the model.
                detection_result = detector.detect(input_tensor)

                # Draw keypoints and edges on input image
                image = utils.visualize(image, detection_result)
                cv2.imshow("Opencv image", image)
                cv2.waitKey(0)
            except:
                print("There was some error in visualization.")


    client.subscribe(topic)
    client.on_message = on_message


def run():
    client = connect_mqtt()
    subscribe(client)
    client.loop_forever()


if __name__ == '__main__':
    run()

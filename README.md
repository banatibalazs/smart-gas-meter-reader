# smart-gas-meter-reader
This project applies an esp-cam to take photos of an analog gas meter. The camera and a linux server communicate via mqtt protocol. An mqtt broker and a client program runs on the server wich periodically sends a message to the camera to take a photo. On the server, a .tflite object detector detects the numberplate's position on the image (bounding box), which will be cut out. The locations of the numbers are detected by an opencv script. The individual numbers are the input of a simple CNN model that classifies the numbers.

<img src="./demo_images/schematic_drawing.png" width="600">

## Labeling the dataset for object detection

[link to labeling tool](github.com/HumanSignal/labelImg)

## Esp-Cam installation:

## Steps of prediction:

1. ### Balancing
    If the image is tilted, the object detector marks a bigger area. Therefor it is important to balance the images.
    For this task, Hugh lines algorithm is used.

<p align="center">
  <img src="./demo_images/balanced_tilted_image.png" width="300">
</p>

2. ### Dial-plate detection
    The object detector is a tensorflow lite model trained in colab.
    Model architecture is EfficientNetV4.
    https://colab.research.google.com/github/khanhlvg/tflite_raspberry_pi/blob/main/object_detection/Train_custom_model_tutorial.ipynb

<p align="center">
    <img src="./demo_images/cropped_raw_dial_plate.png" width="300">
</p>

3. ### Sharpening and resizing 
    The detected images are resized to 140x1000 pixels.

<p align="center">
    <img src="./demo_images/sharpened_resized_dial_plate.png" width="300">
</p>

4. ### Applying Adaptive threshold algorithm 
5. ### Contour searching on threshold image
    The aim is to find the coordinates of the individual numbers on the numberplate.
    On the basis of the found contours' coordinates, the 140x1000 px images are cut into 8 pieces

<p align="center">
    <img src="./demo_images/contour_dial_plate.png" width="300">
</p>

6. ### Classify the image pieces

<p align="center">
    <img src="./demo_images/number_1.png" width="90">
    <img src="./demo_images/number_3.png" width="90">
    <img src="./demo_images/number_6.png" width="90">
</p>

   A simple ad hoc tensorflow CNN classifies the images into 10 classes. Because of the similarity of the problem, for the model training the dataset was combined with MNIST dataet.


## Results

# smart-gas-meter-reader

![Python](https://img.shields.io/badge/Python-3.8-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.4-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5-green)
![Paho MQTT](https://img.shields.io/badge/Paho%20MQTT-1.5.1-red)
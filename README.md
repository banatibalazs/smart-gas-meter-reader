# Smart Gas Meter Reader

This project utilizes an ESP-CAM to capture images of an analog gas meter. The camera communicates with a Linux server via the MQTT protocol. An MQTT broker and a client program run on the server, periodically instructing the camera to take a photo. On the server, a TensorFlow Lite object detector identifies the number plate's position in the image (bounding box), which is then cropped. The locations of the numbers are detected using an OpenCV script. The individual numbers are then classified by a simple CNN model.

## Programming the ESP32-CAM

1. **Install the Arduino IDE**: Download and install the Arduino IDE from the [official website](https://www.arduino.cc/en/software).

2. **Add the ESP32 Board to Arduino IDE**:
    - Open Arduino IDE.
    - Go to `File` > `Preferences`.
    - In the `Additional Board Manager URLs` field, add: `https://dl.espressif.com/dl/package_esp32_index.json`.
    - Go to `Tools` > `Board` > `Boards Manager`.
    - Search for `esp32` and install the `esp32` package.

3. **Connect the ESP32-CAM**:
    - Connect the ESP32-CAM to your computer using a USB-to-serial adapter.
    - Select the correct board and port in Arduino IDE: `Tools` > `Board` > `ESP32 Wrover Module` and `Tools` > `Port`.

4. **Upload the Code**:
    - Open the ESP32-CAM sketch from `File` > `Examples` > `ESP32` > `Camera` > `CameraWebServer`.
    - Modify the WiFi credentials in the sketch.
    - Upload the sketch to the ESP32-CAM.

5. **Test the Camera**:
    - Open the Serial Monitor to get the IP address of the ESP32-CAM.
    - Open a web browser and enter the IP address to see the camera feed.

## Training the Object Detector

1. **Set Up Google Colab**:
    - Open the [Model Maker Object Detection for Android Figurine](https://colab.research.google.com/github/khanhlvg/tflite_raspberry_pi/blob/main/object_detection/Train_custom_model_tutorial.ipynb) notebook in Google Colab.

2. **Prepare the Dataset**:
    - Label the dataset using the [labelImg](https://github.com/HumanSignal/labelImg) tool.
    - Upload the labeled dataset to Google Colab.

3. **Train the Model**:
    - Follow the steps in the Colab notebook to train the object detection model.
    - Download the trained TensorFlow Lite model.

4. **Compile for EdgeTPU**:
    - Use the EdgeTPU compiler to compile the model for EdgeTPU.
    - Download the compiled model to your local computer.

## Training the Image Classifier Model

1. **Prepare the Dataset**:
    - Combine the local dataset with the MNIST dataset.
    - Save the combined dataset.

2. **Train the Model**:
    - Use the `train/create_datasets.py` script to load and preprocess the dataset.
    - Train a simple CNN model using TensorFlow.

3. **Save the Model**:
    - Save the trained model in TensorFlow Lite format.

4. **Deploy the Model**:
    - Deploy the TensorFlow Lite model to the server for inference.


## ESP-CAM installation:

## Steps of prediction:

1. ### Balancing
    If the image is tilted, the object detector marks a larger area. Therefore, it is important to balance the images.
    For this task, the Hough Lines algorithm is used.
    ![img.png](demo_images/gas_meter_whole.png)
2. ### Number plate detection
    The object detector is a TensorFlow Lite model trained in Colab.
    Model architecture is EfficientNetV4.
    https://colab.research.google.com/github/khanhlvg/tflite_raspberry_pi/blob/main/object_detection/Train_custom_model_tutorial.ipynb
    ![img_1.png](demo_images/numbers_raw.png)
3. ### Sharpening and resizing
    The detected images are resized to 140x1000 pixels.
    ![img_2.png](demo_images/numbers_sharpened.png)
4. ### Applying Adaptive threshold algorithm
5. ### Contour searching on threshold image
    The goal is to find the coordinates of the individual numbers on the number plate.
    Based on the found contours' coordinates, the 140x1000 px images are cut into 8 pieces.
    ![img_3.png](demo_images/numbers_contours.png)
6. ### Classify the image pieces
   ![img_4.png](demo_images/number_1.png)
   ![img_5.png](demo_images/number_3.png)
   ![img_6.png](demo_images/number_6.png)
   A simple ad hoc TensorFlow CNN classifies the images into 10 classes. Due to the similarity of the problem, the dataset for model training was combined with the MNIST dataset.

## Results


## ESP32-CAM and MQTT Server Communication

The ESP32-CAM communicates with the MQTT server to send and receive messages. The MQTT broker, such as Mosquitto, runs on a Linux server. The ESP32-CAM connects to the broker over WiFi using the MQTT protocol. It subscribes to a specific topic (e.g., `esp32/cam`) to receive commands. When a message is published to this topic, the ESP32-CAM processes the command, such as taking a photo, and can publish the result back to another topic for the server to process. This setup allows for efficient and real-time communication between the ESP32-CAM and the server.
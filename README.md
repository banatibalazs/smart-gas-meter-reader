# smart-gas-meter-reader
This project applies esp-cam to take photos of an analog gas meter.  The photos are analyzed by a .tflite object detector,  witch detects the numberplate, then a simple CNN  model classifies the number.

## Esp-Cam installation:

## Steps of prediction:

1. ### Balancing
    If the image is tilted, the object detector marks a bigger area. Therefor it is important to balance the images.
    For this task, Hugh lines algorithm is used.
    ![img.png](img.png)
2. ### Numberplate detection
    The object detector is a tensorflow lite model trained in colab.
    Model architecture is EfficientNetV4.
    https://colab.research.google.com/github/khanhlvg/tflite_raspberry_pi/blob/main/object_detection/Train_custom_model_tutorial.ipynb
    ![img_1.png](img_1.png)
3. ### Sharpening and resizing 
    The detected images are resized to 140x1000 pixels.
    ![img_2.png](img_2.png)
4. ### Applying Adaptive threshold algorithm 
5. ### Contour searching on threshold image
    The aim is to find the coordinates of the individual numbers on the numberplate.
    On the basis of the found contours' coordinates, the 140x1000 px images are cut into 8 pieces 
    ![img_3.png](img_3.png)
6. ### Classify the image pieces
   ![img_4.png](img_4.png)
   ![img_5.png](img_5.png)
   ![img_6.png](img_6.png)
   A simple ad hoc tensorflow CNN classifies the images into 10 classes. Because of the similarity of the problem, for the model training the dataset was combined with MNIST dataet.


## Results

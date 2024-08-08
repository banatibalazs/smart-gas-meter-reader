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


You need to install the PubSubClient library. Here are the steps to do so:  
Open the Arduino IDE.
Go to Sketch -> Include Library -> Manage Libraries....
In the Library Manager, type PubSubClient in the search box.
Find the PubSubClient library by Nick O'Leary and click the Install button.

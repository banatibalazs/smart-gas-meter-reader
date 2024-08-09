#include "esp_camera.h"
#include <WiFi.h>
#include <PubSubClient.h>

#define CAMERA_MODEL_AI_THINKER         // Has PSRAM

#define uS_TO_S_FACTOR 1000000ULL  /* Conversion factor for micro seconds to seconds */
#define TIME_TO_SLEEP  20

#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27

#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

// WiFi
const char *ssid = "******";   // Enter your WiFi name
const char *password = "******";  // Enter WiFi password


// MQTT Broker
const char *mqtt_broker      = "192.168.0.33"; // Enter your MQTT broker IP
const char *pub_topic        = "esp_cam_1/from_esp";
const char *sub_topic_photo  = "esp_cam_1/photo";
const char *sub_topic_hello  = "esp_cam_1/hello";
const char *pub_topic_ready  = "esp_cam_1/ready";
const char *pub_topic_hello  = "esp_cam_1/hello_pub";
const char *sub_topic_sleep  = "esp_cam_1/sleep";

const char *mqtt_username = "******"; // Enter your MQTT username
const char *mqtt_password = "******"; // Enter your MQTT password
const int mqtt_port       =  1883;
const int MAX_PAYLOAD     =  60000;


// Flash
#define LED_BUILTIN 4
bool flash;

WiFiClient espClient;
PubSubClient client(espClient);

void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);
  Serial.setDebugOutput(true);
  Serial.println();

  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;
  config.pin_d0       = Y2_GPIO_NUM;
  config.pin_d1       = Y3_GPIO_NUM;
  config.pin_d2       = Y4_GPIO_NUM;
  config.pin_d3       = Y5_GPIO_NUM;
  config.pin_d4       = Y6_GPIO_NUM;
  config.pin_d5       = Y7_GPIO_NUM;
  config.pin_d6       = Y8_GPIO_NUM;
  config.pin_d7       = Y9_GPIO_NUM;
  config.pin_xclk     = XCLK_GPIO_NUM;
  config.pin_pclk     = PCLK_GPIO_NUM;
  config.pin_vsync    = VSYNC_GPIO_NUM;
  config.pin_href     = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn     = PWDN_GPIO_NUM;
  config.pin_reset    = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;

  // if PSRAM IC present, init with UXGA resolution and higher JPEG quality
  //                      for larger pre-allocated frame buffer.
  if (psramFound()) {
    config.frame_size   = FRAMESIZE_UXGA;
    config.jpeg_quality = 10;
    config.fb_count     = 2;
  } else {
    config.frame_size   = FRAMESIZE_SVGA;
    config.jpeg_quality = 12;
    config.fb_count     = 1;
  }

  flash = true;

#if defined(CAMERA_MODEL_ESP_EYE)
  pinMode(13, INPUT_PULLUP);
  pinMode(14, INPUT_PULLUP);
#endif

  // camera init
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x", err);
    return;
  }

  sensor_t * s = esp_camera_sensor_get();
  // initial sensors are flipped vertically and colors are a bit saturated
  if (s->id.PID == OV3660_PID) {
    s->set_vflip(s, 1); // flip it back
    s->set_brightness(s, 1); // up the brightness just a bit
    s->set_saturation(s, -2); // lower the saturation
  }
  // drop down frame size for higher initial frame rate
  s->set_framesize(s, FRAMESIZE_SVGA);

#if defined(CAMERA_MODEL_M5STACK_WIDE) || defined(CAMERA_MODEL_M5STACK_ESP32CAM)
  s->set_vflip(s, 1);
  s->set_hmirror(s, 1);
#endif


  
  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("");
  Serial.println("WiFi connected");

  //connecting to a mqtt broker
  client.setServer(mqtt_broker, mqtt_port);
  client.setBufferSize (MAX_PAYLOAD);
  client.setCallback(callback);
  reconnect();


  pinMode(LED_BUILTIN, OUTPUT);


  esp_sleep_enable_timer_wakeup(TIME_TO_SLEEP * uS_TO_S_FACTOR);
  Serial.println("Setup ready.");
  client.publish(pub_topic_ready, "Setup is ready.");

}


void callback(String topic, byte* message, unsigned int length) {
  String messageTemp;
  Serial.println(topic);
  for (int i = 0; i < length; i++) {
    messageTemp += (char)message[i];
  }
  if (topic == sub_topic_photo) {
    Serial.println("Take picture");
    client.publish(pub_topic_hello, sub_topic_photo);
    take_picture();
  }
  if (topic == sub_topic_sleep) {
    client.publish(pub_topic_hello, sub_topic_sleep);
    Serial.println("Go sleep.");
    try {
      int dur = messageTemp.toInt();
      Serial.println(dur);
      sleep(dur);
    }
    catch (...) {
      Serial.println("Catch block.");
      Serial.println(TIME_TO_SLEEP);
      sleep(TIME_TO_SLEEP);
    }
  }

  if (topic == sub_topic_hello) {
    client.publish(pub_topic_hello, "hello, i'm esp-1.");
    }

}

void sleep(int duration) {
  esp_sleep_enable_timer_wakeup(duration * uS_TO_S_FACTOR);
  Serial.println("Sleep.");
  gpio_deep_sleep_hold_en();
  esp_deep_sleep_start();
}

void take_picture() {
  camera_fb_t * fb = NULL;
  if (true) {
    digitalWrite(LED_BUILTIN, HIGH);
  };
  delay(100);
  Serial.println("Taking picture");
  fb = esp_camera_fb_get(); // used to get a single picture.
  if (!fb) {
    Serial.println("Camera capture failed");
    return;
  }
  Serial.println("Picture taken");
  delay(100);
  digitalWrite(LED_BUILTIN, LOW);
  sendMQTT(fb->buf, fb->len);
  esp_camera_fb_return(fb); // must be used to free the memory allocated by esp_camera_fb_get().
  client.publish(pub_topic_hello, "Picture has taken.");

}

void sendMQTT(const uint8_t * buf, uint32_t len) {
  Serial.println("Sending picture...");
  if (len > MAX_PAYLOAD) {
    Serial.print("Picture too large, increase the MAX_PAYLOAD value");
  } else {
    Serial.print("Picture sent ? : ");
    Serial.println(client.publish(pub_topic, buf, len, false));
  }


}

void set_flash() {
  flash = !flash;
  Serial.print("Setting flash to ");
  Serial.println (flash);
}


void reconnect() {
  while (!client.connected()) {
    String client_id = "esp_cam-client-";
    client_id += String(WiFi.macAddress());
    Serial.print("Attempting MQTT connection...");
    if (client.connect(client_id.c_str(), mqtt_username, mqtt_password)) {
      Serial.println("connected");
      client.subscribe(sub_topic_sleep);
      client.subscribe(sub_topic_photo);
      client.subscribe(sub_topic_hello);
    } else {
      Serial.print("failed, rc=");
      Serial.println(client.state());
      Serial.println(" try again in 5 seconds");
      delay(5000);
    }
  }
}

unsigned long prev_time = millis();

void loop() {

  if (!client.connected()) {
    reconnect();
  }
  client.loop();
}

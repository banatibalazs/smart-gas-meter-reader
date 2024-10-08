# Table of Contents:
- [Configuration of the MQTT broker](#Configuration-of-the-MQTT-broker)
  - [Install the `mosquitto` packages:](#Install-the-mosquitto-packages)
  - [To secure the MQTT broker, create a password file:](#To-secure-the-MQTT-broker-create-a-password-file)
  - [Accessing the MQTT broker over LAN](#Accessing-the-MQTT-broker-over-LAN)
- [Configure flask app](#Configure-flask-app)
- [Creating executable scripts with PyInstaller](#Creating-executable-scripts-with-PyInstaller)
- [Set up cron job to execute the script regularly](#Set-up-cron-job-to-execute-the-script-regularly)



## Configuration of the MQTT broker

### Install the `mosquitto` packages:

```bash
sudo apt update
sudo apt install mosquitto mosquitto-clients
```

Start the `mosquitto` service:

```bash
sudo systemctl start mosquitto
```

Check the status of the `mosquitto` service:

```bash
sudo systemctl status mosquitto
```

Enable the `mosquitto` service to start on boot:

```bash
sudo systemctl enable mosquitto
```

The `mosquitto` service is now running on the default port `1883`.

To test the MQTT broker, open two terminal windows. In the first terminal, subscribe to the topic `test`:

```bash
mosquitto_sub -h localhost -t test
```

In the second terminal, publish a message to the topic `test`:

```bash
mosquitto_pub -h localhost -t test -m "Hello, MQTT!"
```

The message "Hello, MQTT!" should appear in the first terminal.

### To secure the MQTT broker, create a password file:

```bash
sudo mosquitto_passwd -c /etc/mosquitto/passwd <username>
```

Enter a password for the user. Add the following lines to the `/etc/mosquitto/mosquitto.conf` file:

```bash
allow_anonymous false
password_file /etc/mosquitto/passwd
```

Restart the `mosquitto` service:

```bash
sudo systemctl restart mosquitto
```

To test the secure MQTT broker, use the following commands:

```bash
mosquitto_sub -h localhost -t test -u <username> -P <password>
```

```bash
mosquitto_pub -h localhost -t test -m "Hello, Secure MQTT!" -u <username> -P <password>
```

The message "Hello, Secure MQTT!" should appear in the first terminal.

The MQTT broker is now secure and ready to use.


### Accessing the MQTT broker over LAN
If you want to use the MQTT broker over LAN, you need to set a static IP address and open the port `1883` in the firewall.

To set a static IP address, edit the `/etc/dhcpcd.conf` file:

Set the firewall rules:

```bash
sudo ufw allow 1883
```

Reload the firewall:

```bash
sudo ufw reload
```

The MQTT broker is now accessible over LAN.

## Configure flask app

- use gunicorn to run the flask app
- use nginx as a reverse proxy server

## Creating executable file with PyInstaller

- Install the required packages. The script was tested with
  Python 3.9.12
  
  ```commandline
  pip install -r requirements.txt
  ```

- Modify the `mqtt_and_analyze.py` script.
  - Alter the broker address and port if necessary.
- Install PyInstaller:

  ```bash
  pip install pyinstaller
  ```
- Create an executable file:

  ```bash
  pyinstaller --onefile mqtt_and_analyze.py

- The executable file is located in the `dist` folder.
- Copy the executable file to the server.

## Set up cron job

- Run the executable file regularly to process the images.
- Copy the processed images into flask's static/images folder.

I used https://crontab.guru/ to find the proper schedule expression.





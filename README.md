## Ultra Fast Face Detector

### Install
```
curl https://github.com/raspberrypisig/ultrafastfacedetector/raw/master/install.sh | bash -
```

### Run
```
cd ultrafastfacedetector
python3 main.py
```


### Install OpenCV

Installed OpenCV version 4.6.0 on Raspberry Pi 4 (Raspbian 64-bit OS) using instructions here 
https://qengineering.eu/install-opencv-4.5-on-raspberry-64-os.html

The actual script is hosted here.

https://github.com/Qengineering/Install-OpenCV-Raspberry-Pi-64-bits

NOTES:

   - before install, needed to increase swap size mem+swap needs to be about 7GB. 
   - takes a long time to install (did it overnight)

### picamera2,
     ``` 
     sudo apt install -y python3-picamera2
     sudo apt install -y libcamera-dev
     ```

### ultra fast face detector

https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB

### Extra Notes

1. Pi camera ribbon cable (blue tag  of ribbon faces USB/ethternet ports).




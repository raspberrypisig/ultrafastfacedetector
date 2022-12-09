#!/usr/bin/env python3

from picamera2 import Picamera2
from flask import Flask, Response, redirect, url_for
from libcamera import Transform
from ultrafacedetector import UltraFaceDetector
from string import Template

app = Flask(__name__)
camera = Picamera2()
face_detector = UltraFaceDetector(camera)

#camera_image_size = (640,480)
#camera_image_size = (320,240)
camera_image_size = (1280,960)

webpage_image_size = (640,480)

PAGE = """\
<html>
<head>
<title>picamera2 MJPEG streaming demo</title>
</head>
<body>
<h1>Picamera2 MJPEG Streaming Demo</h1>
<img src="/video_feed" width="$width" height="$height" />
</body>
</html>
"""

#def genOpenCVFrames():
#    with picamera2.Picamera2() as camera:
#        camera.configure(camera.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}, transform=Transform(hflip=True)))
#        camera.start()
#        while True:
#            im = camera.capture_array()
#            ret, frame = cv2.imencode(".jpg", im)
#            yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame.tobytes() + b'\r\n')        
            

@app.route('/index')
def index():
    html = Template(PAGE)
    width, height = webpage_image_size
    return Response(html.substitute(width=width, height=height))

@app.route('/video_feed')
def video_feed():
    #return Response(genFrames(),mimetype='multipart/x-mixed-replace; boundary=frame')
    #return Response(genOpenCVFrames(),mimetype='multipart/x-mixed-replace; boundary=frame')
    return Response(face_detector.detect_faces(),mimetype='multipart/x-mixed-replace; boundary=frame')
    

@app.route('/')
def root():
    return redirect(url_for('index'))

if __name__ == "__main__":

    
    #camera.preview_configuration.controls.FrameRate = 30.0
    #camera.preview_configuration.controls
    #fps = 1000000/ FrameRateDuration  

    camera.configure(camera.create_preview_configuration(main={"format": 'XRGB8888', "size": camera_image_size}, transform=Transform(hflip=True)))
    camera.start()
    app.run(host='0.0.0.0', threaded=True)
    
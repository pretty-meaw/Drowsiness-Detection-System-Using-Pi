# Drowsiness Detection System

This project implements a drowsiness detection system using a Raspberry Pi, Pi Camera Module, and OpenCV. The system uses facial landmarks to detect if the user's eyes are closed for a prolonged period, indicating drowsiness. If drowsiness is detected, an alert is displayed on a web interface accessible through a browser on the local network.

## Prerequisites

- Raspberry Pi (Model 3B+ or later)
- Raspberry Pi Camera Module
- Internet connection for the Raspberry Pi
- Virtual environment for Python (optional but recommended)

## Step-by-Step Guide

### 1. Set Up the Raspberry Pi

1. Connect the Raspberry Pi Camera Module to the Raspberry Pi.
2. Enable the camera interface by running `sudo raspi-config`, navigating to `Interfacing Options` > `Camera`, and selecting `Enable`.
3. Update the system:
    ```bash
    sudo apt-get update
    sudo apt-get upgrade
    ```

### 2. Install Required Libraries

#### Create a Virtual Environment (Recommended)

1. Install `virtualenv` if we don't have it:
    ```bash
    sudo apt-get install python3-venv
    ```

2. Create a virtual environment:
    ```bash
    python3 -m venv myenv
    ```

3. Activate the virtual environment:
    ```bash
    source myenv/bin/activate
    ```

#### Install Python Libraries

1. Install required Python packages:
    ```bash
    pip install Flask opencv-python imutils dlib picamera[array]
    ```

### 3. Download the Shape Predictor Model

1. Download the `shape_predictor_68_face_landmarks.dat` file from [dlib's model zoo](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2).
2. Extract the file and place it in a directory named `models` in our project directory:
    ```bash
    mkdir models
    mv shape_predictor_68_face_landmarks.dat models/
    ```

### 4. Create the Flask Application

Create a file named `app.py` with the following content:

```python
from flask import Flask, render_template, Response
import cv2
import imutils
from scipy.spatial import distance
from imutils import face_utils
import dlib
from picamera.array import PiRGBArray
from picamera import PiCamera
import time

app = Flask(__name__)

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

thresh = 0.25
frame_check = 20
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Initialize the PiCamera
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))

# Allow the camera to warm up
time.sleep(0.1)

def generate_frames():
    flag = 0
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        image = frame.array

        frame = imutils.resize(image, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        subjects = detect(gray, 0)

        for subject in subjects:
            shape = predict(gray, subject)
            shape = face_utils.shape_to_np(shape)
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            if ear < thresh:
                flag += 1
                print(flag)
                if flag >= frame_check:
                    cv2.putText(frame, "**ALERT!**", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(frame, "**ALERT!**", (10, 325), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                flag = 0

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        rawCapture.truncate(0)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
```

#### 5. Create the HTML Template

Create a folder named templates in the same directory as our app.py script. Inside this folder, create a file named index.html with the following content:

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Drowsiness Detection</title>
</head>
<body>
    <h1>Drowsiness Detection</h1>
    <img src="{{ url_for('video_feed') }}">
</body>
</html>
```

#### 6. Run the Flask Application

Activate our virtual environment if it's not already activated:

```bash
source myenv/bin/activate
```

#### 6. Run the Flask application:

```bash
python app.py
```

#### 7. Access the Web Interface

Open a web browser on any device connected to the same local network as the Raspberry Pi and navigate to:

```url
http://<raspberry_pi_ip_address>:5000
```

Replace <raspberry_pi_ip_address> with the actual IP address of our Raspberry Pi. we should see the live video feed with drowsiness detection alerts.

#### Conclusion

We've now set up a drowsiness detection system on our Raspberry Pi, using a Pi Camera Module and OpenCV, and made it accessible through a web browser on our local network. This project can be further enhanced by adding features such as sound alerts or data logging.

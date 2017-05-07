import argparse
import base64
import skimage.transform as sktransform
import numpy as np
import socketio
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO
import os

from keras.models import model_from_json
import warnings

# Fix error with Keras and TensorFlow
import tensorflow as tf
tf.python.control_flow_ops = tf

def preprocess(image, top_offset=.4, bottom_offset=.12,left_offset = 0.1, right_offset = 0.05):
    """
    Applies preprocessing pipeline to an image: crops `top_offset` and `bottom_offset`
    portions of image, resizes to 32x128 px and scales pixel values to [0, 1].
    """
    top = int(top_offset * image.shape[0])
    bottom = int(bottom_offset * image.shape[0])
    #right = int(right_offset * image.shape[1])
    #left = int(left_offset * image.shape[1])
    image = sktransform.resize(image[top:-bottom, :], (32, 128, 3))
    return image

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None


@sio.on('telemetry')
def telemetry(sid, data):
    # The current steering angle of the car
    steering_angle = data["steering_angle"]
    # The current throttle of the car
    throttle = data["throttle"]
    # The current speed of the car
    speed = data["speed"]

    #brake = data['brake']
    

    # The current image from the center camera of the car
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    image_array = preprocess(np.asarray(image))
    transformed_image_array = image_array[None, :, :, :]
    steering_angle = 2 * float(model.predict(transformed_image_array, batch_size=1))
    if float(speed) < 5 :
    	throttle = 1
    elif (float(speed) < 10) and (float(speed) > 5) :
    	throttle = 0.1
    elif (float(speed) >10 ) and (float(speed) <12) :
    	throttle = 0
    elif float(speed) > 14 :
    	throttle = -0.2
    print(steering_angle, throttle, speed)
    send_control(steering_angle, throttle)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit("steer", data={
        'steering_angle': steering_angle.__str__(),
        'throttle': throttle.__str__()
    }, skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
                        help='Path to model definition json. Model weights should be on the same path.')
    args = parser.parse_args()
    with open(args.model, 'r') as jfile:
        model = model_from_json(jfile.read())

    model.compile("adam", "mse")
    weights_file = args.model.replace('json', 'h5')
    model.load_weights(weights_file)

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)

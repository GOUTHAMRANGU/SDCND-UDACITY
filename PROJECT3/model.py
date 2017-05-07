# Importing required python packages 
import numpy as np
import tensorflow as tf
import h5py
import os
import pandas as pd
import json
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math 
import random
import skimage
from numpy.random import random
from keras import models, optimizers, backend
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda, ELU
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.callbacks import Callback
from sklearn.model_selection import train_test_split
tf.python.control_flow_ops = tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import h5py
import shutil
from pathlib import Path
outdir = 'E:\TERM 1\solutions\PROJECT3\modelsol'

# get path to data and print number of image frames from train set.
csv_path ='E:\TERM 1\playground\project3\Solution\data\driving_log.csv'
data = pd.read_csv(csv_path, header=None, names =['center','left','right','steer','throttle','break','speed'], index_col = False)
fileModelJSON = 'model.json'
fileWeights = 'model.h5'

# read image from the file path
def read_image(index):
    image= cv2.imread(index)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image 
# normalize and center ot zero mean image
def norm_mean_image(image):
    image = (image/255.) -(0.5)
    return image

# only consider data above certain speed
from sklearn import model_selection
data = data[data.speed>5]
#df_train, df_valid = model_selection.train_test_split(data, test_size=.2)
#print(len(df_train))
line_d = data.iloc[[2]].reset_index()

# CLAHE for adaptive histogram equilization
def clahe(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=0.8, tileGridSize=(6,6))    
    transform = clahe.apply(v)
    res= cv2.merge((h,s,transform))
    res = cv2.cvtColor(res,cv2.COLOR_HSV2BGR)
    return res

# crop the image to right proportion and scale
# remove unwanted part of the image that is trees and resize to 66x200x3 for using nvidia model
#chopping 1/3 from top and 25 pixels from bottom
def cut_reshape_image(image):
    n_row,n_col,n_channel = 200,66,3
    shape = image.shape
    image = image[int(shape[0]/3):shape[0]-25, 0:shape[1]]
    image = cv2.resize(image,(n_row,n_col), interpolation=cv2.INTER_AREA)
    return image    

# Add left right camera images to train data with angle offset.
def lrimage(index):
    cameras = ['left', 'center', 'right']
    cameras_steering_correction = [.25, 0., -.25]
    camera = np.random.randint(len(cameras))
    image = cv2.imread(data[cameras[camera]][index])
    angle = data.steer[index]+cameras_steering_correction[camera]
    return image, angle

def shadow(image):
    h, w = image.shape[0], image.shape[1]
    [x1, x2] = np.random.choice(w, 2, replace=False)
    k = h / (x2 - x1)
    b = - k * x1
    for i in range(h):
        c = int((i - b) / k)
        image[i, :c, :] = (image[i, :c, :] * .5).astype(np.int32)
    return image

def gamma_crtn(image):
    gamma = np.random.uniform(0.4, 1.5)
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def shear_img(image, steering_angle, shear_range=200):
    rows, cols, ch = image.shape
    dx = np.random.randint(-shear_range, shear_range + 1)
    random_point = [cols / 2 + dx, rows / 2]
    pts1 = np.float32([[0, rows], [cols, rows], [cols / 2, rows / 2]])
    pts2 = np.float32([[0, rows], [cols, rows], random_point])
    dsteering = dx / (rows / 2) * 360 / (2 * np.pi * 25.0) / 6.0
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, (cols, rows), borderMode=1)
    steering_angle += dsteering
    return image, steering_angle
# Horizontally flipping the images and negating the steering angles
def flip_index(index):
    cameras = ['left', 'center', 'right']
    cameras_steering_correction = [.25, 0., -.25]
    camera = np.random.randint(len(cameras))
    image = read_image(data[cameras[camera]][index])
    image = cv2.flip(image,1)
    steer = -data.steer(index)
    return image, steer


 def pre_process_train(line):
    i_lrc = np.random.randint(3)
    if (i_lrc == 0):
        path_file = line_d['left'][0].strip()
        shift_ang = .25
    if (i_lrc == 1):
        path_file = line_d['center'][0].strip()
        shift_ang = 0.
    if (i_lrc == 2):
        path_file = line_d['right'][0].strip()
        shift_ang = -.25
    steer = line_d['steer'][0] + shift_ang
    image = cv2.imread(path_file)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image = clahe(image)
    image = cut_reshape_image(image)
    image = shadow(image)
    image = gamma_crtn(image)
    image = np.array(image)
    pre = np.random.randint(10)
    if pre > 5:
        image = cv2.flip(image,1)
        steer = -steer
    if pre in [1,3,5,7,9]:
    	image,steer  =  shear_img(image)
            
    return image, steer    
def preprocess_image_predict(line):
    path_file = line_d['center'][0].strip()
    image = cv2.imread(path_file)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image = cut_reshape_image(image)
    image = np.array(image)
    return image

new_row_size,new_col_size = 66,200
thres_prob = 1
def generator_train(data,batch_size=32):
    b_img = np.zeros((batch_size, new_row_size, new_col_size, 3))
    b_steer = np.zeros(batch_size)
    while 1:
        for i_batch in range(batch_size):
            i_line = np.random.randint(len(data))
            line_d = data.iloc[[i_line]].reset_index()
            
            keep_prob = 0
            while keep_prob == 0:
                x,y = pre_process_train(line_d)
                un_prob = np.random
                if abs(y)<.15:
                    pr_val = np.random.uniform()
                    if pr_val>thres_prob:
                        keep_prob = 1
                else:
                    keep_prob = 1
            b_img[i_batch] = x
            b_steer[i_batch] = y
        yield b_img, b_steer
        
def generator_valid(data):
    while 1:
        for i_line in range(len(data)):
            line_d = data.iloc[[i_line]].reset_index()
            x = preprocess_image_predict(data)
            x = x.reshape(1, x.shape[0], x.shape[1], x.shape[2])
            y = line_d['steer'][0]
            y = np.array([[y]])
            yield x, y

        
val_gen = generator_valid(data)

input_shape = (66,200,3)
pool_size = (2,3)
filter_size= 3
model = Sequential()
model.add(MaxPooling2D(pool_size = pool_size, input_shape = input_shape))
model.add(Lambda(lambda x: x/127.5 - 1.))
model.add(Convolution2D(5, 5, 24, subsample=(4, 4), border_mode="same", name = 'conv0'))
model.add(ELU())
model.add(Convolution2D(5, 5, 36, subsample=(2, 2), border_mode="same", name = 'conv1'))
model.add(ELU())
model.add(Convolution2D(5, 5, 48, subsample=(2, 2), border_mode="same", name = 'conv2'))
model.add(ELU())
model.add(Convolution2D(3, 3, 64, subsample=(2, 2), border_mode="same", name = 'conv3'))
model.add(ELU())
model.add(Convolution2D(3, 3, 64, subsample=(2, 2), border_mode="same", name = 'conv4'))
model.add(Flatten())
model.add(Dropout(.2))
model.add(ELU())
model.add(Dense(1164,  name = 'hidden1'))
model.add(Dropout(.2))
model.add(ELU())
model.add(Dense(100, name = 'hidden2'))
model.add(Dropout(.2))
model.add(ELU())
model.add(Dense(50, name = 'hidden3'))
model.add(Dropout(.2))
model.add(ELU())
model.add(Dense(10))
model.add(Dropout(.2))
model.add(ELU())
model.add(Dense(1, name = 'output'))
adam = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer=adam, loss="mse")
model.summary()

#function to save the model.
def save_model(fileModelJSON,fileWeights):
    #print("Saving model to disk: ",fileModelJSON,"and",fileWeights)
    if Path(fileModelJSON).is_file():
        os.remove(fileModelJSON)
    json_string = model.to_json()
    with open(fileModelJSON,'w' ) as f:
        json.dump(json_string, f)
    if Path(fileWeights).is_file():
        os.remove(fileWeights)
    model.save_weights(fileWeights)

#function to keep trace of all the weights.
class WeightsLogger(Callback):
    def __init__(self, root_path):
        super(WeightsLogger, self).__init__()
        self.weights_root_path = os.path.join(out_dir, 'weights/')
        shutil.rmtree(self.weights_root_path, ignore_errors=True)
        os.makedirs(self.weights_root_path, exist_ok=True)
    def on_epoch_end(self, epoch, logs={}):
        self.model.save_weights(os.path.join(self.weights_root_path, 'model_epoch_{}.h5'.format(epoch + 1)))


#final train
val_s=len(data)
pr_threshold = 1
batch_size = 256
i_best = 0
val_best = 1000
for i_pr in range(5):
    train_gen = generator_train(data,batch_size)
    history = model.fit_generator(train_gen, samples_per_epoch = 5120, nb_epoch = 1, validation_data = val_gen , nb_val_samples = val_s)
    with open(os.path.join(outdir, 'model.json'), 'w') as file:
        file.write(model.to_json())
    val_loss = history.history['val_loss'][0]
    if val_loss < val_best:
        i_best = i_pr
        val_best = val_loss
        fileModelJSON = 'model_best.json'
        fileWeights = 'model_best.h5'
        save_model(fileModelJSON,fileWeights)


    pr_threshold = 1/(i_pr+1)
print('Best model found at iteration # ' + str(i_best))
print('Best Validation score : ' + str(np.round(val_best,4)))








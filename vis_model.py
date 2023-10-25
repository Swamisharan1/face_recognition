import streamlit as st
import tensorflow as tf 
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
from PIL import Image
from mtcnn.mtcnn import MTCNN
import requests
import io
import gdown

gdrive_url = 'https://www.dropbox.com/scl/fi/h5tj3vkmou9brdaqfwsln/img_proc.zip?rlkey=ki66e4lgvt8tgof6hf96u4ijp&dl=1'

# download the dataset from google drive
@st.cache_data
def download_data():
    gdown.download(gdrive_url, 'img_proc.zip', quiet=False)

download_data()

# Extract the downloaded dataset
import zipfile

with zipfile.ZipFile('img_proc.zip', 'r') as zip_ref:
    zip_ref.extractall('dataset')

train = ImageDataGenerator(rescale=1/255)
validation = ImageDataGenerator(rescale=1/255)

# Define the path to the training and validation datasets within the extracted repository
train_data_dir = 'dataset/img_proc/train'
validation_data_dir = 'dataset/img_proc/validation'

train_datast = train.flow_from_directory(train_data_dir,
                                        target_size=(200, 200),
                                        batch_size=3,
                                        class_mode='binary')

validation_datast = train.flow_from_directory(validation_data_dir,
                                             target_size=(200, 200),
                                             batch_size=3,
                                             class_mode='binary')


model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(16,(3,3),activation='relu',input_shape =(200,200,3)),
                                   tf.keras.layers.MaxPool2D(2,2),
                                   #
                                   tf.keras.layers.Conv2D(32,(3,3),activation = 'relu'),
                                   tf.keras.layers.MaxPool2D(2,2),
                                   ##
                                   tf.keras.layers.Conv2D(64,(3,3),activation = 'relu'),
                                   tf.keras.layers.MaxPool2D(2,2),
                                   #
                                   tf.keras.layers.Flatten(),
                                   ##
                                   tf.keras.layers.Dense(512,activation = 'relu'),
                                   ##
                                   tf.keras.layers.Dense(1,activation = 'sigmoid')])

model.compile(loss='binary_crossentropy',
             optimizer = RMSprop(lr=0.001),
             metrics = ['accuracy'])


model.fit = model.fit(train_datast,
                     epochs = 10,
                     validation_data = validation_datast)
# Display the camera input widget
img_file_buffer = st.camera_input("Take a picture")

# check if an image was captured
if img_file_buffer is not None:
    # convert the file-like object to PIL image
    img = Image.open(img_file_buffer)
    img_array = np.array(img)

    def return_face(detector, img_array):
        face = detector.detect_faces(img_array)
        if len(face) != 1:
            print("Error: found " + str(len(face)) + " faces")
            return []
        face_coords = face[0]['box']
        x1, y1, width, height = face_coords[0], face_coords[1], face_coords[2], face_coords[3]
        x2, y2 = x1 + width, y1 + height
        # Ensure coordinates are within image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_array.shape[1] - 1, x2), min(img_array.shape[0] - 1, y2)
        extracted_face = img_array[y1:y2,x1:x2]
        return extracted_face

detector = MTCNN()
extracted_face = return_face(detector, img_array)
if len(extracted_face) > 0:  # If a face was found
    resized_face = cv2.resize(extracted_face, (200, 200))
    x = image.img_to_array(resized_face)
    x = x / 255.0
    x = np.expand_dims(x,axis = 0)
    val = model.predict(x)

if val == 1:
    st.write('swami')
else:
    st.write('not swami')
    
if val != 1:
    from cvzone.HandTrackingModule import HandDetector 
    import cv2
    import os

    detector = HandDetector()

    def name():
        print('''Displaying information:
                Name:- Swamisharan Kumawat''')

    def add():
        print('''Displaying information:
                Address:-Sant Tukaram Nagar''')

    def clas():
        print('''Displaying information:
                Class:- M.Sc. Data Science''')

    def college():
        print('''Displaying information:
                College:- DPU ACS College''')

    def project():
        print('''Displaying information:
                Project Name:- FaceGesture Identity Access''')
    def project_sum():
        print('''Displaying information:
                Description: "FaceGesture Identity Access" is a straightforward and inclusive project that
                integrates facial recognition and hand gestures to provide secure identity verification
                and efficient access to personalized information. This name explicitly conveys the core features
                of the project in a common and accessible manner, making it clear that it's all about identity
                and gesture-based access.''')
  
    


    handphoto = detector.findHands(img, draw = False)

    if handphoto:
        a  = len(handphoto[0])
        def left_right():
            fsum_l = 0
            fsum_r = 0
            for i in range(a):
                if handphoto[0][i]['type']=='Right':
                    lmlist  = handphoto[0][i]
                    fingerstatus_right = detector.fingersUp(lmlist)
                    for i in range(len(fingerstatus_right)):
                        fsum_r = fsum_r+fingerstatus_right[i]
                    break
            for i in range(a):
                if handphoto[0][i]['type']=='Left':
                    lmlist1 = handphoto[0][i]
                    fingerstatus_left = detector.fingersUp(lmlist1)
                    for i in range(len(fingerstatus_left)):
                        fsum_l = fsum_l+fingerstatus_left[i]
                    break
            fsum = fsum_l+fsum_r
            return fsum
        
        if left_right() == 0:
            print('no finger is up')
        elif left_right() == 1:
            name()
        elif left_right() == 2:
            add()
        elif left_right() == 3:
            clas()
        elif left_right() == 4:
            college()
        elif left_right() == 5:
            project()
        elif left_right() ==6:
            project_sum()
        else:
            print('Gesture can not be recognized')
            
    


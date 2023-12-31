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

import requests
import os
import zipfile
from tensorflow.keras.models import load_model

#model 1
# URL of the model file
url = 'https://www.dropbox.com/scl/fi/lllbvh8ql8mmmmhcxasoi/trained_model.zip?rlkey=cjzvolr7jcqhupi5swyurpoj9&dl=1'

# Send a HTTP request to the URL of the file, stream = True means the file will be downloaded as a stream
r = requests.get(url, stream = True)

# Check if the request is successful
if r.status_code == 200:
    # Download the file by chunk
    with open('trained_model.zip', 'wb') as f:
        for chunk in r.iter_content(chunk_size = 1024):
            if chunk:
                f.write(chunk)
else:
    print('Failed to download the model.')

# Extract the downloaded zip file
with zipfile.ZipFile('trained_model.zip', 'r') as zip_ref:
    zip_ref.extractall('model')

# Now you can load your model
model = load_model('model/trained_model')  # replace 'model/trained_model.h5' with the actual path of your .h5 file in the extracted folder

#model 2
# URL of the model file
url = 'https://www.dropbox.com/scl/fi/9a3cvir8ns0zlurrtmvmz/trained_model2.zip?rlkey=4pfyra1pjmgb50ktrtssldpjn&dl=1'

# Send a HTTP request to the URL of the file, stream = True means the file will be downloaded as a stream
r = requests.get(url, stream = True)

# Check if the request is successful
if r.status_code == 200:
    # Download the file by chunk
    with open('trained_model2.zip', 'wb') as f:
        for chunk in r.iter_content(chunk_size = 1024):
            if chunk:
                f.write(chunk)
else:
    print('Failed to download the model.')

# Extract the downloaded zip file
with zipfile.ZipFile('trained_model2.zip', 'r') as zip_ref:
    zip_ref.extractall('model1')

# Now you can load your model
model1 = load_model('model1/trained_model2')

# Display the camera input widget
img_file_buffer = st.camera_input("Take a picture")

# check if an image was captured
if img_file_buffer is not None:
    # convert the file-like object to PIL image
    img = Image.open(img_file_buffer)
    img_array = np.array(img)  # Convert PIL Image to numpy array
    #img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    
    #img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
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
    if len(extracted_face) > 0:
    # Resize and preprocess face
        resized_face = cv2.resize(extracted_face, (200, 200))
        x = image.img_to_array(resized_face)
        x = x / 255.0
        x = np.expand_dims(x,axis = 0)
        images = np.vstack([x])
        X_test = images
        # Generate predictions from each model
        predictions_model1 = model.predict(X_test)
        predictions_model2 = model1.predict(X_test)
        #predictions_model3 = new_model3.predict(X_test)

        # Combine predictions using averaging
        ensemble_predictions = (predictions_model1 + predictions_model2 ) / 2

        # Make binary classification predictions based on a threshold (e.g., 0.5)
        binary_predictions = (ensemble_predictions >= 0.5).astype(int)


    if binary_predictions == 1:
        st.write('swami')
    else:
        st.write('not swami')
    
    if binary_predictions == 1:
        from cvzone.HandTrackingModule import HandDetector 
        import cv2
        import os

        # Capture an image
        img_file_buffer1 = st.camera_input("Take Hand's picture", key="unique_key_1")

        # Check if an image was captured
        if img_file_buffer1 is not None:
            # Convert the file-like object to PIL image
            img1 = Image.open(img_file_buffer1)
            img_array1 = np.array(img1)  # Convert PIL Image to numpy array

            # Check if img_array1 contains valid image data
            if img_array1.size != 0:
            # Convert the image to BGR color space
                img_bgr1 = cv2.cvtColor(img_array1, cv2.COLOR_RGB2BGR)
            else:
                st.write("No valid image data in img_array1")
        else:
            st.write("No image captured")

            #img_bgr1 = cv2.cvtColor(img_array1, cv2.COLOR_RGB2BGR)

        detector = HandDetector()

        def name():
            st.write('One Finger Up')
            st.write('''Displaying information:
                     Name:- Swamisharan Kumawat''')

        def add():
            st.write('Two Fingers Up')
            st.write('''Displaying information:
                    Address:-Sant Tukaram Nagar''')

        def clas():
            st.write('Three Fingers Up')
            st.write('''Displaying information:
                    Class:- M.Sc. Data Science''')

        def college():
            st.write('Six Fingers Up')
            st.write('''Displaying information:
                    College:- DPU ACS College''')

        def project():
            st.write('Four Fingers Up')
            st.write('''Displaying information:
                    Project Name:- FaceGesture Identity Access''')
        def project_sum():
            st.write('Five Fingers Up')
            st.write('''Displaying information:
                    Description: "FaceGesture Identity Access" is a straightforward and inclusive project that
                    integrates facial recognition and hand gestures to provide secure identity verification
                    and efficient access to personalized information. This name explicitly conveys the core features
                    of the project in a common and accessible manner, making it clear that it's all about identity
                    and gesture-based access.''')
  
    


        handphoto = detector.findHands(img_bgr1, draw = False)

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
                st.write('no finger is up')
            elif left_right() == 1:
                name()
            elif left_right() == 2:
                add()
            elif left_right() == 3:
                clas()
            elif left_right() == 6:
                college()
            elif left_right() == 4:
                project()
            elif left_right() ==5:
                project_sum()
            else:
                print('Gesture can not be recognized')
            
    



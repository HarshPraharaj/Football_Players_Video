#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import sys
import numpy as np
import pandas as pd
import matplotlib as plt

import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# In[6]:



#%% load the data, go over training images and store them in a list
maleFaceFiles   = glob.glob('/Users/harshpraharaj/Desktop/Football_Player_Stats/complete-fifa-2017-player-dataset-global/Pictures/*.png')
femaleFaceFiles = glob.glob('/Users/harshpraharaj/Desktop/Football_Player_Stats/complete-fifa-2017-player-dataset-global/Pictures_f/*.png')
faceFiles = maleFaceFiles + femaleFaceFiles


listOfPlayerNames= []

listOfImages = []
for imageFilename in faceFiles:
    currName = imageFilename.split("/")[-1].split('.')[0]
        
    try:
        currImage = mpimg.imread(imageFilename)
        if len(np.unique(currImage[:,:,0].ravel())) <= 40:
            print("no image for '" + currName + "'")
        else:
            listOfPlayerNames.append(currName)
            listOfImages.append(currImage)
    except:
        print("didn't load '" + currName + "'")
        
femaleNames = [x.split("/")[-1].split('.')[0] for x in femaleFaceFiles]
isFemale    = [x in femaleNames for x in listOfPlayerNames]

print('Total number of loaded face images is %d' %(len(listOfImages)))



# In[22]:


import face_recognition
import os
import cv2

known_faces = []

'''
ramos_image = face_recognition.load_image_file("/Users/harshpraharaj/Desktop/Football_Player_Stats/complete-fifa-2017-player-dataset-global/Pictures/Pictures/Sergio Ramos.png")
ramos_face_encoding = face_recognition.face_encodings(ramos_image)[0]

bale_image = face_recognition.load_image_file("/Users/harshpraharaj/Desktop/Football_Player_Stats/complete-fifa-2017-player-dataset-global/Pictures/Pictures/Eden Hazard.png")
bale_face_encoding = face_recognition.face_encodings(bale_image)[0]
'''

known_faces = []

# In[24]:


for image in faceFiles:
    image_i = face_recognition.load_image_file(image)
    #print(image)
    known_faces.append(face_recognition.face_encodings(image_i)[0])

#print(listOfPlayerNames)
# In[28]:


face_locations = []
face_encodings = []
face_names = []
frame_number = 0

input_movie = cv2.VideoCapture("ip.mp4")
length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))

# Create an output movie file (make sure resolution/frame rate matches input video!)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = input_movie.get(cv2.CAP_PROP_FPS)
print(fps)
output_movie = cv2.VideoWriter('output1.avi', fourcc,29.97, (1280, 720))


# In[ ]:


while True:
    # Grab a single frame of video
    ret, frame = input_movie.read()
    frame_number += 1

    # Quit when the input video file ends
    if not ret:
        break

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:

        match = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.50)
        name = None


        for y in range(0,len(match)):
            print(match[y])
            if match[y]:
                face_names.append(listOfPlayerNames[y])
    print(face_names)
    # Label the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        if not name:
            continue
        print(name,top,right,bottom,left)
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)


    # Write the resulting image to the output video file
    print("Writing frame {} / {}".format(frame_number, length))
    output_movie.write(frame)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

input_movie.release()
output_movie.release()
cv2.destroyAllWindows()


# In[ ]:





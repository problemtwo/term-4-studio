# Made with help from https://www.hackster.io/mjrobot/real-time-face-recognition-an-end-to-end-project-a10826

import cv2 # OpenCV2 (Open Source Computer Vision 2) is a library that helps with
           # displaying graphics, image manipulation, and (in this case) facial recognition

import math # The python math library, used for the floor function
import numpy as np # (Sorry Nikhil :) ) We are using numpy because opencv uses numpy arrays,
                   # and we need to convert the list of ids (in the train function) to a numpy array.
import os # The os module is used to string together paths to find the images and the ids.
import sys # The sys module is used solely for argv, which provides user input to the program
           # via a terminal / command prompt.
from PIL import Image # PIL Images are used to load the images to train the cascade. OpenCV will save
                      # the each frame in which it detects a face into a directory of our choice, so
                      # we use PIL to read the images.

# The `detect` function opens up the images from the data directory, and uses their names to create 
# the ids for those images. WARNING: the lists `detected` and `ids` must be the same length (which
# means that the cascade must only detect one face), or else OpenCV will throw an exception.
def detect(cascade):
    path = 'data'
    images = [Image.open(os.path.join(path,filename)).convert('L') for filename in os.listdir(path)]
    ids = [filename.split('_')[1] for filename in os.listdir(path)]
    detected = []
    for iden,img in zip(ids,images):
        np_img = np.array(img,'uint8')
        faces = cascade.detectMultiScale(np_img)
        for (x,y,w,h) in faces:
            detected += [np_img[y:y+h,x:x+w]]
    return detected,ids


# The `train` function will take the faces that were detected and their ids (provided by the `detect`
# function), and will train an LBPH (Local Binary Patterns Histograms) Face Recognizer to detect each
# particular type of face. The LBPH Recognizer can save into a Yaml file, which is convenient if the
# program is stopped, because it can start again without losing any progress. Also, after the cascade
# has finished training, it can train again because the progress is saved.
def train(cascade):
    rec = cv2.face.LBPHFaceRecognizer_create()
    detected,ids = detect(cascade)
    rec.train(detected,np.array([int(i) for i in ids]))
    rec.write('trainer.yml')

# The `predict` function will read the data from the Yaml file, and create a Local Binary Patterns
# Histograms Face Recognizer from that Yaml file. It returns a name (or Unknown / Error if something
# unexpected happened), and a confidence which is a percentage out of 100.
def predict(part_of_image):
    rec = cv2.face.LBPHFaceRecognizer_create()
    rec.read('trainer.yml')
    names = ['Error','Abhi','Sohm']
    iden,conf = rec.predict(part_of_image)
    if conf < 100:
        return names[iden],conf
    return 'Unknown',conf

# The `start` function projects a grayscale version of the camera's video. Depending on the flags passed
# into the program, this function might either prompt the user about the user's identity (which is a number
# from 1 to infinity), or it might display text conveying the cascade's prediction about the user's indentity,
# and how confident it is about it's prediction.
def start(pred=False):
    camera = cv2.VideoCapture(0)
    camera.set(3,600)
    camera.set(4,600)

    cascade = cv2.CascadeClassifier('face_cascade.xml')

    num_faces = 0

    if not pred:
        print('Who is this (enter a number from 1 to infinity)?')
        i = input('=> ')

    while True:
        _,frame = camera.read()
        gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(
                gray_frame,scaleFactor=1.2,minNeighbors=5,minSize=(20,20)
                )
        for (x,y,w,h) in faces:
            if pred:
                iden,conf = predict(gray_frame[y:y+h,x:x+w])
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(gray_frame,'Guess:'+str(iden),(x+10,y-10),font,1,(127,127,127),2)
                cv2.putText(gray_frame,'Confidence:'+str(math.floor(conf)),(x-10,y+h-10),font,1,(127,127,127),1)
            else:
                cv2.rectangle(gray_frame,(x,y),(x+w,y+h),(0,255,0),2)
                num_faces += 1
                cv2.imwrite('data/user_{}_{}.jpg'.format(i,num_faces),gray_frame[y:y+h,x:x+w])
        cv2.imshow('video',gray_frame)
        key = cv2.waitKey(30)
        if key == 27:
            break

    camera.release()
    cv2.destroyAllWindows()



# The `main` function reads from the arguments passed to the python interpreter, and it performs different operations
# depending on the flag.
# -d: The program is in data mode. It will ask the user about their identity, and then proceed to record the user's face.
# -t: The program is in train mode. It will simply train the LBPH Recognizer and quit.
# -p: The program is in predict mode. It will display text on the screen conveying it's prediction about who it is seeing,
#     as well as how confident it is.
def main():
    if len(sys.argv) > 1:
        if sys.argv[1][1] == 'd':
            start()
        if sys.argv[1][1] == 't':
            train(cv2.CascadeClassifier('face_cascade.xml'))
        if sys.argv[1][1] == 'p':
            start(pred=True)

if __name__ == '__main__':
    main()

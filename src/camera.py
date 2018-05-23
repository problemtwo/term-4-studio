# Made with help from https://www.hackster.io/mjrobot/real-time-face-recognition-an-end-to-end-project-a10826

import cv2
import math
import numpy as np
import os
import sys
from PIL import Image

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


def train(cascade):
    rec = cv2.face.LBPHFaceRecognizer_create()
    detected,ids = detect(cascade)
    rec.train(detected,np.array([int(i) for i in ids]))
    rec.write('trainer.yml')

def predict(part_of_image):
    rec = cv2.face.LBPHFaceRecognizer_create()
    rec.read('trainer.yml')
    names = ['Error','Abhi']
    iden,conf = rec.predict(part_of_image)
    if conf < 100:
        return names[iden],conf
    return 'Unknown',conf

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
                cv2.putText(gray_frame,'Guess:'+str(iden),(x+10,y-10),font,1,(255,255,255),2)
                cv2.putText(gray_frame,'Confidence:'+str(math.floor(conf)),(x-10,y+h-10),font,1,(255,255,255),1)
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

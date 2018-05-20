import cv2

WIDTH = 840
HEIGHT = 680

camera = cv2.VideoCapture(0)
camera.set(3,WIDTH)
camera.set(4,HEIGHT)

# Adapted from https://www.hackster.io/mjrobot/real-time-face-recognition-an-end-to-end-project-a10826

while True:
 _,img = camera.read()
 img = cv2.flip(img,-1)
 gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
 faces = haar_cascade.detectMultiScale(
   gray,scaleFactor=1.2,minNeighbors=5,minSize=(20,20)
   )
 for (x,y,width,height) in faces:
  cv2.rectange(img,(x,y),(x+width,y+height),(255,0,0),2)
    
 cv2.imshow('video',img)

 key = cv2.waitKey(30) & 0xff
 if k == 27:
  break

camera.release()

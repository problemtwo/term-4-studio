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

rec = cv2.face.LBPHFaceRecognizer_create()
if len(sys.argv) > 1 and len(sys.argv[1]) > 1 and sys.argv[1][1] != 't' and sys.argv[1][1] != 'd':
	rec.read('trainer.yml')

with open('names.txt') as names_file:
	names = ['Unknown'] + [x.strip() for x in names_file.readlines()]

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
	detected,ids = detect(cascade)
	min_len = min(len(detected),len(ids))
	rec.train(detected[:min_len],np.array([int(i) for i in ids[:min_len]]))
	rec.write('trainer.yml')

# The `predict` function will read the data from the Yaml file, and create a Local Binary Patterns
# Histograms Face Recognizer from that Yaml file. It returns a name (or Unknown / Error if something
# unexpected happened), and a confidence which is a percentage out of 100.
def predict(part_of_image):
	iden,conf = rec.predict(cv2.cvtColor(part_of_image,cv2.COLOR_BGR2GRAY))
	if conf < 100:
		return names[iden],conf
	return 'Unknown',conf

# The `start` function projects a grayscale version of the camera's video. Depending on the flags passed
# into the program, this function might either prompt the user about the user's identity (which is a number
# from 1 to infinity), or it might display text conveying the cascade's prediction about the user's indentity,
# and how confident it is about it's prediction.
def start(ask_user=False,pred=False,acc_out=False,inp_path=None):
	camera = cv2.VideoCapture(0)
	camera.set(3,600)
	camera.set(4,600)

	cascade = cv2.CascadeClassifier('face_cascade.xml')

	num_faces = 0

	if ask_user:
		print('Who is this (enter a number from 1 to infinity)')
		i = input('=> ')

	if inp_path == None:

		while True:
			_,frame = camera.read()
			faces = cascade.detectMultiScale(
					frame,scaleFactor=1.2,minNeighbors=5,minSize=(20,20)
					)
			for (x,y,w,h) in faces:
				if pred:
					iden,conf = predict(cv2.resize(frame[y:y+h,x:x+w],(596,596)))
					if acc_out:
						with open('accuracy.txt','a') as fl:
							if names.index(iden) == int(i):
								fl.write('id: {}, accuracy: {}\n'.format(iden,round(conf)))
							else:
								fl.write('id: {}, accuracy: {}\n'.format(iden,round(conf*-1)))
					font = cv2.FONT_HERSHEY_SIMPLEX
					cv2.putText(frame,'Guess:'+str(iden),(x+10,y-10),font,1,(255,0,255),2)
					cv2.putText(frame,'Confidence:'+str(math.floor(conf)),(x-10,y+h-10),font,1,(255,0,255),1)
				else:
					cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
					num_faces += 1
					cv2.imwrite('data/user_{}_{}.jpg'.format(i,num_faces),frame[y:y+h,x:x+w])
			cv2.imshow('video',frame)
			key = cv2.waitKey(30)
			if key == 27:
				break

		camera.release()
		cv2.destroyAllWindows()
	else:
		img = cv2.imread(inp_path)
		faces = cascade.detectMultiScale(
				img,scaleFactor=1.2,minNeighbors=5,minSize=(20,20)
				)
		avg = 0
		for (x,y,w,h) in faces:
			iden,conf = predict(cv2.resize(img[y:y+h,x:x+w],(596,596)))	
			print('Detected face: (iden:{},conf:{}%)'.format(iden,round(conf)))
		if len(faces) == 0:
			print('Unknown')

def sum_accuracy():
	with open('accuracy.txt','r') as fl:
		content = [int(line.split(':')[2]) for line in fl.readlines()]
		print('Total Accuracy: {}%'.format(round(sum(content)/len(content))))


def replicate(d,isdir=True,n=1,i=None):
	cascade = cv2.CascadeClassifier('face_cascade.xml')
	num_faces = 0

	if i == None:
		print('Who is this? (a number from 1 to infinity)')
		i = input('=> ')

	if isdir:
		for fl in os.listdir(d):
			img = cv2.imread(os.path.join(d,fl))
			faces = cascade.detectMultiScale(
					img,scaleFactor=1.2,minNeighbors=5,minSize=(20,20)
					)
			for (x,y,w,h) in faces:
				fp = 'data/user_{}_{}.jpg'.format(i,num_faces)
				while os.path.exists(fp):
					num_faces += 1
					fp = 'data/user_{}_{}.jpg'.format(i,num_faces)
				print('Writing image {}'.format(fp))
				cv2.imwrite(fp,img[y:y+h,x:x+w])
	else:
		img = cv2.imread(d)
		faces = cascade.detectMultiScale(
				img,scaleFactor=1.2,minNeighbors=5,minSize=(20,20)
				)
		for (x,y,w,h) in faces:
			for v in range(n):
				fp = 'data/user_{}_{}.jpg'.format(i,num_faces)
				while os.path.exists(fp):
					num_faces += 1
					fp = 'data/user_{}_{}.jpg'.format(i,num_faces)
				print('Writing image {}'.format(fp))
				cv2.imwrite(fp,img[y:y+h,x:x+w])

def compare(pos,neg,n=None):
	print('--------- Positive ---------')
	for i,fl in enumerate(os.listdir(pos)):
		start(pred=True,inp_path=os.path.join(pos,fl))
		if n != None and i >= n:
			break
	print('--------- Negative ---------')
	for i,fl in enumerate(os.listdir(neg)):
		start(pred=True,inp_path=os.path.join(neg,fl))
		if n != None and i >= n:
			break

def multiply(num):
	files = os.listdir('data')
	for n,fl in enumerate(files):
		replicate(os.path.join('data',fl),isdir=False,n=num,i=1+math.floor(n/50))

# The `main` function reads from the arguments passed to the python interpreter, and it performs different operations
# depending on the flag.
# -d: The program is in data mode. It will ask the user about their identity, and then proceed to record the user's face.
# -t: The program is in train mode. It will simply train the LBPH Recognizer and quit.
# -p: The program is in predict mode. It will display text on the screen conveying it's prediction about who it is seeing,
#     as well as how confident it is.
def main():
	if len(sys.argv) > 1:
		if sys.argv[1][1] == 's':
			sum_accuracy()
		elif sys.argv[1][1] == 'a':
			start(ask_user=True,pred=True,acc_out=True)
		elif len(sys.argv) > 3 and sys.argv[1][1] == 'c':
			if len(sys.argv) > 5:
				compare(sys.argv[2],sys.argv[3],int(sys.argv[5]))
			else:
				compare(sys.argv[2],sys.argv[3])
		elif sys.argv[1][1] == 'd':
			if len(sys.argv) > 3 and sys.argv[2] == '-i':
				if len(sys.argv) > 5 and sys.argv[4] == '-n':
					replicate(sys.argv[3],isdir=False,n=int(sys.argv[5]))
				else:
					replicate(sys.argv[3],isdir=False)
			elif len(sys.argv) > 3 and sys.argv[2] == '-m':
				replicate(sys.argv[3])
			else:
				start()
		elif sys.argv[1][1] in '123456789':
			multiply(int(sys.argv[1][1]))
		elif sys.argv[1][1] == 't':
			train(cv2.CascadeClassifier('face_cascade.xml'))
		elif sys.argv[1][1] == 'p':
			if len(sys.argv) > 3 and sys.argv[2] == '-i':
				start(pred=True,inp_path=sys.argv[3])
			else:
				start(pred=True)

if __name__ == '__main__':
	main()

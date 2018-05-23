import cv2
import os
import urllib.request

with urllib.request.urlopen('http://image-net.org/api/text/imagenet.synset.geturls?wnid=n07942152') as request:
 urls = request.read().decode().split('\r\n')

if not os.path.exists('back'):
 os.makedirs('back')

for i,e in enumerate(urls):
<<<<<<< HEAD
    print('Getting image {} of {}...'.format(i,len(urls)))
    img_name = 'back/{}.jpg'.format(i)
    if not os.path.exists(img_name):
            try:
                    urllib.request.urlretrieve(e,img_name)
                    img = cv2.imread(img_name)
                    resize = cv2.resize(img,(100,100))
                    cv2.imwrite(img_name,resize)
            except:
                    print('Could not get image {}, continuing...'.format(i))
=======
	print('Getting image {} of {}...'.format(i,len(urls)))
	img_name = 'back/{}.jpg'.format(i)
	if not os.path.exists(img_name):
		try:
			urllib.request.urlretrieve(e,img_name)
			img = cv2.imread(img_name)
			resize = cv2.resize(img,(100,100))
			cv2.imwrite(img_name,resize)
		except:
			print('Could not get image {}, continuing...'.format(i))
>>>>>>> 055579633964080a9e5a889914e71005dc19cc89

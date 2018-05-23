import cv2
import os
import urllib.request

def get_neg_images():
 with urllib.request.urlopen('http://image-net.org/api/text/imagenet.synset.geturls?wnid=n07942152') as request:
  urls = request.read().decode().split('\r\n')

 if not os.path.exists('neg_images'):
  os.makedirs('neg_images')

 for i,e in enumerate(urls):
  print('Getting image {} of {}...'.format(i,len(urls)))
  img_name = 'neg_images/{}.jpg'.format(i)
  if os.path.exists(img_name):
   print('Already have image {}'.format(i))
  else:
   try:
    urllib.request.urlretrieve(e,img_name)
    img = cv2.imread(img_name)
    resize = cv2.resize(img,(100,100))
    cv2.imwrite(img_name,resize)
   except:
    print('Could not get image {}, continuing...'.format(i))



def fmt(i):
 if i < 10:
  return '000' + str(i)
 elif i < 100:
  return '00' + str(i)
 return '0' + str(i)

def get_pos_images():
 if not os.path.exists('pos_images/'):
  os.makedirs('pos_images')
 for i in range(2,62):
  img_name = '../training/images/IMG_{}.JPG'.format(fmt(i))
  if os.path.exists(img_name):
   img = cv2.imread(img_name)
   resize = cv2.resize(img,(50,71))
   cv2.imwrite('pos_images/pos_{}.jpg'.format(i),resize)

def get_info_txt():
 for img in os.listdir('pos_images') + os.listdir('neg_images'):
  if img[:3] == 'pos':
   with open('info.txt','a') as f:
    f.write('pos_images/'+img+' 1 0 0 50 71\n')
  else:
   with open('info.txt','a') as f:
    f.write('neg_images/'+img+'\n')

def get_bg_txt():
 with open('bg.txt','a') as f:
  for img in os.listdir('neg_images'):
   f.write('neg_images/'+img+'\n')

if __name__ == '__main__':
 #get_images()
 #get_pos_images()
 get_info_txt()
 get_bg_txt()

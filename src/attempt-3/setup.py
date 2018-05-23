import os
import platform

try:
 import cv2
except ImportError:
 if platform.system() == 'Linux':
  os.system('sudo pip3 install opencv-python')
 elif platform.system() == 'Darwin':
  os.system('brew install opencv')

if not os.path.exists('back'):
    os.system('python3 make_cascade.py')
os.system('python3 create_bg_txt.py')
os.system('chmod +x create_samples.bash')
os.system('./create_samples.bash')
os.system('chmod +x create_vec.bash')
os.system('./create_vec.bash')
os.system('chmod +x train_cascade.bash')
os.system('./train_cascade.bash')

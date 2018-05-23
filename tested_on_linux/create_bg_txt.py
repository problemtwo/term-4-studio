import os

with open('bg.txt','w') as fl:
	# Borrowed from https://stackoverflow.com/questions/2632205/how-to-count-the-number-of-files-in-a-directory-using-python
	for name in [name for name in os.listdir('./back/') if os.path.isfile(os.path.join('./back/',name))]:
		fl.write(os.path.join('back/',name) + '\n')

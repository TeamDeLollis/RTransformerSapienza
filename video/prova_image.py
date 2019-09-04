import numpy as np
from PIL import Image
import os
import torch
import glob

num_sequences = 2
max_sequence_length = 1
train = np.zeros([num_sequences, max_sequence_length, 500, 500])
"""for j in range(0,num_sequences):
	train[j] = np.zeros(max_sequence_length)
	print(train[j])
	for k in range(0,max_sequence_length):
		train[j][k] = np.zeros([500,500])
"""
images = glob.glob("*.jpg")
for image in images:
	actual_seq = image[8:11]
	index_n = image[12:15]
	img = Image.open(image)     #Open an Image via PILLOW
	img = img.convert('L')      #Grayscale transformation
	imVect = np.array(img)
	actual_seq = int(actual_seq) - 1
	index_n = int(index_n) - 1
	print(train[0])

	for i in range(0,500):
		for j in range(0,500):
			train[actual_seq][index_n][i][j] = imVect[i][j]



train = torch.from_numpy(train)
print(train.shape)

#How to save an image
#imgG = img.convert('L')
#imgG.save('grayscale.jpg')

#We can create a matric of image value
#imVect = np.array(img)
#imTens = torch.from_numpy(imVect)


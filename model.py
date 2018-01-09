import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import cv2
from scipy.ndimage import imread
import os
import h5py

def modeling(x,y,z):
	model = Sequential()
	# input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
	# this applies 32 convolution filters of size 3x3 each.

	model.add(Conv2D(4, (3, 3), activation='relu', input_shape=(x, y, z)))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	#model.add(Dropout(0.25))

	model.add(Conv2D(8, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))	
	#model.add(Dropout(0.25))

	model.add(Conv2D(16, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	#model.add(Dropout(0.25))

	model.add(Conv2D(32, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	#model.add(Dropout(0.25))
	#model.add(Conv2D(64, (3, 3), activation='relu'))
	#model.add(MaxPooling2D(pool_size=(2, 2)))
	
	#model.add(Conv2D(16, (3, 3), activation='relu'))
	#model.add(MaxPooling2D(pool_size=(2, 2)))
	
	#model.add(Conv2D(256, (3, 3), activation='relu'))
	#model.add(MaxPooling2D(pool_size=(2, 2)))
	#model.add(Conv2D(512, (3, 3), activation='relu'))
	#model.add(MaxPooling2D(pool_size=(2, 2)))




	model.add(Flatten())
#	model.add(Dropout(0.25))
#		model.add(Dense(32768, activation='relu'))
	#model.add(Dropout(0.25))
	#model.add(Dense(16384, activation='relu'))
	#model.add(Dropout(0.1))
	#model.add(Dense(8192, activation='relu'))
	#model.add(Dropout(0.1))
	#model.add(Dense(4096, activation='relu'))
	
	#model.add(Dense(2048, activation='relu'))
	#model.add(Dropout(0.1))
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.1))
	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.1))
	model.add(Dense(20, activation='sigmoid'))
	
	#ada = optimizers.Adamax(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	ada = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	#ada = optimizers.Adadelta(lr=0.1, rho=0.95, epsilon=1e-08, decay=0.0)
	#ada = optimizers.Adagrad(lr=0.001, epsilon=1e-08, decay=0.0)
	#sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(loss='binary_crossentropy', optimizer=ada, metrics=['accuracy','precision','fmeasure','recall'])
	
	return model


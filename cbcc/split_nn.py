import tensorflow
import keras
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import feature
import cv2
from keras.models import Sequential
from keras.layers import Dense, Dropout
import pixel_selection


# This is used when calling the nn.py with multiple data sets, as it is for training the neural network with the
# select pixels
def train(image, pixels, v, color, texture, combine, sd_matrix):
	# The features will be [6 texture, 6 color], see feature.py to see what the descriptors are
	data = feature.run_pixels(image, pixels, sd_matrix)

	# The first column of each pixel is the 1 or 0 for rust or no rust respectively
	labels = pixels[:, 0].reshape(-1, 1)
	texture_data = data[:, :6]
	color_data = data[:, 6:]

	# running the training using Keras
	color.fit(color_data, labels, epochs=1000, batch_size=250, verbose=v)
	texture.fit(texture_data, labels, epochs=2000, batch_size=250, verbose=v)
	color_return = color.predict_on_batch(color_data)
	texture_return = texture.predict_on_batch(texture_data)
	combine_data = np.concatenate((color_return, texture_return), axis=1)
	combine.fit(combine_data, labels, epochs=1000, batch_size=250, verbose=v)
	return


def generate_prediction(image, color, texture, combine, sd_matrix):
	shape = image.shape[:2]
	print("Getting features...")
	data = feature.run_image(image, sd_matrix)
	texture_data = data[:, :6]
	color_data = data[:, 6:]
	print("Running networks...")
	color_return = color.predict_on_batch(color_data)
	texture_return = texture.predict_on_batch(texture_data)
	combine_data = np.concatenate((color_return, texture_return), axis=1)
	bitmap = combine_nn.predict_on_batch(combine_data)
	bitmap = np.ceil(np.reshape(bitmap, shape)).astype(int)
	grayscale = get_grayscale(bitmap)
	cv2.imshow("Result", grayscale)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	return grayscale


# takes in an array of 0's and 1's and returns an unsigned 8 bit int for to show as a grayscale image
def get_grayscale(array):
	h, w = array.shape
	temp = []
	for i in range(h):
		for j in range(w):
			tp = array[i, j]
			if tp == 1:
				temp.append(0)
			else:
				temp.append(255)
	return np.array(temp).reshape((h, w)).astype(np.uint8)


def update_model(image, train_pix, color, texture, combine, matrix):

	return train(image, train_pix, 0, color, texture, combine, matrix)


if __name__ == '__main__':
	if len(sys.argv) == 4:
		color_nn = keras.models.load_model("color_" + sys.argv[2])
		texture_nn = keras.models.load_model("texture_" + sys.argv[2])
		combine_nn = keras.models.load_model("combine_" + sys.argv[2])
		sd_matrix = np.load(sys.argv[3])
		base_image = cv2.imread(sys.argv[1])
		gen_image = generate_prediction(base_image, color_nn, texture_nn, combine_nn, sd_matrix)
		cv2.imwrite((sys.argv[1][:-4] + "_prediction_" + sys.argv[2][:-3] + ".png"), gen_image)
	elif len(sys.argv) == 5:
		ground_truth = cv2.imread(sys.argv[1])
		image = cv2.imread(sys.argv[2])
		network_name = sys.argv[3]
		sd_matrix = np.load(sys.argv[4])
		pixels = pixel_selection.get_pixels(ground_truth)

		# using a simple sequential neural network
		color_nn = Sequential()
		texture_nn = Sequential()
		combine_nn = Sequential()

		# the input layer is a dense, fully connected, layer with 10 neurons with tanh activation and 12 inputs
		color_nn.add(Dense(100, activation='tanh', input_dim=6, kernel_initializer='uniform'))
		texture_nn.add(Dense(100, activation='tanh', input_dim=6, kernel_initializer='uniform'))
		combine_nn.add(Dense(100, activation='tanh', input_dim=2, kernel_initializer='uniform'))
		# dropout randomly sets input units to 0 during training time to help prevent overfitting
		color_nn.add(Dropout(0.35))
		texture_nn.add(Dropout(0.35))
		combine_nn.add(Dropout(0.35))

		# the hidden layer is 8 neurons with tanh activation
		color_nn.add(Dense(80, activation='tanh', kernel_initializer='uniform'))
		texture_nn.add(Dense(80, activation='tanh', kernel_initializer='uniform'))
		combine_nn.add(Dense(80, activation='tanh', kernel_initializer='uniform'))
		color_nn.add(Dropout(0.35))
		texture_nn.add(Dropout(0.35))
		combine_nn.add(Dropout(0.35))

		# the output layer is only one neuron, for binary classification, and uses tanh for the activation
		color_nn.add(Dense(1, activation='tanh', kernel_initializer='uniform'))
		texture_nn.add(Dense(1, activation='tanh', kernel_initializer='uniform'))
		combine_nn.add(Dense(1, activation='tanh', kernel_initializer='uniform'))

		# compile the network with loss, or what to minimize, being the binary crossentropy, and the optimizer,
		# or what the network uses to improve, being adadelta, and what to report as the
		# performance of the network being the accuracy
		color_nn.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])
		texture_nn.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])
		combine_nn.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])

		train(image, pixels, 1, color_nn, texture_nn, combine_nn, sd_matrix)

		color_nn.save("color_" + network_name)
		texture_nn.save("texture_" + network_name)
		combine_nn.save("combine_" + network_name)

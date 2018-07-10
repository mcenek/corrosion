import tensorflow
import keras
import numpy as np
import matplotlib.pyplot as plt
import sys
import feature
import cv2
from keras.models import Sequential
from keras.layers import Dense, Dropout


# This is used when calling the nn.py with multiple data sets, as it is for training the neural network with the
# select pixels
def train(image, network, pixels, v):
	# The features will be [6 texture, 6 color], see feature.py to see what the descriptors are
	data = feature.run_pixels(image, pixels)

	# The first column of each pixel is the 1 or 0 for rust or no rust respectively
	labels = pixels[:, 0].reshape(-1, 1)

	# running the training using Keras
	return network.fit(data, labels, epochs=5000, batch_size=150 , verbose=v)


def test(image, network, pixels):
	# The features will be [6 texture, 6 color], see feature.py to see what the descriptors are
	data = feature.run_pixels(image, pixels)
	# The first column of each pixel is the 1 or 0 for rust or no rust respectively
	labels = pixels[:, 0].reshape(-1, 1)
	# running the network on the testing data
	score = network.evaluate(data, labels, batch_size=100)
	return score


def generate_prediction(image, network):
	shape = image.shape[:2]
	print("Getting features...")
	data = feature.run_image(image)
	print("Running network...")
	bitmap = network.predict_on_batch(data)
	bitmap = np.ceil(np.reshape(bitmap, shape)).astype(int)
	grayscale = get_grayscale(bitmap)
	cv2.imshow("Result", grayscale)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


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


def update_model(image, train_pix, network):

	return train(image, network, train_pix, 0)


if __name__ == '__main__':
	if len(sys.argv) == 5:
		# the first argument will always be the training data
		train_pixels = np.genfromtxt(sys.argv[1]).astype(int)
		# the second argument will always be the test data
		test_pixels = np.genfromtxt(sys.argv[2]).astype(int)
		# the third argument will always be the image that the pixels came from
		base_image = cv2.imread(sys.argv[3])

		# using a simple sequential neural network
		model = Sequential()

		# the input layer is a dense, fully connected, layer with 100 neurons with softsign activation and 12 inputs
		# softsign activation is x/(abs(x) + 1)
		model.add(Dense(100, activation='tanh', input_dim=12, kernel_initializer='uniform'))
		# dropout randomly sets input units to 0 during training time to help prevent overfitting
		model.add(Dropout(0.65))

		# the hidden layer is 100 neurons with softsign activation
		model.add(Dense(100, activation='tanh', kernel_initializer='uniform'))
		model.add(Dropout(0.65))

		# the output layer is only one neuron, for binary classification, and uses softsign for the activation
		model.add(Dense(1, activation='tanh', kernel_initializer='uniform'))

		# compile the network with loss, or what to minimize, being the mean squared error, and the optimizer,
		# or what the network uses to improve, being stochastic gradient descent, and what to report as the
		# performance of the network being the accuracy
		model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])

		history = train(base_image, model, train_pixels, 1)

		plt.plot(history.history['acc'])
		plt.title('model accuracy')
		plt.ylabel('accuracy')
		plt.xlabel('epoch')
		plt.show()

		accuracy = test(base_image, model, test_pixels)
		print(accuracy)
		model.save(sys.argv[4])

	elif len(sys.argv) == 3:
		model = keras.models.load_model(sys.argv[2])
		base_image = cv2.imread(sys.argv[1])
		generate_prediction(base_image, model)

	else:
		print("Please use the correct input of:")
		print("nn.py <training_pixels.txt> <test_pixels.txt> <image.jpg> <save_name.h5>")
		print("or nn.py <image.jpg> <neural_network.h5>")

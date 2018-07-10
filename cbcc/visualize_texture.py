import cv2
import feature
import sys
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
	if len(sys.argv) == 2:
		print("Loading image " + sys.argv[1])
		image = cv2.imread(sys.argv[1])
		shape = image.shape
		print(shape)
		print("Extracting features")
		features = feature.run_image(image)
		textures = features[:, :6]
		print(textures.shape)
		test = textures[:, :3] + textures[:, 3:]
		test = test.reshape(shape)
		test = np.sum(test, axis=2)
		print(test)
		plt.imshow(test, cmap='gray_r', interpolation='nearest')
		plt.show()
	else:
		print("Please add a path to an image")

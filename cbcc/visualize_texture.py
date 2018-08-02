import cv2
import feature
import sys
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
	if len(sys.argv) == 3:
		print("Loading image " + sys.argv[1])
		image = cv2.imread(sys.argv[1])
		sd_matrix = np.load(sys.argv[2])
		shape = image.shape
		print(shape)
		print("Extracting features")
		features = feature.run_image(image, sd_matrix)
		textures = features[:, :6]
		print(textures[:, 0].shape)
		plt.boxplot(textures, vert=False)
		plt.show()
	else:
		print("Please add a path to an image")

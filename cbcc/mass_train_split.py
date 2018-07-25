import numpy as np
import split_nn
import pixel_selection
import cv2
import os
import sys
import tensorflow
import keras


def init(filepath):
	image_paths = []
	for image in os.walk(filepath):
		for i in range(len(image[2])):
			image_paths.append(filepath + "\\" + image[2][i])
	images = np.array([cv2.imread(i) for i in image_paths])
	return images


def get_pixels(images):
	pixels = []
	for image in images:
		pixels.append(pixel_selection.get_pixels(image))
	return np.array(pixels)


if __name__ == '__main__':
	if len(sys.argv) == 4:
		print("Loading images...")
		groundtruths = init(sys.argv[1])
		images = init(sys.argv[2])
		print("Getting pixels")
		pixel = get_pixels(groundtruths)
		network_name = sys.argv[3]
		color_nn = keras.models.load_model("color_" + network_name)
		texture_nn = keras.models.load_model("texture_" + network_name)
		combine_nn = keras.models.load_model("combine_" + network_name)
		toolbar_width = len(groundtruths)

		# setup toolbar
		sys.stdout.write("[%s]" % (" " * toolbar_width))
		sys.stdout.flush()
		sys.stdout.write("\b" * (toolbar_width + 1))  # return to start of line, after '['
		for i in range(len(groundtruths)):
			split_nn.update_model(images[i], pixel[i], color_nn, texture_nn, combine_nn)
			sys.stdout.write("-")
			sys.stdout.flush()
		sys.stdout.write("\n")
		color_nn.save("color_" + network_name)
		texture_nn.save("texture_" + network_name)
		combine_nn.save("combine_" + network_name)
		print("All done")
	else:
		print("Please input the folder of groundtruths, the folder of the original images, then the h5 file for the "
		      "neural network")
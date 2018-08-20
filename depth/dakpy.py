import numpy as np
import cv2


def __split_pixels(label):
	h, w = label.shape[:2]
	positive = []
	negative = []
	for i in range(h):
		for j in range(w):
			if label[i, j] == 255:
				negative.append((i, j))
			else:
				positive.append((i, j))
	return np.array(positive), np.array(negative)


def __get_pixels(image):
	positive, negative = __split_pixels(image)
	np.random.shuffle(positive)
	np.random.shuffle(negative)
	labeled_positive = np.insert(positive, 0, 1, axis=1)
	labeled_negative = np.insert(negative, 0, -1, axis=1)
	pixels = np.concatenate((labeled_positive[:2000], labeled_negative[:2000])).astype(int)
	return pixels


def __get_patch(pixel, image, height, width, size):
	radius = size  # Used for patch size
	diameter = 2 * radius
	# max_row, max_col, not_used = np.array(image).shape Having this was making it super slow, so just manually put
	# in the size of the images i guess
	max_row = height
	max_col = width
	if pixel[0] >= (max_row - radius):
		corner_row = max_row - (diameter + 2)

	elif pixel[0] >= radius:
		corner_row = pixel[0] - radius

	else:  # With the row coordinate being less than the radius of the patch, it has to be at the top of the image
		corner_row = 0  # meaning the row coordinate for the patch will have to be 0
	# should be the same as the pixel coming in as it should be less than 11

	if pixel[1] >= (max_col - radius):
		corner_col = max_col - (diameter + 2)

	elif pixel[1] >= radius:
		corner_col = pixel[1] - radius

	else:  # With the column coordinate being less than the radius of the patch, it has to be in the left side of the
		corner_col = 0  # Image, meaning the column coordinate for the patch will have to be 0
	diameter += 1  # Added 1 for the center pixel

	return image[corner_row:(corner_row + diameter), corner_col:(corner_col + diameter)]


def get_train(ground_path, imagepath, size):
	groundtruth = cv2.imread(ground_path, cv2.IMREAD_GRAYSCALE)
	image = cv2.imread(imagepath, 2)
	h, w = image.shape[:2]

	pixels = __get_pixels(groundtruth)

	patches = []

	for i in range(len(pixels)):
		patches.append(__get_patch(pixels[i, 1:], image, h, w, size))

	return pixels, np.array(patches)


def get_full(imagePath, size):
	image = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
	h, w = image.shape[:2]
	
	diameter = (size * 2) + 1

	shape = (h, w, diameter, diameter)

	patches = []

	for i in range(h):
		for j in range(w):
			coordinate = (i, j)
			patches.append(__get_patch(coordinate, image, h, w, size))

	return np.array(patches).reshape(shape)

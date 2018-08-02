import numpy as np
import cv2
import os
import sys


# Gets the p points in a circle with a radius specified, this will be passed to get_texture
def __radial_points():
	radius = 3  # Used to get the neighbors, needs to be smaller than the patch radius
	p = 8  # total number of neighbors selected
	k = np.array(list(range(1, p + 1)))  # k is the index of the neighbors, 1 through p, the range function is [a, b)
	# which is why i use p+1
	a_k = [((k - 1) * 2 * np.pi) / p]  # a_k is the radial transformation of the k indexes
	x = radius * np.cos(a_k)  # The x coordinate for the neighbors
	y = -radius * np.sin(a_k)  # The y coordinate for the neighbors
	return [x, y]


# Get patch returns a cropped portion of the image provided using the globally defined radius
# pixel is a tuple of (row, column) which is the row number and column number of the pixel in the picture
# image is a cv2 image
def __get_patch(pixel, image, height, width):
	radius = 6  # Used for patch size
	diameter = 2 * radius
	# max_row, max_col, not_used = np.array(image).shape Having this was making it super slow, so just manually put
	# in the size of the images i guess
	max_row = height
	max_col = width
	if pixel[0] >= (max_row - radius):
		corner_row = max_row - (diameter + 2)
		center_row = radius + pixel[0] - (max_row - (radius + 1))
	elif pixel[0] >= radius:
		corner_row = pixel[0] - radius
		center_row = radius
	else:  # With the row coordinate being less than the radius of the patch, it has to be at the top of the image
		corner_row = 0  # meaning the row coordinate for the patch will have to be 0
		center_row = pixel[0]  # Because the pixel in question is less than the radius, the center of the patch
	# should be the same as the pixel coming in as it should be less than 11

	if pixel[1] >= (max_col - radius):
		corner_col = max_col - (diameter + 2)
		center_col = radius + pixel[1] - (max_col - (radius + 1))
	elif pixel[1] >= radius:
		corner_col = pixel[1] - radius
		center_col = radius
	else:  # With the column coordinate being less than the radius of the patch, it has to be in the left side of the
		corner_col = 0  # Image, meaning the column coordinate for the patch will have to be 0
		center_col = pixel[1]  # same as the row
	diameter += 1  # Added 1 for the center pixel

	return image[corner_row:(corner_row + diameter), corner_col:(corner_col + diameter)], (center_row, center_col)


# Uses a modified version of the method detailed in the article "Vision-Based Corrosion Detection Assisted by a
# Micro-Aerial Vehicle in a Vessel Inspection Application" by Ortiz et. Al.. It describes using the RGB values to
# create a texture feature vector based on the difference in color from the center pixel and the neighbors selected
# in a circle around the pixel in question. p neighbors are selected in a similar fashion to the article,
# expect instead of using bilinear interpolation, I just rounded the x and y values to get a proper index for the
# patch matrix.
def __get_difference(patch, pixel, radial):
	c_b, c_g, c_r = patch[pixel[0], pixel[1]]  # The rgb values of the center pixel
	neighbor = [[], [], []]
	if pixel[0] != 6 or pixel[1] != 6:  # With the center not being the actual center,
		# neighbors are chosen at random
		x = np.random.choice(12, 8)  # Very Important!! This needs to go to the size of the patch,
		y = np.random.choice(12, 8)  # which is done manually, so if the size of the patch is changed this
	#  needs to be as well
	else:
		x = np.round(pixel[0] + radial[0])  # pixel is the x & y coordinate for the central pixel, which is the offset
		y = np.round(pixel[1] + radial[1])  # for the neighbors, which is then rounded off so it is a whole number
		x = x.astype(int).flatten()
		y = y.astype(int).flatten()
	for i in range(len(x)):  # getting the rgb values for all of the neighbors
		b, g, r = patch[x[i], y[i]]
		neighbor[0].append(r)
		neighbor[1].append(g)
		neighbor[2].append(b)
	# remaking the neighbors into a numpy array, and changing them to a 32 bit int instead of the 8 bit int of rgb
	# values so they can be negative
	neighbor = np.array(neighbor).astype(int)
	# getting the difference from the neighbors color and the center pixels color
	diff = np.array([neighbor[0] - c_r, neighbor[1] - c_g, neighbor[2] - c_b])
	# splitting the difference into positive results and negative results
	# pos_diff = np.array([diff[0][diff[0] > 0], diff[1][diff[1] > 0], diff[2][diff[2] > 0]])
	# neg_diff = np.array([diff[0][diff[0] < 0], diff[1][diff[1] < 0], diff[2][diff[2] < 0]])
	binned = diff // 8
	return binned


def get_matrix(image):
	shape = image.shape[:2]
	matrix = []
	radial = __radial_points()
	for row in range(shape[0]):
		ro_array = []
		for column in range(shape[1]):
			patch, pixel = __get_patch((row, column), image, shape[0], shape[1])
			ro_array.append(__get_difference(patch, pixel, radial))
		matrix.append(ro_array)
	return np.array(matrix)


def init(filepath):
	image_paths = []
	image_names = []
	for image in os.walk(filepath):
		for i in range(len(image[2])):
			image_paths.append(filepath + "\\" + image[2][i])
			image_names.append(image[2][i])
	images = np.array([cv2.imread(i) for i in image_paths])
	return images, image_names


if __name__ == '__main__':
	image_array, names = init(sys.argv[1])
	# os.mkdir("sd")
	os.chdir("./sd")
	i = 0
	for im in image_array:
		save_matrix = get_matrix(im)
		np.save("binned_sd_" + names[i], save_matrix)
		i += 1

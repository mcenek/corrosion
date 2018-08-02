import numpy as np
import cv2


# Get patch returns a cropped portion of the image provided using the globally defined radius
# pixel is a tuple of (row, column) which is the row number and column number of the pixel in the picture
# image is a cv2 image
def get_patch(pixel, image, height, width, sd_matrix):
	radius = 6  # Used for patch size
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

	if pixel[1] >= (max_col - radius):
		corner_col = max_col - (diameter + 2)
	elif pixel[1] >= radius:
		corner_col = pixel[1] - radius
	else:  # With the column coordinate being less than the radius of the patch, it has to be in the left side of the
		corner_col = 0  # Image, meaning the column coordinate for the patch will have to be 0
	diameter += 1  # Added 1 for the center pixel

	return image[corner_row:(corner_row + diameter), corner_col:(corner_col + diameter)], sd_matrix[corner_row:(
			corner_row + diameter), corner_col:(corner_col + diameter)]


def k_means_color(patch):
	z = patch.reshape((-1, 3))
	# Set the rgb values to be floats so it can be used in the k-means function
	z = np.float32(z)
	# Create the criteria for k-means clustering, 1st: Stop kmeans when the specified accuracy is met, or when the
	# max iterations specified is met. 2nd: max iterations. 3rd: epsilon, or required accuracy
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
	k = 2
	# run the K means clustering using cv2, so it can be done easily with images
	# label and center being the important returns, with label being important for producing an image to show the clusters
	# and center being useful for the NN and the producing the image to show the clusters, as its the average color of each
	# cluster. Arguments for the kmeans, 1st: input data, 2nd: number of clusters needed, 3rd: not sure,
	# 4th: the criteria specified above, 5th: number of times to run the clustering taking the best result, 6th: flags
	ret, label, center = cv2.kmeans(z, k, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)
	center = np.uint8(center)  # Center will contain the Dominant Colors in their respective color channels
	# i.e [[DCb, DCg, DCr], [DCb, DCg, DC3r]] with k = 2
	return center.flatten()


# Returns the dominate colors in a patch, which are the average colors based upon what is the center of clusters that
# are built from the rgb values in the patch, patch is a 3d array which is 50x50x3

def get_dominate_color(patch):
	b_root = patch[:, :, 0]
	g_root = patch[:, :, 1]
	r_root = patch[:, :, 2]

	b_root_mean = np.mean(b_root)
	g_root_mean = np.mean(g_root)
	r_root_mean = np.mean(r_root)

	b_child_0 = b_root[b_root > b_root_mean]
	b_child_1 = b_root[b_root <= b_root_mean]

	if b_child_0.size == 0:
		half = b_root.size // 2
		b_child_0 = b_root[:half]
		b_child_1 = b_root[half:]

	g_child_0 = g_root[g_root > g_root_mean]
	g_child_1 = g_root[g_root <= g_root_mean]

	if g_child_0.size == 0:
		half = g_root.size // 2
		b_child_0 = g_root[:half]
		b_child_1 = g_root[half:]

	r_child_0 = r_root[r_root > r_root_mean]
	r_child_1 = r_root[r_root <= r_root_mean]

	if r_child_0.size == 0:
		half = r_root.size // 2
		b_child_0 = r_root[:half]
		b_child_1 = r_root[half:]

	center = [np.mean(b_child_0), np.mean(g_child_0), np.mean(r_child_0), np.mean(b_child_1), np.mean(g_child_1),
	          np.mean(r_child_1)]

	return center


def get_texture(sd_patch):
	blue = sd_patch[:, :, 0]
	green = sd_patch[:, :, 1]
	red = sd_patch[:, :, 2]

	r_values, r_counts = np.unique(red, return_counts=True)
	b_values, b_counts = np.unique(blue, return_counts=True)
	g_values, g_counts = np.unique(green, return_counts=True)


	r_neg_len = len(r_values[r_values < 0])
	b_neg_len = len(b_values[b_values < 0])
	g_neg_len = len(g_values[g_values < 0])

	r_neg_count = r_counts[:r_neg_len]
	r_pos_count = r_counts[r_neg_len:]

	r_neg_divisor = np.sum(r_neg_count)
	r_pos_divisor = np.sum(r_pos_count)

	b_neg_count = b_counts[:b_neg_len]
	b_pos_count = b_counts[b_neg_len:]

	b_neg_divisor = np.sum(b_neg_count)
	b_pos_divisor = np.sum(b_pos_count)

	g_neg_count = g_counts[:g_neg_len]
	g_pos_count = g_counts[g_neg_len:]

	g_neg_divisor = np.sum(g_neg_count)
	g_pos_divisor = np.sum(g_pos_count)

	r_neg_prob = r_neg_count / r_neg_divisor
	r_pos_prob = r_pos_count / r_pos_divisor

	b_neg_prob = b_neg_count / b_neg_divisor
	b_pos_prob = b_pos_count / b_pos_divisor

	g_neg_prob = g_neg_count / g_neg_divisor
	g_pos_prob = g_pos_count / g_pos_divisor
	return np.array([np.sum(r_neg_prob**2), np.sum(r_pos_prob**2), np.sum(b_neg_prob**2), np.sum(b_pos_prob**2),
	                 np.sum(g_neg_prob**2), np.sum(g_pos_prob**2)])


def run_pixels(image, data, sd_matrix):
	h, w = image.shape[:2]  # getting the height and width of the image for the patch calculations
	return_array = []
	texture = []
	color = []
	coordinates = data[:, 1:]  # removing the label for the data
	for coordinate in coordinates:
		patch, sd_patch = get_patch(coordinate, image, h, w, sd_matrix)
		descriptor_color = k_means_color(patch)
		descriptor_texture = get_texture(sd_patch)
		texture.append(descriptor_texture)
		color.append(descriptor_color)
	return_array.extend(np.concatenate((texture, color), axis=1))
	return np.array(return_array)


def run_image(image, sd_matrix):
	h, w = image.shape[:2]  # getting the height and width of the image for the patch calculations
	return_array = []
	texture = []
	color = []
	for i in range(h):
		for j in range(w):
			coordinate = (i, j)
			patch, sd_patch = get_patch(coordinate, image, h, w, sd_matrix)
			descriptor_color = k_means_color(patch)
			descriptor_texture = get_texture(sd_patch)
			texture.append(descriptor_texture)
			color.append(descriptor_color)
	return_array.extend(np.concatenate((texture, color), axis=1))
	return np.array(return_array)


# The rest of the code is for running an entire folder of images
# def init(filepath):
# 	for image in os.walk(filepath):
# 		image_paths = filepath + "\\" + image[2]
# 	images = np.array([cv2.imread(i) for i in image_paths])
# 	return images
#
#
# if __name__ == '__main__':
# 	path = "b:\\cbcc\\defects1\\image047.jpg"
# 	start_time = time.time()
# 	radial = radial_points()
# 	# images = init(path)
# 	image = cv2.imread(path)
# 	t_time = time.time()
# 	print("Images loaded:" + str(t_time - start_time))
# 	total_time = t_time
# 	# for image in images:
# 	h, w = image.shape[:2]
# 	for i in range(0, h):
# 		for j in range(0, w):
# 			center_pixel = (i, j)
# 			patch, pixel = get_patch(center_pixel, image, h, w)
# 			descriptor_color = get_dominate_color(patch)
# 			descriptor_texture = get_texture(patch, pixel, radial)
# 	print("Final time: " + str(time.time() - total_time))

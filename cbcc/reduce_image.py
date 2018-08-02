import cv2
import os
import sys
import numpy as np


def init(filepath):
	image_paths = []
	image_names = []
	for im in os.walk(filepath):
		for i in range(len(im[2])):
			image_paths.append(filepath + "\\" + im[2][i])
			image_names.append(im[2][i][:-4])
	images = np.array([cv2.imread(i) for i in image_paths])
	return images, image_names


if __name__ == '__main__':
	if len(sys.argv) == 3:
		image, image_name = init(sys.argv[1])
		ground_truth, ground_name = init(sys.argv[2])
		os.mkdir("reduce75")
		os.mkdir("reduce75gt")
		os.mkdir("reduce5")
		os.mkdir("reduce5gt")
		os.mkdir("reduce25")
		os.mkdir("reduce25gt")
		for i in range(len(image)):
			reduce75 = cv2.resize(image[i], None, fx=.75, fy=.75)
			reduce75gt = cv2.resize(ground_truth[i], None, fx=.75, fy=.75)
			reduce5 = cv2.resize(image[i], None, fx=.5, fy=.5)
			reduce5gt = cv2.resize(ground_truth[i], None, fx=.5, fy=.5)
			reduce25 = cv2.resize(image[i], None, fx=.25, fy=.25)
			reduce25gt = cv2.resize(ground_truth[i], None, fx=.25, fy=.25)
			cv2.imwrite("reduce75\\" + image_name[i] + "_75.jpg", reduce75)
			cv2.imwrite("reduce75gt\\" + ground_name[i] + "_75.jpg", reduce75gt)
			cv2.imwrite("reduce5\\" + image_name[i] + "_5.jpg", reduce5)
			cv2.imwrite("reduce5gt\\" + ground_name[i] + "_5.jpg", reduce5gt)
			cv2.imwrite("reduce25\\" + image_name[i] + "_25.jpg", reduce25)
			cv2.imwrite("reduce25gt\\" + ground_name[i] + "_25.jpg", reduce25gt)
	else:
		print("Please use this following this pattern:\nreduce_image.py <imagefolder> <groundtruthfolder>")

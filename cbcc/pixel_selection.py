import cv2
import numpy as np
import sys
import time


def split_pixels(label):
	h, w = label.shape[:2]
	positive = []
	negative = []
	for i in range(h):
		for j in range(w):
			if list(label[i, j]) == [0, 0, 0]:
				positive.append((i, j))
			else:
				negative.append((i, j))
	return np.array(positive), np.array(negative)


# For running this script from another one
def get_pixels(image):
	positive, negative = split_pixels(image)
	np.random.shuffle(positive)
	np.random.shuffle(negative)
	labeled_positive = np.insert(positive, 0, 1, axis=1)
	labeled_negative = np.insert(negative, 0, -1, axis=1)
	pixels = np.concatenate((labeled_positive[:100], labeled_negative[:300])).astype(int)
	return pixels


if __name__ == '__main__':
	# If you run this code via CLI the input needs to be the image of the groundtruth
	if len(sys.argv) == 2:
		ground_truth_path = sys.argv[1]
	else:
		ground_truth_path = "B:\\cbcc\\labelled1\\image047gt.jpg"
	start_time = time.time()
	ground_truth = cv2.imread(ground_truth_path)
	rust, no_rust = split_pixels(ground_truth)

	np.random.shuffle(rust)
	np.random.shuffle(no_rust)
	labeled_rust = np.insert(rust, 0, 1, axis=1)
	labeled_no_rust = np.insert(no_rust, 0, -1, axis=1)
	training = np.concatenate((labeled_rust[:200], labeled_no_rust[:200])).astype(int)
	test = np.concatenate((labeled_rust[201:251], labeled_no_rust[201:251])).astype(int)
	np.random.shuffle(training)
	np.random.shuffle(test)
	np.savetxt("training.txt", training, fmt='%-5.1i')
	np.savetxt("test.txt", test, fmt='%-5.1i')
	t_time = time.time()
	print('EXECUTION TIME: %.4f ' % (t_time - start_time))

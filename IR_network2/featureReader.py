import cv2
import numpy as np
import random
import os
import re
import math
import constants
import scipy.misc
import pywt
from skimage.feature import hog
from skimage import exposure
from segmentModule import *
from matplotlib import pyplot as plt

#rule for getting the corresponding label image name from the test image name
#if the rule changes you'll have to change this function
def getImageMaskName(img_name):
    split = os.path.splitext(img_name)
    base = split[0][:7]
    num = split[0][7:]

    return base + 'GT' + num + split[1]
#reads in training image for cnn using pixel data as the training set
#28 x 28 surrounding area of each pixel used for training
#3x3 conv, 7x7 conv
#all training images must be passed when calling nn.py
def cnn_readOneImg2(image_dir):
    inputs = []
    img = cv2.imread(image_dir,cv2.IMREAD_COLOR)
    original,markers = getSegments(img,False)
    uniqueMarkers = np.unique(markers)
    canvas = original.copy()
    for uq_mark in uniqueMarkers:
        #make a canvas and paint each unique segment
        b = random.randint(0,255)
        g = random.randint(0,255)
        r = random.randint(0,255)
        canvas[markers == uq_mark] = [b,g,r]

    return(canvas,markers)

#gets n patches from an image with its respective label
def getPixelBatch(n):

    #initialize variablees
    cat_dir = constants.CAT_DIR
    label_dir = constants.LABEL_DIR
    categories = [constants.CAT1_ONEHOT,constants.CAT2_ONEHOT]
    inputs = []
    labels = []

    #check if the file directories exists
    if os.path.exists(cat_dir) and os.path.exists(label_dir):
        files = os.listdir(cat_dir)
        rand_findex = random.randint(0,len(files) - 1)

        #get a random image and label and read the images in
        catfile = os.path.join(cat_dir,files[rand_findex])
        labelfile = os.path.join(label_dir,getImageMaskName(files[rand_findex]))

        #create the input image with extra channels describing texture
        img = cv2.imread(catfile,cv2.IMREAD_COLOR)
        gray_img = cv2.imread(catfile,cv2.IMREAD_GRAYSCALE)
        hogimg = getHOG(img)
        wt = extractWT(gray_img)
        img = np.concatenate((img,hogimg.reshape((hogimg.shape[0],hogimg.shape[1],1))),axis=-1)
        img = np.concatenate((img,wt.reshape((wt.shape[0],wt.shape[1],1))),axis=-1)

        #turns out the reads aren't completely binary. We have a lot of gray areas so we convert it to binary
        mask = cv2.imread(labelfile,cv2.IMREAD_GRAYSCALE)

        #create the training / testing set by picking a batch size of random points from the random image
        #even out the training between points that are and are not rust
        h,w,d = img.shape
        low = int(constants.IMG_SIZE / 2)
        offset = low
        high_w = int(w - (constants.IMG_SIZE / 2) - 1)
        high_h = int(h - (constants.IMG_SIZE / 2) - 1)
        tmp = mask[low:high_h,low:high_w]
        x1s,y1s = np.where(tmp == 0)
        x2s,y2s = np.where(tmp != 0)

        #cut off the edges so we don't choose pixels on the boundry
        for i in range(n):
            j = random.randint(0,9)
            if j % 2 == 0:
                pt_id = random.randint(0,len(x1s) - 1)
                x = x1s[pt_id]
                y = y1s[pt_id]
                labels.append([0.0])
            else:
                pt_id = random.randint(0,len(x2s) - 1)
                x = x2s[pt_id]
                y = y2s[pt_id]
                labels.append([1.0])

            x += offset
            y += offset
            box_low1 = int(x - int(constants.IMG_SIZE / 2) )
            box_low2 = int(y - int(constants.IMG_SIZE / 2) )
            box_high1 = int(x + int(constants.IMG_SIZE / 2))
            box_high2 = int(y + int(constants.IMG_SIZE / 2))
            box = img[box_low1:box_high1,box_low2:box_high2,:]
            inputs.append(box)

            if len(box) == 0:
                print("ERROR")
                quit()
    else:
        print("%s directory does not exist" % cat_dir)
        quit()

    #shuffle the inputs so we mix the rust vs non rust instances well
    tmp = list(zip(inputs,labels))
    random.shuffle(tmp)
    inputs,labels = zip(*tmp)

    return np.array(inputs),np.array(labels)

#gets n patches from an image with its respective label
def getTestBatch(n):

    #initialize variablees
    cat_dir = constants.TEST_CATDIR
    label_dir = constants.TEST_LABELDIR
    inputs = []
    labels = []
    imgs = []
    masks = []

    #check if the file directories exists
    if os.path.exists(cat_dir) and os.path.exists(label_dir):

        #initialize variables
        imgfiles = os.listdir(cat_dir)
        n = int(n/len(imgfiles))

        #get every image in the validation set
        for a in range(len(imgfiles)):
            #make sure we get the corresponding image with the ground truth
            imgname = imgfiles[a]
            gtname = getImageMaskName(imgname)
            img_path= os.path.join(cat_dir,imgname)
            label_path = os.path.join(label_dir,gtname)

            #read in the imgs
            img = cv2.imread(img_path,cv2.IMREAD_COLOR)
            mask = cv2.imread(label_path,cv2.IMREAD_GRAYSCALE)
            gray_img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
            hogimg = getHOG(img)
            wt = extractWT(gray_img)
            img = np.concatenate((img,hogimg.reshape((hogimg.shape[0],hogimg.shape[1],1))),axis=-1)
            img = np.concatenate((img,wt.reshape((wt.shape[0],wt.shape[1],1))),axis=-1)

            #make sure our masks are binary
            mask[mask != 255] = 0

            #create the training / testing set by picking a batch size of random points from the random image
            #even out the training between points that are and are not rust
            h,w,d = img.shape
            low = int(constants.IMG_SIZE / 2)
            offset = low
            high_w = int(w - (constants.IMG_SIZE / 2) - 1)
            high_h = int(h - (constants.IMG_SIZE / 2) - 1)
            tmp = mask[low:high_h,low:high_w]
            x1s,y1s = np.where(tmp == 0)
            x2s,y2s = np.where(tmp != 0)

            #cut off the edges so we don't choose pixels on the boundry
            for i in range(n):
                j = random.randint(1,2)
                if j % 2 == 0:
                    pt_id = random.randint(0,len(x1s) - 1)
                    x = x1s[pt_id]
                    y = y1s[pt_id]
                    labels.append([0.0])
                else:
                    pt_id = random.randint(0,len(x2s) - 1)
                    x = x2s[pt_id]
                    y = y2s[pt_id]
                    labels.append([1.0])

                x += offset
                y += offset
                box_low1 = int(x - int(constants.IMG_SIZE / 2) )
                box_low2 = int(y - int(constants.IMG_SIZE / 2) )
                box_high1 = int(x + int(constants.IMG_SIZE / 2))
                box_high2 = int(y + int(constants.IMG_SIZE / 2))
                box = img[box_low1:box_high1,box_low2:box_high2,:]
                inputs.append(box)

                if len(box) == 0:
                    print("ERROR")
                    quit()
    else:
        print("%s directory does not exist" % cat_dir)
        quit()

    #shuffle the inputs so we mix the rust vs non rust instances well
    tmp = list(zip(inputs,labels))
    random.shuffle(tmp)
    inputs,labels = zip(*tmp)

    return np.array(inputs),np.array(labels)

#extract hog from an image
def getHOG(img):
    cellsize = (int(constants.IMG_SIZE / 2),int(constants.IMG_SIZE / 2))
    fd,hog_image = hog(img,orientations=8,pixels_per_cell=cellsize,cells_per_block=(1,1),block_norm='L2-Hys', visualize=True,multichannel=True)
    return hog_image

#extract wavelet transform of an image
def extractWT(image):
    #convert to float
    imArray =  np.float32(image)
    imArray /= 255;
    # compute coefficients
    coeffs=pywt.wavedec2(imArray, 'haar', level=1)

    #Process Coefficients
    coeffs_H=list(coeffs)
    coeffs_H[0] *= 0;

    # reconstruction
    imArray_H=pywt.waverec2(coeffs_H,'haar');
    imArray_H *= 255;
    imArray_H =  np.uint8(imArray_H)

    return imArray_H[:image.shape[0],:image.shape[1]]



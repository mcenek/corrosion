import sys
import segmentModule as seg
import extractionModule as analyze
import numpy as np
import cv2
import time
import re
import pickle
import gabor_threads_roi as gabor
import os

#GLOBAL FLAG VARIABLES
#flags that are handled
showflag = 'show' in sys.argv
hogflag = 'hog' in sys.argv
binflag = 'bin' in sys.argv
sizeflag = 'size' in sys.argv
colorflag = 'color' in sys.argv
gaborflag = 'gabor' in sys.argv
hsvflag = 'hsv' in sys.argv
msflag = 'meanshift' in sys.argv
msbinflag = 'meanshiftbin' in sys.argv
fjmsflag = 'fjmeanshift' in sys.argv
qsflag = 'quickshift' in sys.argv
dbscanflag = 'dbscan' in sys.argv
pcaflag = 'pca' in sys.argv
ldaflag = 'lda' in sys.argv
wtflag = 'wt' in sys.argv
bifflag = 'bif' in sys.argv
blurflag = 'blur' in sys.argv
#Check system argument length and mode
#if mode is bin do 3d color binning
start_time = time.time()

#display different modes
def display(original,labels=None,SHOWFLAG=True):

    #if mode is meanshift, apply meanshift
    if msflag:
        image, labels = analyze.meanshift(original)
        print(labels)
        if SHOWFLAG:
            seg.showSegments(image,labels)

    #if mode is meanshiftbin, convert 2d image to 3d using bin method and apply meanshift
    elif msbinflag:
        image,labels = analyze.meanshift(original,binning=True)
        print(labels)
        if SHOWFLAG:
            seg.showSegments(image,labels)

    #if mode is fjmeanshift, do fjmeanshift
    elif fjmsflag:
        if SHOWFLAG:
            image,labels = seg.getSegments(original,True)
        else:
            image,labels = seg.getSegments(original,False)
        print(labels)

    #if mode is fjmeanshift, do fjmeanshift
    elif qsflag:
        labels = analyze.quickmeanshift(original)
        if SHOWFLAG:
            seg.showSegments(original,labels)

    #if mode is meanshiftbin, convert 2d image to 3d using bin method and apply meanshift
    elif dbscanflag:
        image,labels = analyze.dbscan(original,binning=True)
        print(labels)
        if SHOWFLAG:
            seg.showSegments(image,labels)

    #if mode is size
    elif sizeflag:
        combined_filename = sys.argv[1]

        # Generate and save blob size for this blob we assume black as background
        size = analyze.extractBlobSize(original)
        print('--------------SIZE---------------')
        if SHOWFLAG:
            print(size)
        return size

    #if mode is hog, show hog feature vector of image
    elif hogflag:
        hist = analyze.extractHOG(original,False)
        featurevector = hist.flatten()
        norm = analyze.normalize(featurevector)
        analyze.visualizeHOG(original)
        print('-------------HOG----------------')
        if SHOWFLAG:
            analyze.displayHistogram(featurevector)
        return norm

    #if mode is wavelettransform
    elif wtflag:
        analyze.visualizeWT(original,show=True)
        return 1


    #if mode is gabor, extract gabor feature from image using several orientations
    elif gaborflag:
        orientations = 16
        filters = gabor.build_filters(orientations)
        combined_filename = sys.argv[1]

        # Generate and save ALL hogs for this image
        result = gabor.run_gabor(original, filters, combined_filename, orientations, mode='training')
        featurevector = result.flatten()[1:]
        norm = analyze.normalize(featurevector)
        print('--------------Gabor---------------')
        if SHOWFLAG:
            analyze.displayHistogram(featurevector,'r--')
        return norm

    #if mode is color, show color histogram of image
    elif colorflag:
        hist = analyze.extractColorHist(original,False)
        print('-------------Color----------------')
        if SHOWFLAG:
            analyze.displayHistogram(hist)
        return hist

    elif binflag:
        hist = analyze.extractbinHist(original,False)
        norm = analyze.normalize(hist)
        if SHOWFLAG:
            analyze.displayHistogram(norm)
        return norm

    elif bifflag:
        gray = cv2.cvtColor(original,cv2.COLOR_BGR2GRAY)
        y,x = np.where(gray >= 0)
        out = analyze.bilinear_interpolate(gray,x,y).reshape((gray.shape))
        cv2.namedWindow('bilinear filtering', cv2.WINDOW_NORMAL)
        cv2.imshow('bilinear filtering', out.astype(np.uint8))
        cv2.waitKey(0)
        return 1

    elif blurflag:
        analyze.blurErode(original)

        return 1

    elif hsvflag:
        hsvimg = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
        hist = analyze.extractHSVHist(hsvimg,False)
        if SHOWFLAG:
            analyze.displayHistogram(hist)

        return hist


def scatter(instances,labels):
    if pcaflag:
        analyze.showPCA(instances,labels)

    elif ldaflag:
        optionID = sys.argv.index('lda') + 1
        if optionID <= len(sys.argv) - 1:
            option = sys.argv[optionID]
            analyze.showLDA2(instances,labels,classes=option)
        else:
            analyze.showLDA(instances,labels,mode='int')


if __name__ == '__main__':

    if len(sys.argv) >= 2:

        #if user input is doing lda or pca
        if os.path.isfile(sys.argv[1]) and (ldaflag or pcaflag):

            if os.path.splitext(sys.argv[1])[1] == '.npy':
                tmp = np.load(sys.argv[1],mmap_mode='r')
                instances = tmp[:,:-1].astype(float)
                labels = tmp[:,-1:].astype(int)
            elif os.path.isfile(sys.argv[2]):
                img = cv2.imread(sys.argv[1],cv2.IMREAD_COLOR)
                mask = cv2.imread(sys.argv[2],cv2.IMREAD_COLOR)
                seg_image,labels = seg.getSegments(img,False)

            scatter(instances,labels)

        #if user input to visuzlize is an image file
        else:
            #evaluate single image
            #check if the image was read in correctly
            if os.path.isfile(sys.argv[1]):
                original = cv2.imread(sys.argv[1],cv2.IMREAD_COLOR)
                display(original)
            else:
                print('invalid image! Could not open: %s' % sys.argv[1])
                quit()

    #if less than 3 args given
    else:
        print("wrong number of files as arguments expecting 3:")
        print("argv1 = image file/directory")
        print("argv2 + = modes of operation")
        sys.exit()

    #find out execution time
    print("--- %s seconds ---" % (time.time() - start_time))





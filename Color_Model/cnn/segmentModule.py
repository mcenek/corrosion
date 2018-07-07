#############################################################################################################
#Masa Hu
#
#                               Trash Classification Project
#############################################################################################################
#Library imports
import numpy as np
import cv2
import random
import math
import pymeanshift as pms
import constants
from matplotlib import pyplot as plt
############################################################################################################
#Flag options for imread are self explanatory
#cv2.IMREAD_GRAYSCALE
#Cv2.IMREAD_COLOR
#cv2.IMREAD_UNCHANGED
#############################################################################################################
#Global Variables
allimages = {}                          #put all images in this dictionary here to show them later
#############################################################################################################
###############################################################################################################################
#Documentation
########################################################################
#BilateralFilter
########################################################################
#http://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html?highlight=bilateralfilter#bilateralfilter
########################################################################
#Prameters:
#    src - src image
#    dst - Destination image of the same size and type as src .
#    d - Diameter of each pixel neighborhood that is used during filtering. If it is non-positive, it is computed from sigmaSpace .
#    sigmaColor - Filter sigma in the color space. A larger value of the parameter means that farther colors within the pixel neighborhood (see sigmaSpace ) will be mixed together, resulting in larger areas of semi-equal color.
#    sigmaSpace - Filter sigma in the coordinate space. A larger value of the parameter means that farther pixels will influence each other as long as their colors are close enough (see sigmaColor ). When d>0 , it specifies the neighborhood size regardless of sigmaSpace . Otherwise, d is proportional to sigmaSpace .
#
# bilateralFilter(src, d, sigmaColor, sigmaSpace)

MIN_DENSITY = constants.MIN_DENSITY
SPATIAL_RADIUS = constants.SPATIAL_RADIUS
RANGE_RADIUS = constants.RANGE_RADIUS

def showSegmentDistribution(original,markers):
    features = []
    uniqueMarkers = np.unique(markers)

    #get the sizes of the discovered segments
    size_array = []
    size_dict = {}
    for x in uniqueMarkers:
        count = np.count_nonzero(markers == x)
        size_array.append(count)
        size_dict[x] = count

    #get segment info
    mean = np.mean(size_array)
    total = np.sum(size_array)
    seg_count = len(uniqueMarkers)

    #remove markers given condition ange get unique markers again
    for k in size_dict.keys():
        if(size_dict[k] < mean / 2):
            markers[markers == k] = 0
    uniqueMarkers = np.unique(markers)
    reduced_count = len(uniqueMarkers)

    blank = original.copy()
    blank = original - original
    for label in uniqueMarkers[1:]:
        b = random.randint(0,255)
        g = random.randint(0,255)
        r = random.randint(0,255)
        blank[ markers == label] = [b,g,r]

    #show the segmenting size selection process
    print("mean size: %s" % mean)
    print("segment counts: %s" % seg_count)
    print("reduced counts: %s" % reduced_count)
    size_hist = np.array(size_array)
    y1, x1 = np.histogram(size_array,bins='auto')
    subset = size_hist[size_hist > mean]
    y2, x2 = np.histogram(subset,bins='auto',density=True)

    plt.figure(1)
    plt.subplot(211)
    plt.plot(x1[:-1],y1,'r--')

    plt.subplot(212)
    plt.plot(x2[:-1],y2,'r--')
    plt.pause(0.1)

    cv2.imshow('segments reduced',blank)
    cv2.waitKey(0)

#saves the segments of the original image as png files given the labels
def saveSegments(original,labels,SHOW,out_dir,category):

    unique_labels = np.unique(labels)
    blank = original - original

    #get the sizes of the discovered segments
    size_array = []
    size_dict = {}
    for x in unique_labels:
        count = np.count_nonzero(labels == x)
        size_array.append(count)
        size_dict[x] = count

        #color the blank canvas with the different segments
        b = random.randint(0,255)
        g = random.randint(0,255)
        r = random.randint(0,255)
        blank[labels == x] = [b,g,r]

    #save the blank canvas
    fout_original = "segmented_" + category
    cv2.imwrite(fout_original,blank)

    #get information about the segments
    mean = np.mean(size_array)
    total = np.sum(size_array)
    t_count = len(unique_labels)

    #remove markers given condition and get unique markers again
    for k in size_dict.keys():
        if(size_dict[k] < mean / 2):
            labels[labels == k] = 0
    unique_labels = np.unique(labels)
    reduced_count = len(unique_labels)

    count = 0
    for l in unique_labels[1:]:
        segment = original.copy()
        segment[labels != l] = [0,0,0]

        blank = original.copy()
        blank = blank - blank
        blank[labels == l] = [255,255,255]

        grey = cv2.cvtColor(blank,cv2.COLOR_BGR2GRAY)
        x,y,w,h = cv2.boundingRect(grey)
        cropped = original[y:y+h,x:x+w]
        cropped = np.uint8(cropped)
        resized = cv2.resize(cropped,(256, 256), interpolation = cv2.INTER_CUBIC)

        f_out =  out_dir + str(count) + "_" + category

        cv2.imwrite(f_out,resized)

        count += 1

        if(SHOW):
            cv2.imshow(resized)
            cv2.waitKey(0)

    print("original count: %s     reduced count: %s     category: %s" % (str(t_count),str(len(unique_labels)),str(category)))

###############################################################################################################################


########################################################################

########################################################################
#Canny image
#http://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html?highlight=canny
########################################################################
#Parameters:
#    image - single-channel 8-bit input image.
#    edges - output edge map; it has the same size and type as image .
#    threshold1 - first threshold for the hysteresis procedure.
#    threshold2 - second threshold for the hysteresis procedure.
#    apertureSize - aperture size for the Sobel() operator.
#    L2gradient - a flag, indicating whether a more accurate L_2 norm =\sqrt{(dI/dx)^2 + (dI/dy)^2} should be used to calculate the image gradient magnitude ( L2gradient=true ), or whether the default L_1 norm =|dI/dx|+|dI/dy| is enough ( L2gradient=false ).
def getSegments(original, SHOW):
    allimages["original"] = original
    ##############################################################################################################
    #gaussian Blur
    #blur = cv2.GaussianBlur(gray_img,(5,5),0)
    #allimages["gaussianBlur"] = blur

    #mean shift segmentation on bgr image
    #https://github.com/fjean/pymeanshift
    #http://ieeexplore.ieee.org/document/1000236/
    (segmented_image,labels_image,number_regions) = pms.segment(
            original,
            spatial_radius=SPATIAL_RADIUS,
            range_radius=RANGE_RADIUS,
            min_density=MIN_DENSITY,
            speedup_level=2)
    print("Number of Regions Found: %s" % number_regions)
    unique_labels = np.unique(labels_image)
    blank = original - original
    for label in unique_labels:
        b = random.randint(0,255)
        g = random.randint(0,255)
        r = random.randint(0,255)
        blank[ labels_image == label] = [b,g,r]

    if SHOW == "save":
        cv2.imwrite("saved_segmentation.png",blank)

    allimages["shift segmentation"] = blank
    ################################################################################
    ################################################################################
    ################################################################################
    ################################################################################
    ################################################################################
    if SHOW == True or SHOW == "show":
        root = tk.Tk()
        width = root.winfo_screenwidth()
        height = root.winfo_screenheight()
        def quit():
            root.destroy()
        quit()
        if len(allimages) < 5:
            width = int(width / 2)
            height = int(height / 2)
            x,y = 0,0
            imgCount = 1
            for key,val in allimages.items():
                if imgCount > 2:
                    row = 1
                else:
                    row = 0
                if imgCount % 2 == 1:
                    col = 0
                else:
                    col = 1
                cv2.namedWindow(key,cv2.WINDOW_NORMAL)
                cv2.imshow(key,val)
                cv2.resizeWindow(key,width,height)
                cv2.moveWindow(key, width * col, height * row)
                imgCount += 1

        ########################################################
        #The else isn't ever used but I left it since more images may want to be added during a SHOW
        else:
            width = int(width / 3)
            height = int(height / 3)
            x,y = 0,0
            imgCount =0
            for key,val in allimages.items():
                row = int(imgCount % 3)
                col = int(math.floor(imgCount / 3))
                cv2.namedWindow(key,cv2.WINDOW_NORMAL)
                cv2.resizeWindow(key,width,height)
                cv2.moveWindow(key, width * col, height * row)
                cv2.imshow(key,val)
                imgCount += 1
        cv2.waitKey(0)
        #There is a bug that makes it so that you have to close windows like this on ubuntu 12.10 sometimes.
        #http://code.opencv.org/issues/2911
        cv2.destroyAllWindows()
        cv2.waitKey(-1)

    return original, labels_image


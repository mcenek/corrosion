#############################################################################################################
#Masa Hu
#
#                               Trash Classification Project
#############################################################################################################
#Library imports
import cv2
import numpy as np
import math
import constants
import matplotlib
import matplotlib.patches as mpatches
import gabor_threads_roi as gabor
import gc
import pywt
from matplotlib import pyplot as plt
from pylab import *
from mpl_toolkits.mplot3d import Axes3D
from skimage.feature import hog
from skimage import exposure
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.cluster import MeanShift
from sklearn.cluster import estimate_bandwidth
from sklearn.cluster import DBSCAN
from matplotlib import style

style.use("ggplot")

#############################################################################################################
#                               Description of Module
#
#The output of the exported extract features function is:
#   array([np.array,np.array,np.array], ...]    features
#
#brief description:
#
#this module takes a source image with marked regions and extracts HSV color histograms
#as features for each region defined in markers. The HSV color value [0,0,0] gets dropped
#due to constraints on the opencv calcHist() function which must take a rectangular image
#as the input parameter. Since, marked regions are not rectangular, a copy of the original image
#is used with a particular marking to make an image that is all black except for the specified
#region. This image is then used to extract the histogram distribution of that region. This process
#is repeated until all regions are stored in features.
#
#Setting the show flag allows the user to specify how slowly they would like to see the histogram
#distribution extraction for each region.
#############################################################################################################
#############################################################################################################

def showSegmentSizeDistribution(image,markers):
    #remove markers given condition ange get unique markers again
    for k in size_dict.keys():
        if(size_dict[k] < mean):
            markers[markers == k] = 0
    uniqueMarkers = np.unique(markers)
    reduced_count = len(uniqueMarkers)

    #show the segmenting size selection process
    print("mean size: %s" % mean)
    print("segment counts: %s" % count)
    print("reduced counts: %s" % reduced_count)
    size_array.sort()
    size_hist = np.array(size_array)
    subset = size_hist[size_hist > mean]
    plt.figure(1)
    plt.subplot(211)
    plt.title('size distribution of segments')
    plt.plot(size_hist,'r--')

    plt.subplot(212)
    plt.title('size distribution after reduction')
    plt.plot(subset,'r--')
    plt.pause(0.1)
    cv2.waitKey(0)

#reduces rgb dimension into single dimension using bins
#just for my sanity sake i made this because I thought it might be very cool to see
def showImage3D(image,mode):
    #initialize variables
    height,width = image.shape[:2]
    bins = 8
    X = np.array(range(height))
    Y = np.array(range(width))
    x = []
    y = []
    color = []

    for i in X:
        for j in Y:
            pixel = image[i][j]
            if(isinstance(pixel,np.uint8)):
                val = pixel
                x.append(i)
                y.append(j)
                color.append(val)
            else:
                val = 0
                if(mode == 'b'):
                    val = pixel[0]
                elif(mode == 'g'):
                    val = pixel[1]
                elif(mode == 'r'):
                    val = pixel[2]
                elif(mode == 'bin'):
                    b = pixel[0]
                    g = pixel[1]
                    r = pixel[2]

                    bbin = int(float(b) / float(256) * float(bins))
                    gbin = int(float(g) / float(256) * float(bins))
                    rbin = int(float(r) / float(256) * float(bins))
                    for a,bval in enumerate([bbin,gbin,rbin]):
                        val += bval * pow(bins,a)
                else:
                    val = 1

                x.append(i)
                y.append(j)
                color.append(val)


    #create the x,y,z axis of length x * y
    xcoor = np.array(x)
    ycoor = np.array(y)
    Z = np.array(color)
    #For use with the contour map but I get an error "out of memory" when using meshgrid

    #create the figure
    fig = plt.figure()
    plt.title('3-d visualization of 2d image using color bins')

    #show the 3D scatter plot.
    #if i can ever figure out how to get the countour map to work I will do so.
    ax = plt.axes(projection = '3d')
    ax.view_init(-90, 180)
    ax.scatter(xcoor,ycoor,Z,c=Z,cmap='viridis')
    #ax.contour3D(X,Y,Z,50,cmap='viridis')

    plt.show()

#reduces rgb dimension into single dimension using bins
#just for my sanity sake i made this because I thought it might be very cool to see
'''
inputs:
    1. image to be converted to collection of 3d points
output:
    1. numpy 2d array with collection of 3d points
'''
def cvtImage3DPoints(image,method):
    #initialize variables
    if(len(image.shape) != 3):
        print('invalid image depth! Cannot be greyscaled image')

    #get image shape
    height,width,depth = image.shape
    bins = 8
    X = np.array(range(height))
    Y = np.array(range(width))
    points = np.copy(image)

    #define broadcasting function to extract bgr information
    def foo(pixel,method='binning'):
        val = 0
        b = pixel[0]
        g = pixel[1]
        r = pixel[2]
        print(pixel)
        if(method == 'binning'):

            bbin = int(float(b) / float(256) * float(bins))
            gbin = int(float(g) / float(256) * float(bins))
            rbin = int(float(r) / float(256) * float(bins))
            for a,bval in enumerate([bbin,gbin,rbin]):
                val += bval * pow(bins,a)

        #https://stackoverflow.com/questions/687261/converting-rgb-to-grayscale-intensity
        #uses human perception of color intensities?
        elif(method == 'original'):
            val = 0.2989 * r + 0.5870 * g + 0.1140 * b

        return val

    #broadcast the points and convert 2d image into n feature array of points and return it as a
    #(1,h * w, d) numpy array
    for i in X:
        points[:,i,1] = i
    for j in Y:
        points[:,j,0] = j

    np.vectorize(image,foo)
    #print(points)
    #points[:,:,2] = foo(image,method=method)
    #print(image)
    #print(foo(image))

    return np.reshape(points,(1,height * width,depth))

#reduces rgb dimension into single dimension using bins
#just for my sanity sake i made this because I thought it might be very cool to see
#http://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
'''
inputs:
    1. image to be segmented then shown
output:
    1. collection of points and the labels for each point
'''
def dbscan(image,binning=False):

    if(binning):
        points = cvtImage3DPoints(image,'binning')
    else:
        points = cvtImage3DPoints(image,'original')
    #http://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
    height,width,depth = image.shape
    db = DBSCAN(
            eps=30,
            min_samples=200,
            metric='euclidean',
            metric_params=None,
            algorithm='auto',
            leaf_size=30,
            p=None,
            n_jobs=-1
            )

    db.fit(points)
    labels = db.labels_
    cluster_count = len(np.unique(labels))

    print("Number of estimated clusters: ", cluster_count)

    label_image = np.reshape(labels,(height,width))

    return image,label_image

def quickmeanshift(image):
    mask = quickshift(image)

    return mask

#reduces rgb dimension into single dimension using bins
#just for my sanity sake i made this because I thought it might be very cool to see
#http://scikit-learn.org/stable/modules/generated/sklearn.cluster.MeanShift.html
'''
inputs:
    1. image to be segmented then shown
output:
    1. collection of points and the labels for each point
'''
def meanshift(image,binning=False):

    if(binning):
        points = cvtImage3DPoints(image,'binning')
    else:
        points = cvtImage3DPoints(image,'original')
    #http://scikit-learn.org/stable/modules/generated/sklearn.cluster.MeanShift.html
    height,width,depth = image.shape
    bw = estimate_bandwidth(points,quantile=0.2,n_samples=((height * width) / 2))
    ms = MeanShift(
            bandwidth=100,
            seeds=None,
            bin_seeding=True,
            min_bin_freq=100,
            cluster_all=True,
            n_jobs=-1
            )

    ms.fit(points)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    cluster_count = len(np.unique(labels))

    print("Number of estimated clusters: ", cluster_count)

    label_image = np.reshape(labels,(height,width))

    return image,label_image

#The function extractFeatures() takes in the inputs:
#   Mat         image
#   np.array    markers
#   bool        SHOW
#
#The output of the exported extract features function is a 1-d np array
#
#According to http://stackoverflow.com/questions/17063042/why-do-we-convert-from-rgb-to-hsv/17063317
#HSV is better for object recognition compared to BGR
# H max = 170
# S max = 255
# V max = 255
def extractbinHist(imageIn,SHOW):
    bins = 8
    colors = np.zeros((bins,bins,bins))
    height,width = imageIn.shape[:2]

    #create the histogram of bins^3 colors
    for i in range(height):
        for j in range(width):
            pixel = imageIn[i][j]
            if(isinstance(pixel,np.uint8)):
                b = pixel
                g = pixel
                r = pixel
                bbin = int(float(b) / float(256/bins))
                gbin = int(float(g) / float(256/bins))
                rbin = int(float(r) / float(256/bins))

                colors[bbin][gbin][rbin] += 1
            else:
                b = pixel[0]
                g = pixel[1]
                r = pixel[2]

                bbin = int(float(b) / float(256/bins))
                gbin = int(float(g) / float(256/bins))
                rbin = int(float(r) / float(256/bins))

                colors[bbin][gbin][rbin] += 1

    #flatten the 3-d feature fector into 1-d
    hist = colors.flatten()

    #show the results
    if(SHOW):
        cv2.namedWindow('Processing Segment',cv2.WINDOW_NORMAL)
        cv2.imshow('Processing Segment',imageIn)   #
        plt.plot(hist)
        plt.show()

    #lop off black and white
    return hist[1:-1]


#The function extractFeatures() takes in the inputs:
#   Mat         image
#   np.array    markers
#   bool        SHOW
#
#The output of the exported extract features function is a 1-d np array
#
#According to http://stackoverflow.com/questions/17063042/why-do-we-convert-from-rgb-to-hsv/17063317
#HSV is better for object recognition compared to BGR
# H max = 170
# S max = 255
# V max = 255
def extractColorHist(imageIn,SHOW):

    color = ('b','g','r')
    hist = []
    zeropix = np.count_nonzero(np.all(imageIn == [0,0,0],axis=2))
    for i,col in enumerate(color):
        series = cv2.calcHist([imageIn],[i],None,[256],[0,256])
        series[0] = series[0] - zeropix
        hist.append(np.ravel(series))

    #show the results
    if(SHOW):
        cv2.namedWindow('Processing Segment',cv2.WINDOW_NORMAL)
        cv2.imshow('Processing Segment',imageIn)   #
        plt.plot(hist)
        plt.show()

    #lop off black and white
    return normalize(np.concatenate(np.array(hist)))

#According to http://stackoverflow.com/questions/17063042/why-do-we-convert-from-rgb-to-hsv/17063317
#HSV is better for object recognition compared to BGR
# H max = 170
# S max = 255
# V max = 255
def extractHSVHist(imageIn,SHOW):

    color = ('h','s','v')
    hist = []
    zeropix = np.count_nonzero(np.all(imageIn == [0,0,0],axis=2))
    for i,col in enumerate(color):
        if col == 'h':
            series = cv2.calcHist([imageIn],[i],None,[170],[0,170])
        else:
            series = cv2.calcHist([imageIn],[i],None,[256],[0,256])

        series[0] -= zeropix
        hist.append(np.ravel(series))

    #show the results
    if(SHOW):
        cv2.namedWindow('Processing Segment',cv2.WINDOW_NORMAL)
        cv2.imshow('Processing Segment',imageIn)   #
        plt.plot(hist)
        plt.show()

    #lop off black and white
    return normalize(np.concatenate(np.array(hist)))

#visualte wavelet transform
#https://stackoverflow.com/questions/24536552/how-to-combine-pywavelet-and-opencv-for-image-processing
def visualizeWT(imageIn, show=False):
    imArray = cv2.cvtColor(imageIn,cv2.COLOR_BGR2GRAY )
    #convert to float
    imArray =  np.float32(imArray)
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

    #Display result
    if(show):
        cv2.namedWindow('image',cv2.WINDOW_NORMAL)
        cv2.imshow('image',imArray_H)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

#visualize hog
#http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_hog.html
def visualizeHOG(imageIn):
    fd, hog_image = hog(imageIn, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True, multichannel=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

    ax1.axis('off')
    ax1.imshow(imageIn, cmap=plt.cm.gray)
    ax1.set_title('Input image')

    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('Histogram of Oriented Gradients')
    plt.show()

#https://stackoverflow.com/questions/12729228/simple-efficient-bilinear-interpolation-of-images-in-numpy-and-python
def bilinear_interpolate(im, x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    x0 = np.floor(x).astype(int)
    x1 = x0 + 3
    y0 = np.floor(y).astype(int)
    y1 = y0 + 3

    x0 = np.clip(x0, 0, im.shape[1] - 1);
    x1 = np.clip(x1, 0, im.shape[1] -1 );
    y0 = np.clip(y0, 0, im.shape[0]-1);
    y1 = np.clip(y1, 0, im.shape[0]-1);

    Ia = im[ y0, x0 ]
    Ib = im[ y1, x0 ]
    Ic = im[ y0, x1 ]
    Id = im[ y1, x1 ]

    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    return wa*Ia + wb*Ib + wc*Ic + wd*Id

#extract the edge distribution from the image segment
def extractHOG(imageIn, SHOW):
    #necessary for seeing the plots in sequence with one click of a key

    h,w,d = imageIn.shape
    new_w = (int(int(w) / int(16)) + 1 ) * 16
    new_h = (int(int(h) / int(16)) + 1 ) * 16

    #resize the image to 64 x 128
    resized = cv2.resize(imageIn,(new_w, new_h), interpolation = cv2.INTER_CUBIC)

    #HOG DESCRIPTOR INITILIZATION
    #https://stackoverflow.com/questions/28390614/opencv-hogdescripter-python
    #https://docs.opencv.org/2.4/modules/gpu/doc/object_detection.html
    #https://www.learnopencv.com/histogram-of-oriented-gradients/
    winSize = (new_w,new_h)                               #
    blockSize = (16,16)                             #only 16x16 block size supported for normalization
    blockStride = (8,8)                             #only 8x8 block stride supported
    cellSize = (8,8)                                #individual cell size should be 1/4 of the block size
    nbins = 9                                       #only 9 supported over 0 - 180 degrees
    derivAperture = 1                               #
    winSigma = 4.                                   #
    histogramNormType = 0                           #
    L2HysThreshold = 2.0000000000000001e-01         #L2 normalization exponent ex: sqrt(x^L2 + y^L2 + z^L2)
    gammaCorrection = 0                             #
    nlevels = 64                                    #
    cvhog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                                    histogramNormType,L2HysThreshold,gammaCorrection,nlevels)

    hist = cvhog.compute(resized)

    #create the feature vector
    feature = []
    for i in range(nbins):
        feature.append(0)
    for i in range(len(hist)):
        feature[i % (nbins)] += hist[i]
    feature_hist = np.array(feature)

    #show the results of the HOG distribution for the section
    if(SHOW):
        cv2.namedWindow('Processing Segment',cv2.WINDOW_NORMAL)
        cv2.imshow('Processing Segment',imageIn)   #
        plt.plot(feature_hist)
        plt.draw()
        plt.show()

    norm = normalize(feature_hist.ravel())
    return norm

#blur and erode away non cluster points iteratively
def blurErode(image):
    for i in range(2):
        blur = cv2.medianBlur(image,7)
        kernel = np.ones((3,3),np.uint8)
        #since the rust is black
        dilate = cv2.dilate(blur,kernel,iterations=1)
        image = dilate
        name = 'Iteration: ' + str(i)
        cv2.namedWindow(name,cv2.WINDOW_NORMAL)
        cv2.imshow(name,image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return 1

#get the blob size from the blob
'''
Inputs:
    1. image
Outputs:
    1. blob size
'''
def extractBlobSize(image):
    blob_size = np.count_nonzero(np.all(image != [0,0,0],axis=2))

    return np.array([blob_size])


#applies pca analysis to feature vector with n instances and vector length i
'''
Inputs:
   1. 1D feature vector
Outputs:
    2. new 1D feature vector
'''
def pcaAnalysis(featurevector,SAVEFLAG=True, fnameout='pca_details.txt'):
    instance_count,feature_count = featurevector.shape

    #http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
    #all default values except for n_components
    pca = PCA(copy=True, iterated_power='auto', n_components=0.99, random_state=None,
              svd_solver='auto', tol=0.0, whiten=False)

    #apply pca on the feature vector
    new_vector = pca.fit_transform(featurevector)

    #save pca details to text file
    np.set_printoptions(threshold=np.inf)
    if SAVEFLAG:
        with open(fnameout,'w') as fout:
            fout.write('------------------------------------------------------------------\n')
            fout.write('------------------------------------------------------------------\n')
            fout.write("PCA EXPLAINED VARIANCE RATIO\n\n\n")
            fout.write('' + str(pca.explained_variance_ratio_) + '\n')
            fout.write('------------------------------------------------------------------\n')
            fout.write('------------------------------------------------------------------\n')
            fout.write("PCA SINGULAR VALUES\n\n\n")
            fout.write('' + str(pca.singular_values_) + '\n')
            fout.write('------------------------------------------------------------------\n')
            fout.write('------------------------------------------------------------------\n')
            fout.write("PCA STACKED COVARIANCE VALUES\n\n\n")
            fout.write('' + str(np.sum(pca.get_covariance(),axis=1)) + '\n')
            fout.write('------------------------------------------------------------------\n')
            fout.write('------------------------------------------------------------------\n')
            fout.write("PCA COMPONENTS\n\n\n")
            fout.write('' + str(pca.components_) + '\n')
            fout.write('------------------------------------------------------------------\n')
            fout.write('------------------------------------------------------------------\n')

    #PRINT OUT PCA VALUES to console
    print('-------------------------------------------------------------------------------------')
    print('-------------------------------------------------------------------------------------')
    print('-------------------------------------------------------------------------------------')
    print("PCA EXPLAINED VARIANCE RATIO\n\n\n")
    print('' + str(pca.explained_variance_ratio_) + '\n')
    print('-------------------------------------------------------------------------------------')
    print('-------------------------------------------------------------------------------------')
    print('-------------------------------------------------------------------------------------')
    print("PCA SINGULAR VALUES\n\n\n")
    print('' + str(pca.singular_values_) + '\n')
    print('-------------------------------------------------------------------------------------')
    print('-------------------------------------------------------------------------------------')
    print('-------------------------------------------------------------------------------------')
    print("PCA STACKED COVARIANCE VALUES\n\n\n")
    print('' + str(np.sum(pca.get_covariance(),axis=1)) + '\n')
    print('-------------------------------------------------------------------------------------')
    print('-------------------------------------------------------------------------------------')
    print('-------------------------------------------------------------------------------------')
    print("PCA FEATURE COUNT FOUND\n\n\n")
    print('' + str(len(pca.singular_values_)) + '\n')
    print('------------------------------------------------------------------\n')
    print('------------------------------------------------------------------\n')

    return new_vector

#Writes the features out to a file called extraction_out.txt in the working directory by default
'''
INPUT:
    1. features to write out
    2. (option) file name to write the features to
OUTPUT:
    1. True
'''
def writeFeatures(features, fnameout='output', label=None):

    if len(features) == 0 or type(features) != type(np.array([])):
        print ("features type: %s" % type(features))
        print ("expected type: %s" % type(np.array([])))
        print ("length features: %i" % len(features))
        print ("error with the input to the extractionModule.writeFeatures()")
        return False
    else:
        if label is not None:

            tmp = np.hstack((features,label))
            np.save(fnameout,tmp)

    return True

#Display a histogram
def displayHistogram(hist,normalize=False):
    plt.figure()
    plt.plot(hist)
    plt.show()

#normalize values to max of the set
def normalize(instances):
    norm_instances = instances.astype(np.float) / np.amax(instances)
    return np.nan_to_num(norm_instances)

#get the PCA analysis and fit it to the featurevector of instances
def getLDA(featurevector,labels,featurelength=constants.DECOMP_LENGTH):

    #all default values except for n_components
    lda = LDA()

    lda.fit(featurevector[labels >= 0],labels[labels >= 0])

    return lda

#get the PCA analysis and fit it to the featurevector of instances
def getPCA(featurevector,featurelength=constants.DECOMP_LENGTH):

    #http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
    #all default values except for n_components
    pca = PCA(n_components=featurelength)

    #apply pca on the feature vector
    newfeatures = pca.fit_transform(featurevector)

    return newfeatures, pca

# show the 1st and 2nd component on a graph with labels
def showPCA(featurevector,labels,featurelength=constants.DECOMP_LENGTH,pcaobject=None):

    if pcaobject == None:
        newfeatures, pca = getPCA(featurevector,featurelength=featurelength)
    else:
        pca = pcaobject
        newfeatures = pca.transform(featurevector)

    print("features reduced from %i to %i" % (len(featurevector[0]),pca.n_components_))

    x = newfeatures[:,0]
    y = newfeatures[:,1]
    clabel = np.zeros(labels.shape[0])
    tmp = labels.reshape((labels.shape[0]))
    clabel[tmp == 0] = 0
    clabel[tmp == 1] = 1
    clabel[tmp == 2] = 2
    clabel[tmp == 3] = 3
    clabel[tmp == 4] = 4
    clabel[tmp == 5] = 5
    clabel[tmp == -1] = 6

    colors = ['red','green','blue','yellow','magenta','cyan','black']

    fig = plt.figure()
    plt.scatter(x,y,c=clabel,cmap=matplotlib.colors.ListedColormap(colors))

    red_patch = mpatches.Patch(color='red',label='treematter')
    green_patch = mpatches.Patch(color='green',label='plywood')
    blue_patch = mpatches.Patch(color='blue',label='cardboard')
    yellow_patch = mpatches.Patch(color='yellow',label='bottles')
    magenta_patch = mpatches.Patch(color='magenta',label='trashbag')
    cyan_patch = mpatches.Patch(color='cyan',label='blackbag')
    black_patch = mpatches.Patch(color='black',label='mixed')
    plt.legend(handles=[red_patch,green_patch,blue_patch,yellow_patch,magenta_patch,cyan_patch,black_patch])

    plt.show()

# show the 1st and 2nd component on a graph with labels
def showLDA(featurevector,labels,classes='all',mode='int',ldaobject=None):
    #create our color legend
    colors = ['red','green','blue','yellow','magenta','cyan','black']
    allclasses = [0,1,2,3,4,5,6]
    tmp = labels.reshape((labels.shape[0]))

    #-1 is mixed blobs
    labelID = np.full(labels.shape[0],-1)

    if mode == 'binary':
        labelID[tmp == 0] = 0
        labelID[tmp == 1] = 1
    elif mode == 'string':
        labelID[tmp == 'treematter'] = 0
        labelID[tmp == 'plywood'] = 1
        labelID[tmp == 'cardboard'] = 2
        labelID[tmp == 'bottles'] = 3
        labelID[tmp == 'trashbag'] = 4
        labelID[tmp == 'blackbag'] = 5
        labelID[tmp == 'mixed'] = 6
    elif mode == 'int':
        labelID[tmp == 0] = 0
        labelID[tmp == 1] = 1
        labelID[tmp == 2] = 2
        labelID[tmp == 3] = 3
        labelID[tmp == 4] = 4
        labelID[tmp == 5] = 5
        labelID[tmp == -1] = 6

    if ldaobject == None:
        lda = getLDA(featurevector[labelID < 6],labelID[labelID < 6])
        newfeatures = lda.transform(featurevector)
    else:
        lda = ldaobject
        newfeatures = lda.transform(featurevector)

    print('lda acquired')
    print('new features extracted')

    x = newfeatures[:,0]
    y = newfeatures[:,1]

    #create our figure
    fig = plt.figure()

    #scatter the x,y coordinates with our color legend on the labels
    plt.scatter(x,y,c=labelID,cmap=matplotlib.colors.ListedColormap(colors))

    #show visual legend map on top right of screen
    red_patch = mpatches.Patch(color='red',label='treematter')
    green_patch = mpatches.Patch(color='green',label='plywood')
    blue_patch = mpatches.Patch(color='blue',label='cardboard')
    yellow_patch = mpatches.Patch(color='yellow',label='bottles')
    magenta_patch = mpatches.Patch(color='magenta',label='trashbag')
    cyan_patch = mpatches.Patch(color='cyan',label='blackbag')
    black_patch = mpatches.Patch(color='black',label='mixed')
    plt.legend(handles=[red_patch,green_patch,blue_patch,yellow_patch,magenta_patch,cyan_patch,black_patch])

    #show plot
    plt.show()

# show the 1st and 2nd component on a graph with labels
def showLDA2(featurevector,labels,classes='all',featurelength=constants.DECOMP_LENGTH):
    #create our color legend
    colors = ['red','green','blue','yellow','magenta','cyan']
    allclasses = [0,1,2,3,4,5]
    tmp = labels.reshape((labels.shape[0]))
    labelID= np.full(labels.shape[0],-1)
    labelID[tmp == 'treematter'] = 0
    labelID[tmp == 'plywood'] = 1
    labelID[tmp == 'cardboard'] = 2
    labelID[tmp == 'bottles'] = 3
    labelID[tmp == 'trashbag'] = 4
    labelID[tmp == 'blackbag'] = 5

    if classes == 'all':
        lda = getLDA(featurevector,labelID)
        newfeatures = lda.transform(featurevector)

        x = newfeatures[:,0]
        y = newfeatures[:,1]
    else:
        tree = 'tree' in classes
        plywood = 'ply' in classes
        cardboard = 'cardboard' in classes
        bottles = 'bottle' in classes
        trashbag = 'trashbag' in classes
        blackbag = 'black' in classes
        flags = [tree,plywood,cardboard,bottles,trashbag,blackbag]
        names = ['treematter','plywood','cardboard','bottles','trashbag','blackbag']

        tmp_labels = np.full(labels.shape[0],-1)
        tmp_colors = []
        i = 0
        for flag,cat,col in zip(flags,names,colors):
            if flag:
                tmp_labels[tmp == cat] = i
                tmp_colors.append(col)
                i += 1

        tmp_labels[tmp == -1] = i
        tmp_colors.append(col)
        tmp_labels = tmp_labels.astype(int)

        #fit lda on the instances and their labels removing instances without the labels
        lda = getLDA(featurevector,tmp_labels)

        #transform the dataset
        newfeatures = lda.transform(featurevector)
        if newfeatures.shape[1] >= 2:
            x = newfeatures[:,0]
            y = newfeatures[:,1]
        else:
            x = newfeatures
            y = np.zeros(x.shape[0])

        labelID = tmp_labels
        colors = tmp_colors

    #print out number of features reduced
    #score = lda.score(tmp_instances,labelID)
    #print("LDA SCORE: %f" % score)

    #create our figure
    fig = plt.figure()

    #scatter the x,y coordinates with our color legend on the labels
    plt.scatter(x,y,c=labelID,cmap=matplotlib.colors.ListedColormap(colors))

    #show visual legend map on top right of screen
    red_patch = mpatches.Patch(color='red',label='treematter')
    green_patch = mpatches.Patch(color='green',label='plywood')
    blue_patch = mpatches.Patch(color='blue',label='cardboard')
    yellow_patch = mpatches.Patch(color='yellow',label='bottles')
    magenta_patch = mpatches.Patch(color='magenta',label='trashbag')
    cyan_patch = mpatches.Patch(color='cyan',label='blackbag')
    plt.legend(handles=[red_patch,green_patch,blue_patch,yellow_patch,magenta_patch,cyan_patch])

    #show plot
    plt.show()


#takes a single image and extracts all features depending on flag constants
#based on user input
'''
INPUTS:
    1. segment of type numpy array
    2. (optional) hogflag  of type bool
    3. (optional) gaborflag of type bool
    4. (optional) colorflag of type bool
OUTPUTS:
    1. feature vector
'''
def evaluateSegment(segment,hogflag=False,gaborflag=False,colorflag=False,sizeflag=False,hsvflag=False):
    #extract features for each image depending on the flag constants
    features = []

    if colorflag:
        features.append(evaluate(segment,'color'))
    if gaborflag:
        features.append(evaluate(segment,'gabor'))
    if hogflag:
        features.append(evaluate(segment,'hog'))
    if hsvflag:
        features.append(evaluate(segment,'hsv'))
    if sizeflag:
        features.append(evaluate(segment,'size'))

    #create the full feature vector for the given instance image and push to instances
    #and also push the file name as the label for the instance
    full_vector = np.array([])
    for i in range(len(features)):
        full_vector = np.hstack((full_vector,features[i]))

    return full_vector

#EVALUATE AN IMAGE GIVEN THE MODE
def evaluate(original,mode,SHOWFLAG=False):
    #check if the image was read in correctly
    if original is None:
        print('invalid image! Could not open image')

    #if mode is size we have to normalize this later across all instances
    if mode == 'size':
        combined_filename = sys.argv[1]

        # Generate and save blob size for this blob we assume black as background
        size = extractBlobSize(original)
        #print('--------------SIZE---------------')
        return size

    #if mode is hog, show hog feature vector of image
    elif mode == 'hog':
        hist = extractHOG(original,False)
        featurevector = hist.flatten()
        norm = normalize(featurevector)
        return norm

    #if mode is color, show color histogram of image
    elif mode == 'color':
        hist = extractColorHist(original,False)
        norm = normalize(hist)
        #print('-------------Color----------------')
        #print(norm)
        return norm

    #if mode is gabor, extract gabor feature from image using several orientations
    elif mode == 'gabor':
        orientations = 16
        filters = gabor.build_filters(orientations)
        combined_filename = sys.argv[1]

        # Generate and save ALL hogs for this image
        result = gabor.run_gabor(original, filters, combined_filename, orientations, mode='training')
        featurevector = result.flatten()[1:]
        norm = normalize(featurevector)
        #print('--------------Gabor---------------')
        #print(norm)
        return norm

    elif mode == 'hsv':
        hsvimg = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
        hist = extractHSVHist(hsvimg,False)
        norm = normalize(hist)
        return norm



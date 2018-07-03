
#Imports
import segmentModule
import numpy as np
import sys
import cv2

#Constants
INGROUP=np.array([255,255,255],np.uint8)        #White
OUTGROUP=np.array([0,0,0],np.uint8)             #Black
UNKNOWN=np.array([125,125,125],np.uint8)        #Grey
GROUP1=np.array([0,0,255],np.uint8)             #Red
GROUP2=np.array([0,255,0],np.uint8)             #Green
GROUP3=np.array([255,0,0],np.uint8)             #Blue
GROUP4=np.array([255,255,0],np.uint8)           #Turquoise
THRESHHOLD=1.0

COLORS=["red","green","blue","turquoise","grey"]
GNAMES = ["construction waste", "tree matter", "plywood", "cardboard"]
GROUPS = [GROUP1,GROUP2,GROUP3,GROUP4,UNKNOWN]
SINGLE = [INGROUP,OUTGROUP,UNKNOWN]

#Local functions

#findMax
# - Finds the index of the maximum value in the list
def findMax(args):
    if type(args) != type([]):
        return -1

    index = 0
    tmp = args[0]
    for i,a in enumerate(args):
        if a > tmp:
            index = i
            tmp = a

    return index

def findMin(args):
    if type(args) != type([]):
        return -1

    index = 0
    tmp = args[0]
    for i,a in enumerate(args):
        if a < tmp:
            index = i
            tmp = a

    return index

#main program
#When using a single classification
if len(sys.argv) == 3:
    imageFileIn = sys.argv[1]
    classificationsIn = sys.argv[2]

    #initialize image, markers using segmentModule
    #initialize classifications using classificationsIn
    #segmentModule.getSegments always produces the same result so this works. Since classification for each segment is known using same function in execute.py.
    original = segmentModule.normalizeImage(imageFileIn)
    image, markers = segmentModule.getSegments(original, False)
    uniquemarkers = np.unique(markers)
    classifications = []
    with open(classificationsIn,'r') as cin:
        lines = cin.read().splitlines()
        for l in lines:
            classifications.append(float(l))

    if len(classifications) == len(uniquemarkers):

        blank = image.copy()
        blank = blank - blank
        blank[markers == -1] = UNKNOWN
        for c,um in zip(classifications,uniquemarkers):
            if  c > 0:
                blank[markers == um] = INGROUP
            elif c <= 0:
                blank[markers == um] = OUTGROUP

        total = 0
        pixcounts = []
        for group in SINGLE:
            tmp = cv2.inRange(blank,group,group)
            num = cv2.countNonZero(tmp)
            pixcounts.append(num)
            total += num

        percent1 = float(pixcounts[0]) / float(total) * 100
        percent2 = float(pixcounts[1]) / float(total) * 100
        percent3 = float(pixcounts[2]) / float(total) * 100

        cv2.namedWindow(imageFileIn,cv2.WINDOW_NORMAL)
        cv2.namedWindow("original", cv2.WINDOW_NORMAL)
        print ""
        print("ingroup white:  %.5f%%" % percent1)
        print("outgroup black: %.5f%%" % percent2)
        print("unkown grey:    %.5f%%" % percent3)
        print ""

        cv2.imshow("original",image)
        cv2.imshow(imageFileIn,blank)
        cv2.waitKey(0)

    else:
        print "Are you sure the classification file is for that image?"

#When using multiple classifications. No more than 6
elif len(sys.argv) > 3 and len(sys.argv) <= 7:
    #get command line arguments
    imageFileIn = sys.argv[1]
    classification_names = []
    for fname in sys.argv[2:]:
        classification_names.append(fname)

    #recreate markers
    original = segmentModule.normalizeImage(imageFileIn)
    image, markers = segmentModule.getSegments(original, False)
    uniquemarkers = np.unique(markers)

    #get classfications for segments from command line arguments
    classifications = []
    for fname in classification_names:
        classifications.append([])
        with open(fname,'r') as fin:
            lines = fin.read().splitlines()
            for l in lines:
                classifications[-1].append(float(l))

    same = True
    for c in classifications:
        for d in classifications:
            if c != d:
                if len(c) != len(d):
                    same = False

    #If classifications are for the same image, then start stitching according to best classifier.
    if same:
        #make blank image
        blank = image.copy()
        blank = blank - blank
        blank[markers == -1] = UNKNOWN

        #Color using max of the classifications. The max according to the svm is the one farthest from the Support vector line separating the different 2 group classification. The most positive is taken
        #Handles up to 5 categories
        length = len(classifications)
        for index,um in enumerate(uniquemarkers):
            tmp = []
            #Go through each classification for that segment and record it into tmp
            for i in range(length):
                tmp.append(classifications[i][index])

            max_index = findMax(tmp)
            min_index = findMin(tmp)

            #Check to see if a high svm score is found. Otherwise make it 0
            #if (max_index - min_index) > THRESHHOLD:
                #color according to the group index
            blank[markers == um] = GROUPS[max_index]
            #else:
            #    blank[markers == um] = GROUPS[-1]

        pixcounts = []
        total = 0
        for g in GROUPS:
            tmp = cv2.inRange(blank,g,g)
            num = cv2.countNonZero(tmp)
            pixcounts.append(num)
            total += num

        percents = []
        for count in pixcounts:
            percents.append(float(count) / float(total) * 100)

        cv2.namedWindow(imageFileIn,cv2.WINDOW_NORMAL)
        cv2.namedWindow("original", cv2.WINDOW_NORMAL)

        #Print the group id, color, and its pixel percentage
        print ""
        for i,p,c in zip(GNAMES,percents,COLORS):
            print ("%s (%s): %.5f%%" % (i,c,p))

        print ""
        cv2.imshow("original",image)
        cv2.imshow(imageFileIn,blank)
        cv2.waitKey(0)

    else:
        print "classifications are not the same length"

else:
    print "wrong number of arguments passed. Expecting 3 or 6:"
    print "arg1 = imageFileIn"
    print "arg2 = classification1"
    print "arg3 = classification2"
    print "arg4 = classification3"
    print "arg5 = classification4"

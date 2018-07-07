
import cv2
import numpy as np
import os
import sys

BORDER = 15

#get dice score for binary image
def dice(img,gt,fout='dice_output.txt',writemode='w'):
    h1,w1 = img.shape
    h2,w2 = gt.shape

    TP = float(np.count_nonzero(cv2.bitwise_and(cv2.bitwise_not(img),cv2.bitwise_not(gt))))
    TN = float(np.count_nonzero(cv2.bitwise_and(img,gt)))
    FP = float(np.count_nonzero(cv2.bitwise_and(img,cv2.bitwise_not(gt))))
    FN = float(np.count_nonzero(cv2.bitwise_and(cv2.bitwise_not(img),gt)))
    P = TP + FN
    N = TN + FP

    #for debugging purposes
    '''
    binary1 = np.count_nonzero(np.all(img == 0,axis = 1),dtype=np.uint8)
    binary2 = np.array(np.all(gt == 0,axis = 1),dtype=np.uint8)
    b1_and_b2 = np.array(np.logical_and(binary1,binary2),dtype=np.uint8)
    binary1[binary1 == 1] = 255
    binary2[binary2 == 1] = 255
    b1_and_b2[b1_and_b2 == 1] = 255
    cv2.imshow('binary1',cv2.resize(binary1,(500,500),interpolation=cv2.INTER_CUBIC))
    cv2.imshow('binary2',cv2.resize(binary2,(500,500),interpolation=cv2.INTER_CUBIC))
    cv2.imshow('b1_and_b2',cv2.resize(b1_and_b2,(500,500),interpolation=cv2.INTER_CUBIC))
    cv2.waitKey(0)
    '''

    PREC = (TP) / (TP + FP)
    ACC = (TP + TN) / (P + N)
    SENS= TP / P
    SPEC= TN / N

    DICE = TP / (P + FP)

    print('True Positive: %f' % TP)
    print('True Negative: %f' % TN)
    print('False Positive: %f' % FP)
    print('False Negative: %f' % FN)
    print('Positive: %f' % P)
    print('Negative: %f\n' % N)
    print('PRECICSION: %f' % PREC)
    print('ACCURACY: %f' % ACC)
    print('SENSITIVITY: %f' % SENS)
    print('SPECIFICITY: %f' % SPEC)
    print('ACCURACY: %f' % ACC)
    print('DICE: %f' % DICE)
    print('--------------')

    with open(fout,writemode) as fo:
        fo.write('----------------\n\n\n')
        fo.write('True Positive: %f\n' % TP)
        fo.write('True Negative: %f\n' % TN)
        fo.write('False Positive: %f\n' % FP)
        fo.write('False Negative: %f\n' % FN)
        fo.write('Positive: %f\n' % P)
        fo.write('Negative: %f\n\n' % N)
        fo.write('PRECICSION: %f\n' % PREC)
        fo.write('SENSITIVITY: %f\n' % SENS)
        fo.write('SPECIFICITY: %f\n' % SPEC)
        fo.write('ACCURACY: %f\n' % ACC)
        fo.write('DICE: %f\n\n\n' % DICE)
        fo.write('---------------------------------------------------\n')

    return ACC,DICE

#main function
if __name__ == '__main__':
    #maker sure of correct sys args

    #WHEN DICING A directory
    if len(sys.argv) == 3 and os.path.isdir(sys.argv[1]):
        dir1 = sys.argv[1]
        dir2 = sys.argv[2]
        if not os.path.isdir(dir1):
            print('%s is not a directory!' % dir1)
            sys.exit()
        if not os.path.isfile(dir2):
            print('$s is not a file!' % dir2)
            sys.exit()
        #output to results directory
        if not os.path.exists('results'):
            os.makedirs('results')

    #WHEN DICING A SINGLE IMAGE
    elif len(sys.argv) == 3 and os.path.isfile(sys.argv[1]):
        img1 = sys.argv[1]
        img2 = sys.argv[2]

        #check if imgectory exists then read the images
        if os.path.exists(img1) and os.path.exists(img2):

            #read the image
            img = cv2.imread(img1,0)
            gt = cv2.imread(img2,0)

            #make sure the shapes match
            h,w = img.shape
            h2,w2 = gt.shape
            if h != h2 or w != w2:
                gt = cv2.resize(gt,(h,w),interpolation = cv2.INTER_CUBIC)

            #make the image binary just in case it turns gray during resizing or the image just is not binary
            gt[gt != 255] = 0
            img[img != 255] = 0

            #make the output directory
            if not os.path.exists('results'):
                os.makedirs('results')
            #create file name
            fname = "RESULTS_" + str(os.path.splitext(os.path.basename(sys.argv[1]))[0]) + ".txt"
            fout = os.path.join('results',fname)

            #apply dice score calculation while removing the border
            h_low = BORDER
            h_high = h - BORDER - 1
            w_low = BORDER
            w_high = w - BORDER - 1
            acc_score,dice_score = dice(img[h_low:h_high,w_low:w_high] ,gt[h_low:h_high,w_low:w_high],fout,writemode='w')

        else:
            print("PATH DOES NOT EXIST: \n\t%s, \n\t%s" %(sys.argv[1],sys,argv[2]))
            sys.exit()
    else:
        print("wrong number of arguments")
        print("expecting 2")
        print("python dice.py [segmentation] [ground truth]")


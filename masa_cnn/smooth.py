import cv2
import numpy as np
import sys
import os



#blur and erode away non cluster points iteratively
def blurErode(image):
    for i in range(2):
        blur = cv2.medianBlur(image,7)
        kernel = np.ones((3,3),np.uint8)
        #since the rust is black
        dilate = cv2.dilate(blur,kernel,iterations=1)
        image = dilate
        name = 'Iteration: ' + str(i)

        #cv2.namedWindow(name,cv2.WINDOW_NORMAL)
        #cv2.imshow(name,image)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

    return image

if __name__ == '__main__':
    out_dir = 'smooth_images'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    #run on entire directory
    if len(sys.argv) == 2 and os.path.isdir(sys.argv[1]):
        for f in os.listdir(sys.argv[1]):
            fullpath = os.path.join(sys.argv[1],f)
            img = cv2.imread(fullpath,cv2.IMREAD_GRAYSCALE)
            smoothed = blurErode(img)
            outfile = os.path.join(out_dir,os.path.splitext(f)[0] + '.png')
            cv2.imwrite(outfile,smoothed)

    #run smoothing on single image
    elif len(sys.argv) == 2 and os.path.isfile(sys.argv[1]):
        img = cv2.imread(sys.argv[1],cv2.IMREAD_GRAYSCALE)
        smoothed = blurErode(img)
        outfile = os.path.join(out_dir,'smoothed_' + os.path.splitext(os.path.basename(sys.argv[1]))[0] + '.png')
        cv2.imwrite(outfile,smoothed)

    #some error handling
    else:
        print("error reading input")
        print('python smooth.py [dir]')
        print('python smooth.py [img]')



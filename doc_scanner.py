import cv2 as cv
import numpy as np

from utility import getcontours,reorder,warp_doc

IMAGE_PATH = 'samples/sample1.jpg'
OUTPUT_PATH = 'output/scanned_sample1.jpg'

kernel = np.ones((5,5))
img = cv.imread(IMAGE_PATH)

imgc=img.copy()
gframe = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
gframe = cv.GaussianBlur(gframe,(5,5),1)
canny = cv.Canny(gframe,10,200)
dila = cv.dilate(canny,kernel,iterations=2)
erode = cv.erode( dila,kernel,iterations=1)
biggest = getcontours(erode)
if biggest is not None :
    Doc = warp_doc(img,biggest)
    cv.imwrite(OUTPUT_PATH, Doc)
    print('scanned document saved at{OUTPUT_PATH}')
else :
    print('could not find contour')



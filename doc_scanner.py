import cv2 as cv
import numpy as np
import os

from utility import getcontours,reorder,warp_doc

input_folder = 'samples'
output_folder = 'output'

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

kernel = np.ones((5,5))


# Supported image file extensions
image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')

for filename in os.listdir(input_folder):
    if filename.lower().endswith(image_extensions):
        IMAGE_PATH = os.path.join(input_folder, filename)
        OUTPUT_PATH = os.path.join(output_folder, filename)

        img = cv.imread(IMAGE_PATH)
        if img is not None:
            imgc=img.copy()
            #preprocessing 
            gframe = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
            gframe = cv.GaussianBlur(gframe,(5,5),1)
            canny = cv.Canny(gframe,10,200)
            dila = cv.dilate(canny,kernel,iterations=2)
            erode = cv.erode( dila,kernel,iterations=1)
            biggest = getcontours(erode)
            if biggest is not None :
                #Warping Doc 
                Doc = warp_doc(img,biggest)
                cv.imwrite(OUTPUT_PATH, Doc)
                print('scanned document saved at output folder')
            else :
                print('could not find Document')
        else:
            print(f"Failed to read: {IMAGE_PATH}")

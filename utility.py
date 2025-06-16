import cv2 as cv
import numpy as np

width = 480
height = 640

def getcontours(img):
    contours,hierarchy = cv.findContours(img,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    max_area = 0
    biggest = np.array([])
    for contour in contours:
        area = cv.contourArea(contour)
        if area >5000 :
            perimeter = cv.arcLength(contour,True)
            approx = cv.approxPolyDP(contour,0.02*perimeter,True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    cv.drawContours(img,biggest,-1,(0,255,0),15)
    return biggest

def reorder(points):
    points = points.reshape((4,2))
    pts = np.zeros((4,2),np.float32)
    add = np.sum(points,axis=1)
    # print("add",add)
    diff = np.diff(points,axis=1)
    pts[0] = points[np.argmin(add)]
    pts[1] = points[np.argmin(diff)]
    pts[2] = points[np.argmax(diff)]
    pts[3] = points[np.argmax(add)]

    return pts 
def warp_doc(img,biggest):
    if len(biggest) == 0:
        cv.putText(img, "No document detected", (50, 50), 
                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return img

    biggest=reorder(biggest)
    img_corners = np.float32(biggest)
    scanned_corners = np.float32([[0,0],[width,0],[0,height],[width,height]])
    matrix = cv.getPerspectiveTransform(img_corners,scanned_corners)
    Doc = cv.warpPerspective(img,matrix,(width,height))
    return Doc 

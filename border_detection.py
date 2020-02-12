import cv2
import numpy as np

def border(img):
    edges = cv2.Canny(img,50,150, apertureSize = 3)
    # cv2.imwrite('edges-50-150.jpg',edges)
    minLineLength=100
    lines = cv2.HoughLinesP(image=edges,rho=1,theta=np.pi/180, threshold=100,lines=np.array([]), minLineLength=minLineLength,maxLineGap=80)
    return lines
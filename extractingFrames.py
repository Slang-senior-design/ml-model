import cv2
import math
from os import listdir
import pandas as pd 
import csv
import numpy as np
from sklearn import preprocessing
import sys
import border_detection
m = "mapping.csv"

# img => class
def extract(folder):    
    for mov in listdir(folder):
        extractvid(folder, folder+"/"+mov)
    
def crop(img):
    lines = border_detection.border(img)

    if(lines is None): return img
    max_y = np.max([lines[:,[0],[0]],lines[:,[0],[2]]])
    max_x = np.max([lines[:,[0],[1]],lines[:,[0],[3]]])
    min_y = np.min([lines[:,[0],[0]],lines[:,[0],[2]]])
    min_x = np.min([lines[:,[0],[1]],lines[:,[0],[3]]])
    cropped = img[min_x:max_x, min_y:max_y]
    return cropped

def resize(img, desired_size):
    old_size = img.shape[:2] # old_size is in (height, width) format

    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    # new_size should be in (width, height) format
    resized = cv2.resize(img, (new_size[1], new_size[0]))
    return resized

def fill_border(img, desired_size):
    new_size = img.shape
    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    counts = np.bincount(img.flatten())
    fill = int(np.argmax(counts))
    background = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,
    value=[fill, fill, fill])
    return background

def show(label, img):
    cv2.imshow(label, img) 
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def extractvid(sign, vid):
    with open(m, 'a') as f:
        cap = cv2.VideoCapture(vid)
        frameRate = cap.get(5) #fps

        while(True):
            frameId = cap.get(1)
            ret, frame = cap.read()
            
            if(ret != True):
                break
                
            if(frameId % math.floor(frameRate) == 0):
                mid = (int) (frame.shape[0]/2)
                top_border, right_border, bottom_border, left_border = 15, 15, 15, 15
                frame = frame[top_border:mid-bottom_border,right_border:(-1)*right_border] # top half, no label, no bottom border, no right border
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # show("frame", frame)

                ###
                cropped = crop(frame)
                # show("crop", cropped)

                # reshaping
                desired_size = 100
                resized = resize(cropped, desired_size)

                # show("resize", resized)
                
                background = fill_border(resized, desired_size)

                # frame /= 255
                # frame = preprocessing.scale(frame, axis=1) # mean = 0,  
                
                # show("border", background)
                ###

                n = np.array(background)
                x, y = background.shape
                n = np.reshape(n, (1,x*y))
                f.write(sign + ",")
                np.savetxt(f, n, delimiter=",", fmt="%d")
        cap.release()
    f.close()

extract("alone")
extract("twenty")
extract("lonely")
# extract("bachelor")

# extractvid("bachelor", "bachelor\Brady_15729.mov")

if(len(sys.argv) > 1):
    extract(sys.argv[1])
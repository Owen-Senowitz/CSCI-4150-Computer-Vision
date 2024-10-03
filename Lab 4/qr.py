"""
This file was written to help decode a QR code.  It utilizes the zxing.org API to do the trick. 
The zxing.org API returns an HTML page with a table in it to display results.
In order to parse this table and extract the message, the library beautifulsoup was used.
You will need to install this via the command: conda install beautifulsoup4.
"""
from bs4 import BeautifulSoup
import json
import os
import requests
import matplotlib.pyplot as plt
import numpy as np

class Point():
    def __init__(self,x,y):
        self.x = x
        self.y = y

    def __str__(self):
        return "({},{})".format(self.x,self.y)
    
    def __repr__(self):
        return self.__str__()

def getCornerPoints(img):

    # Format image
    if np.amax(img) > 1:
        img = img / 255.0

    if img.ndim == 3:
        img = img[:,:,0]

    #Threshhold for pure black and white
    img = img > 0.5

    rows, cols = img.shape

    candidates = getCandidates(img)

    centers = []
    
    for c in candidates:
        if verifyCenter(img,c):
            centers.append(c)

    centers = neighborSuppression(centers,thresh=rows/5)

    pt = getAlignmentPoint(img,centers)

    centers.append(pt)
    point_list = [Point(c[1],c[0]) for c in centers]

    return point_list



def checkRatio(c1,c2,c3,c4,c5, thresh = 2.0, debug=False):
    
    total = (c1+c2+c3+c4+c5)/7.0
    
    if debug:
        print(c1,c2,c3,c4,c5)
        print(total)
        print(abs(c1 - total),abs(c2 - total),abs(c3 - 3*total),abs(c4 - total),abs(c5 - total))
    
    # Left edge check
    if total < 1.0:
        return False
    
    if abs(c1 - total) < thresh and abs(c2 - total) < thresh and abs(c3 - 3*total) < thresh and abs(c4 - total) < thresh and abs(c5 - total) < thresh:
        if debug:
            print("Cleared")
        return True
    else:
        return False

def getCandidates(image):

    rows, cols = image.shape
    candidates = []
    
    for i in range(rows):
        
        counter1 = 0
        counter2 = 0
        counter3 = 0
        counter4 = 0
        counter5 = 0
        
        #debug = False
        
        # Black = False, White = True
        prev = True
        
        for j in range(cols):
            
            # We are at a white pixel
            if image[i,j]:
                # We were previously at a white pixel
                if prev:
                    counter4 += 1
                        
                # We were previously at a black pixel
                else:
                    if checkRatio(counter1,counter2,counter3,counter4,counter5):
                        candidates.append((i,j-counter5-counter4-counter3//2))
                        candidates.append((i,j-counter5-counter4-counter3//2 + 1))
                    
                    counter2 = counter4
                    counter4 = 1
                
                prev = True
                
            # We are at a black pixel
            else:
                # We were previously at a white pixel
                if prev:
                    counter1 = counter3
                    counter3 = counter5
                    counter5 = 1
                
                # We were previously at a black pixel
                else:
                    counter5 += 1
                    
                prev = False
                
        # Right edge check
        if checkRatio(counter1,counter2,counter3,counter4,counter5):
            candidates.append((i,j+1-counter5-counter4-counter3//2))
            candidates.append((i,j+1-counter5-counter4-counter3//2 + 1))
            
    return candidates

def verifyRatio(c1,c2,c3,thresh=1.0):
    
    total = (c1+c2+c3)/3.5

    if abs(c1 - 1.5*total) < thresh and abs(c2 - total) < thresh and abs(c3 - total) < thresh:
        return True
    else:
        return False

def verifyDirection(image,loc,dx=0,dy=1):
        
    prev = 0
    counter1 = 0
    counter2 = 0
    counter3 = 0
    rows,cols = image.shape
    i,j = loc
    while i>=0 and i<rows and j>=0 and j<cols:
        
        # We are at a white pixel
        if image[i,j]:
            if prev < 2:
                counter2 += 1
            
            # We were previously at a black pixel and we are at the end
            else:
                return verifyRatio(counter1,counter2,counter3)
                
            prev = 1

        # We are at a black pixel
        else:
            if prev==0:
                counter1 += 1 
                
            # We were previously at a white pixel
            else:
                counter3 += 1
                prev = 2
        
        i += dy
        j += dx
        
    # Edge Check
    return verifyRatio(counter1,counter2,counter3)


def verifyCenter(image, loc):
    check_up = verifyDirection(image,loc,0,-1)
    check_down = verifyDirection(image,loc,0,1)
    check_diag = verifyDirection(image,loc,1,1)
    
    return check_up and check_down and check_diag

def neighborSuppression(centers,thresh = 10.0):

    i=0
    while i < len(centers):
        c1 = centers[i]

        removal_list = []

        for j in range(i+1,len(centers)):
            c2 = centers[j]
            if (c1[0] - c2[0])**2 + (c1[1]-c2[1]) ** 2 < thresh:
                removal_list.append(j)

        # Remove indices in reverse order to not mess up indexing
        removal_list = removal_list[::-1]
        for j in removal_list:
            del centers[j]

        i += 1

    return centers

def getAlignmentPoint(image,centers):

    top_left = centers[0]
    top_right = centers[1]
    bottom_left = centers[2]
    
    v1 = (top_right[0] - top_left[0], top_right[1] - top_left[1])
    v2 = (bottom_left[0] - top_left[0], bottom_left[1] - top_left[1])
    
    x = top_left[1] + v1[1] + v2[1]
    y = top_left[0] + v1[0] + v2[0]

    return (y,x)


def decode(arr):
    """
    Function: decode
    ----------------
    decodes a numpy arrayified QR code

    Parameters:
    -----------
    arr: the numpy array

    Returns:
    --------
    the parsed message
    """
    rows,cols = arr.shape
    if rows > 41 or cols > 41:
        print("Array is too large. Data appears to not have been properly discretized.")
        return None
    
    if arr.dtype != 'bool':
        print("Array is not type Boolean. Data appears to not have been properly discretized.")
        return None
    
    
    
    plt.imsave('tmp.png', arr)
    with open("tmp.png", "rb") as f:
        response = requests.post('https://zxing.org/w/decode', files=dict(f=f))
    os.remove("tmp.png")
    html = response.text
    table_data = [[cell.text for cell in row("td")] for row in BeautifulSoup(html, 'html.parser')("tr")]
    obj = dict(table_data)
    return obj['Parsed Result']
import numpy as np
import cv2
import os

def best_match(img):
    maxval = 0
    label = ""
    tHmax = 0
    tWmax = 0
    maxLocmax = 0
    for _, dirs, _ in os.walk("templates_examples/"):
        for dir in dirs:
            for _, _, files in os.walk("templates_examples/"+str(dir)):
                for file in files:
                    template = cv2.imread("templates_examples/"+str(dir) + "/"+ str(file))
                    #cv2.imshow("template", template)
                    #cv2.waitKey(0)
                    template = preprocess(template)


                    result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
                    (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
                    (tH, tW) = template.shape[:2]

                    if maxVal > maxval:
                        maxval = maxVal
                        tHmax = tH
                        tWmax = tW
                        maxLocmax = maxLoc
                        label = str(dir)

    return maxval, tHmax, tWmax, maxLocmax, label

def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);
    _, img = cv2.threshold(img, 127, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    img = cv2.erode(img, kernel, iterations=2)
    img = cv2.dilate(img, kernel, iterations=4)
    img = cv2.Canny(img, 180, 255)
    cv2.bitwise_not(img, img);
    img = cv2.erode(img, kernel, iterations=2)
    return img

def run(read_img):
    # open image to process
    #read_img = getCroppedSheet(read_img)
    img_main  = read_img
    img_color = read_img

    cv2.imshow('img', img_main)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    img_main = preprocess(img_main)



    cv2.imshow('img', img_main)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    '''  sliding window
    a = 120
    b = 120 + 122
    
    for i in range(0,16):
        #img = img_main[23:200, a:b]
        #imgcolor = img_color[23:200, a:b]
        img = img_main
        imgcolor = img_color
    
        #a = a + 122;
        #b = b + 122;
    
    
    '''

    #img_main = img_main[0:200, 140:2362]
    #img_color = img_color[0:200, 140:2362]
    offset = 10  # px offset to delete
    #cv2.imshow('img', img_main)
    #cv2.imshow('imgcolor', img_color)
    #cv2.waitKey(0)

    while(True):   # it will stop when there isnt a good match (all white)
        #print(best_match(img_main))
        try:
            maxval, tH, tW, maxLoc, label = best_match(img_main)
            if (maxLoc==0):
                print("There are no more samples to be recognized !")
                break;
        except Exception as e:
            print(e)
            break

        color = (255, 128, 64)
        cv2.rectangle(img_main, (maxLoc[0]-offset, maxLoc[1]-offset), (maxLoc[0] + tW+offset, maxLoc[1] + tH+offset), (255, 255, 255), cv2.FILLED) # paint white
        cv2.rectangle(img_color, (maxLoc[0], maxLoc[1]), (maxLoc[0] + tW, maxLoc[1] + tH), color, 2)
        cv2.putText(img_color, label, (maxLoc[0], maxLoc[1] + tW), 1, 1.1, (0, 128, 255), 2, cv2.LINE_AA)
        cv2.imshow('img', img_main)
        cv2.imshow('imgcolor', img_color)
        cv2.waitKey(1)


    #cv2.imshow('img', img_main)
    #cv2.imshow('imgcolor', img_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def getCroppedSheet(img):
    img_aux = img
    cropped = img
    # processment to detect sheet
    img_aux = cv2.cvtColor(img_aux, cv2.COLOR_BGR2GRAY);
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    img_aux = cv2.erode(img_aux, kernel, iterations=30)
    ret, img_aux = cv2.threshold(img_aux, 127, 255, 0)
    _, contours, _ = cv2.findContours(img_aux, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    ret, img_aux = cv2.threshold(img_aux, 127, 255, 0)
    _, contours, _ = cv2.findContours(img_aux, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(img, contours, 1, (0, 255, 0), 3)

    # draw rectangle over the sheet
    maxHeight, minHeight, maxWidth, minWidth = contourRectangleCoordinates(contours)
    offset = 10  # px offset
    #cv2.rectangle(img_color, (minHeight-offset, minWidth-offset), (maxHeight-offset, maxWidth-offset), (255, 0, 0), 2)

    # crop the rectangle
    cropped = cropped[ minWidth - offset : maxWidth - offset , minHeight - offset : maxHeight - offset]

    cv2.imshow('cropped', cropped)
    cv2.waitKey(0)
    return np.uint8(cropped)

def contourRectangleCoordinates(contours):
    # max values
    maxHeight = -1      # x
    maxWidth = -1       # y

    # min values
    minHeight = 10000000    # x
    minWidth = 10000000     # y

    first = True
    c=0
    for i in contours:
        if first:
            first = False
            continue;
        for j in i: # check contour coordinates
            for k in j:
                point = k
                x = point[0]
                y = point[1]
                if x > maxHeight:
                    maxHeight = x
                if x < minHeight:
                    minHeight = x
                if y > maxWidth:
                    maxWidth = y
                if y < minWidth:
                    minWidth = y

    return maxHeight, minHeight, maxWidth, minWidth


################################################################################################################

img = cv2.imread("sheets/sheet_easy.png")
run(img)

#TODO : problems when image is cropped, dunno why
#TODO : matching with scaling


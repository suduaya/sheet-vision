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

def discoverNote(img):
    print("Seeking for Notes")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    corners = cv2.goodFeaturesToTrack(gray, 5, 0.001, 10)  # Determines strong corners on an image
    corners = np.int0(corners)

    notes = []

    for corner in corners:
        x, y = corner.ravel()
        cv2.circle(img, (x, y), 3, 255, -1)
        notes.append(y)

    notes.sort() # ascending

    min = notes[0]
    max = notes[4]

    distances = []

    for i in range(0, 3):
        distance = notes[i+1] - notes[i]
        distances.append(distance)

    mean_distance = int(np.mean(distances))

    notes.append(min - mean_distance)
    notes.append(max + mean_distance)
    notes.sort()  # ascending

    #for corner in notes:
    #    y = corner
    #    cv2.circle(img, (6, y), 3, 255, -1)

    #print(notes)
    #cv2.imshow("note", img)
    #cv2.waitKey(0)

    return notes



def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);
    _, img = cv2.threshold(img, 127, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    img = cv2.dilate(img, kernel, iterations=4)
    img = cv2.erode(img, kernel, iterations=4)
    img = cv2.Canny(img, 180, 255)
    cv2.bitwise_not(img, img);
    img = cv2.erode(img, kernel, iterations=3)

    return img

def run(read_img):
    # open image to process
    read_img = getCroppedSheet(read_img)
    img_main  = read_img
    img_color = read_img
    _, templates = cv2.threshold(read_img, 127, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY)

    #cv2.imshow('img', img_main)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    img_main = preprocess(img_main)

    #cv2.imshow('img', img_main)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()


    offset = 10  # px offset to delete
    notes = False


    while(True):   # it will stop when there isnt a good match (all white)
        #print(best_match(img_main))
        try:
            maxval, tH, tW, maxLoc, label = best_match(img_main)
            # match
            (imgtH, imgtW) = templates.shape[:2]
            #new_match = templates[maxLoc[1]:(maxLoc[1] + tH), maxLoc[0]:(maxLoc[0] + tW)]  # exact match
            new_match = templates[0:(imgtH), maxLoc[0]:(maxLoc[0] + tW)]

            if (maxLoc==0):
                print("There are no more samples to be recognized !")
                break;
        except Exception as e:
            #print(e)
            print("Finished !")
            break

        color = (255, 128, 64)

        cv2.rectangle(img_main, (maxLoc[0]-offset, maxLoc[1]-offset), (maxLoc[0] + tW+offset, maxLoc[1] + tH+offset), (255, 255, 255), cv2.FILLED) # paint white
        cv2.rectangle(img_color, (maxLoc[0], maxLoc[1]), (maxLoc[0] + tW, maxLoc[1] + tH), color, 2)
        cv2.putText(img_color, label, (maxLoc[0], maxLoc[1] + tW), 1, 1.1, (0, 128, 255), 2, cv2.LINE_AA)
        ##
        #if label != "clave" and label != "end" and label != "split" and label != "pause":
        if label == "split" and notes == False :
            discoverNote(new_match)
            notes = True
        ##
        cv2.imshow('new_match', new_match)
        cv2.imshow('img', img_main)
        cv2.imshow('imgcolor', img_color)
        cv2.waitKey(0)

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
    offset = 0  # px offset
    #cv2.rectangle(cropped, (minHeight-offset, minWidth-offset), (maxHeight-offset, maxWidth-offset), (255, 0, 0), 2)
    # crop the rectangle
    cropped = cropped[ minWidth - offset : maxWidth - offset , minHeight - offset : maxHeight - offset]

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

def getChessRectangle(corners):
    # max values
    maxHeight = -1  # x
    maxWidth = -1  # y

    # min values
    minHeight = 10000000  # x
    minWidth = 10000000  # y

    first = True
    c = 0
    for i in corners:
        for j in i:  # check contour coordinates
            #print(j)
            point = j
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

def findChessboards(img):
    pattern_size = (3,3)
    centers = []
    while(True):
        ret, corners = cv2.findChessboardCorners(img, pattern_size, None)
        #print(corners)
        # If found, add object points, image points
        if ret == True:
            # Draw and display the corners
            cv2.drawChessboardCorners(img, pattern_size, corners, ret)
            #write_name = 'corners_found'+str(idx)+'.jpg'
            #cv2.imwrite(write_name, img)
            #cv2.imshow('img', img)
            #cv2.waitKey(0)
            center = corners[4]
            centers.append(center)
            maxHeight, minHeight, maxWidth, minWidth = getChessRectangle(corners)

            offset = np.float32((corners[1][0][0] - corners[0][0][0]))
            offset = np.float32(50)

            cv2.rectangle(img, (minHeight-offset, minWidth-offset), (maxHeight+offset, maxWidth+offset), (255, 255, 255), cv2.FILLED)
            #cv2.imshow('img', img)
            #cv2.waitKey(0)

        if ret == False:
            break

    sorted_ctrs = sorted(centers, key=lambda centers: cv2.boundingRect(centers)[0]* img.shape[0] - cv2.boundingRect(centers)[1] * img.shape[1] )
    lb= sorted_ctrs[0]
    lt= sorted_ctrs[1]
    rb= sorted_ctrs[2]
    rt= sorted_ctrs[3]

    src_pts = np.array([lt[0], rt[0], rb[0], lb[0]], dtype=np.float32)

    warp = perspective_transform(img, src_pts)
    return warp

def perspective_transform(image, rect):
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # apply perspectives
    M = cv2.getPerspectiveTransform(rect, dst)
    toRet = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return toRet

################################################################################################################

img = cv2.imread("sheets/warp8.png")

run(img)

#
#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);
#img = findChessboards(img)
#_, img = cv2.threshold(img, 127, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY)
#cv2.imshow("warp", img)
#cv2.imwrite("sheets/warp8.png", img)
#cv2.waitKey(0)
#
#TODO : matching with scaling
#TODO : camera calibration
#TODO : detect notes


import numpy as np
import cv2
import os
import glob
import sys


'''
    Calibration and Undistortion procedure
'''
def calibrateAndUndistort(new_capture):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((9 * 6, 3), np.float32)
    objp[:, :2] = np.mgrid[0:6, 0:9].T.reshape(-1, 2)

    objpoints = []
    imgpoints = []

    images = glob.glob('calibration/*_calib.jpg')
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (6, 9), None)

        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            img = cv2.drawChessboardCorners(img, (6, 9), corners2, ret)

    cv2.destroyAllWindows()
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    img = new_capture
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    img = cv2.undistort(img, mtx, dist, None, newcameramtx)

    return np.uint8(img)


'''
    Matching round algorithm, returning the best candidate at that iteration
'''
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
                    if file == ".DS_Store":
                        continue
                    template = cv2.imread("templates_examples/"+str(dir) + "/"+ str(file))
                    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY);
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


'''
    Min absolute distance calc
'''
def minDistanceLine(points, lines):
    min_indexes = []

    for point in points:
        distances = []
        for line in lines:
            distances.append(abs(point[1] - line))
        min_indexes.append(np.argmin(distances))

    return min_indexes


'''
    Give an template (img), the line coordinates and the label of the given template,
    finds the nearest line, predicting the note
'''
def best_match_note(img, lines, label):
    cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    kernel = np.ones((3, 3), np.uint8)

    if label == 'half':
        img = cv2.erode(img, kernel, iterations=2)
        img = cv2.dilate(img, kernel, iterations=6)
        img = cv2.erode(img, kernel, iterations=2)
    else:
        img = cv2.dilate(img, kernel, iterations=4)
        img = cv2.erode(img, kernel, iterations=2)

    ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    crop, contours, hierarchy = cv2.findContours(thresh, 1, 2)

    centers_list = []
    for cnt in contours:
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        radius = int(radius)

        if radius < 30:
            cv2.circle(cimg, center, radius, (0, 255, 0), 2)
            cv2.circle(cimg, center, 2, (0, 0, 255), 3)
            centers_list.append(center)

    centers_list.sort(key=lambda centers: centers[0])
    min_indexes = minDistanceLine(centers_list, lines)
    points_and_indexes = list(zip(centers_list, min_indexes))

    return points_and_indexes


'''
    Custom image processing sequence applied on the input image and on the templates
'''
def preprocess(img):
    _, img = cv2.threshold(img, 127, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    img = cv2.dilate(img, kernel, iterations=4)
    img = cv2.erode(img, kernel, iterations=4)
    img = cv2.Canny(img, 180, 255)
    cv2.bitwise_not(img, img)
    img = cv2.erode(img, kernel, iterations=3)

    return img


'''
    Main
'''
def run(read_img):
    # open image to process

    img_main = read_img
    img_color = read_img
    _, templates = cv2.threshold(read_img, 127, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY)
    img_main = preprocess(img_main)

    check_notes = ["a", "f", "d", "b", "g", "e", "c", "g", "e", "c", "a", "f", "d"]

    # static sheet coordinates
    linearOffset = 1
    (y, x) = img_color.shape[:2]
    lines = []
    line1 = int(y * 0.4) + linearOffset
    line2 = int(y * 0.43) + linearOffset
    line3 = int(y * 0.47) + linearOffset
    line4 = int(y * 0.50) + linearOffset
    line5 = int(y * 0.54) + linearOffset
    line6 = int(y * 0.5706) + linearOffset
    line7 = int(y * 0.606) + linearOffset
    line1_2 = int(y * 0.42)
    line2_3 = int(y * 0.45) + linearOffset
    line3_4 = int(y * 0.485) + linearOffset
    line4_5 = int(y * 0.52125)
    line5_6 = int(y * 0.55125) + linearOffset
    line6_7 = int(y * 0.585) + linearOffset
    lines.append(line1)
    lines.append(line2)
    lines.append(line3)
    lines.append(line4)
    lines.append(line5)
    lines.append(line6)
    lines.append(line7)
    lines.append(line1_2)
    lines.append(line2_3)
    lines.append(line3_4)
    lines.append(line4_5)
    lines.append(line5_6)
    lines.append(line6_7)


    img_color = cv2.cvtColor(img_color, cv2.COLOR_GRAY2BGR)
    cv2.imshow("imgcolor", img_color)
    cv2.imshow('img', img_main)
    cv2.waitKey(0)

    while(True):   # it will stop when there isnt a good match (all white)
        try:       # try a match
            maxval, tH, tW, maxLoc, label = best_match(img_main)
            # match
            (imgtH, imgtW) = templates.shape[:2]

            new_match = templates[0:(imgtH), maxLoc[0]:(maxLoc[0] + tW)]

            if (maxLoc==0):
                print("There are no more samples to be recognized !")
                break
        except Exception as e:
            #print(e)
            print("Finished !")
            break

        # Draw and paint

        color = (255, 128, 64)

        if label == "pause" or label == "split":
            offset = 0
            color = (255, 128, 64)
        else:
            offset = 5
        cv2.rectangle(img_main, (maxLoc[0]-offset, maxLoc[1]-offset), (maxLoc[0] + tW+offset, maxLoc[1] + tH+2*offset), (255, 255, 255), cv2.FILLED) # paint white
        if label == "pause":
            cv2.rectangle(img_color, (maxLoc[0], maxLoc[1]), (maxLoc[0] + tW, maxLoc[1] + tH), color, 1)
            cv2.putText(img_color, label, (maxLoc[0], maxLoc[1] + tW), 1, 1.1, (0, 128, 255), 2, cv2.LINE_AA)

        if label != "split" and label != "end" and label != "pause" and label != "clave":
            #cv2.rectangle(img_main, (maxLoc[0]-offset, maxLoc[1]-offset), (maxLoc[0] + tW+offset, maxLoc[1] + tH+offset), (255, 255, 255), cv2.FILLED) # paint white
            cv2.rectangle(img_color, (maxLoc[0], maxLoc[1]), (maxLoc[0] + tW, maxLoc[1] + tH), color, 1)
            cv2.putText(img_color, label, (maxLoc[0], maxLoc[1] + tW), 1, 1.1, (0, 128, 255), 2, cv2.LINE_AA)
            try:
                points_and_indexes = best_match_note(new_match, lines, label)
                for toLabel in points_and_indexes:
                    note_x = int(toLabel[0][0])+ int(maxLoc[0])
                    note_y = int(toLabel[0][1])
                    note_label = check_notes[toLabel[1]]
                    cv2.putText(img_color, note_label, (note_x, note_y + 15), 2, 1.1, (0,0,255), 2, cv2.LINE_AA)

            except Exception as e:
                print("Error while detecting note tones. (best_match_note)")
                print(e)

        # Refresh
        cv2.imshow('img', img_main)
        cv2.imshow('imgcolor', img_color)
        cv2.waitKey(1)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return


'''
    Auxiliary function to get the chess rectangles and afterwards delete it (paint as white)
'''
def getChessRectangle(corners):
    # max values
    maxHeight = -1  # x
    maxWidth = -1  # y

    # min values
    minHeight = 10000000  # x
    minWidth = 10000000  # y

    for i in corners:
        for j in i:  # check contour coordinates
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


'''
    Finds chessboards centers on the input img and applies perspective transformation
'''
def findChessboards(img):#
    pattern_size = (3,3)
    centers = []
    while(True):
        ret, corners = cv2.findChessboardCorners(img, pattern_size, None)
        # If found, add object points, image points
        if ret == True:
            # Draw and display the corners
            cv2.drawChessboardCorners(img, pattern_size, corners, ret)
            center = corners[4]
            centers.append(center)
            maxHeight, minHeight, maxWidth, minWidth = getChessRectangle(corners)

            offset = np.float32(50)
            cv2.rectangle(img, (minHeight-offset, minWidth-offset), (maxHeight+offset, maxWidth+offset), (255, 255, 255), cv2.FILLED)

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


'''
    Applies perspective transformation based on chess centers
'''
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


'''
    Reads input image, calibrates and rectifies lens distortion, returning a clean image from perspective transformation
'''
def readCalibUndist(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = calibrateAndUndistort(img)
    _, img = cv2.threshold(img, 127, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY)
    img = findChessboards(img)

    return img


################################################################################################################

img = cv2.imread("sheets/"+sys.argv[1])
img = readCalibUndist(img)
cv2.imwrite("sheets/processed.png", img)

run(img)




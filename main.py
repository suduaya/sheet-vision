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
            #print(dir)
            for _, _, files in os.walk("templates_examples/"+str(dir)):
                for file in files:
                    #print("   " + file)
                    template = cv2.imread("templates_examples/"+str(dir) + "/"+ str(file))
                    #cv2.imshow("template", template)
                    #cv2.waitKey(0)
                    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY);
                    ret, template = cv2.threshold(template, 127, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY)
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                    template = cv2.erode(template, kernel, iterations=2)
                    template = cv2.dilate(template, kernel, iterations=4)
                    template = cv2.Canny(template, 180,255)
                    cv2.bitwise_not(template, template);
                    template = cv2.erode(template, kernel, iterations=2)


                    result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
                    (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
                    (tH, tW) = template.shape[:2]

                    if maxVal > maxval:
                        maxval = maxVal
                        tHmax = tH
                        tWmax = tW
                        maxLocmax = maxLoc
                        label = str(dir)

                    #print(maxVal)
                    #print("\n")
                    #cv2.waitKey(0)

    return maxval, tHmax, tWmax, maxLocmax, label


def main():
    # open image to process
    img_main  = cv2.imread('sheets/sheet_easy.png')
    img_color = cv2.imread('sheets/sheet_easy.png')

    cv2.imshow('img', img_main)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # pre processment
    img_main = cv2.cvtColor(img_main, cv2.COLOR_BGR2GRAY);
    ret,img_main = cv2.threshold(img_main,127, 255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    img_main = cv2.erode(img_main,kernel,iterations = 2)
    img_main = cv2.dilate(img_main,kernel,iterations = 4)
    img_main = cv2.Canny(img_main,180,255)
    cv2.bitwise_not ( img_main, img_main  );
    img_main = cv2.erode(img_main, kernel, iterations=2)



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
    for i in range(0,100000):   # it will stop when there isnt a good match (all white)
        print(best_match(img_main))
        try:
            maxval, tH, tW, maxLoc, label = best_match(img_main)
            if (maxLoc==0):
                break;
        except:
            break

        color = (255, 128, 64)
        cv2.rectangle(img_main, (maxLoc[0], maxLoc[1]), (maxLoc[0] + tW, maxLoc[1] + tH), (255, 255, 255), cv2.FILLED) # paint white
        cv2.rectangle(img_color, (maxLoc[0], maxLoc[1]), (maxLoc[0] + tW, maxLoc[1] + tH), color, 2)
        cv2.putText(img_color, label, (maxLoc[0], maxLoc[1] + tW), 1, 1.1, (0, 128, 255), 2, cv2.LINE_AA)
        #cv2.imshow('img', img_main)
        cv2.imshow('imgcolor', img_color)
        cv2.waitKey(1)


    #cv2.imshow('img', img_main)
    #cv2.imshow('imgcolor', img_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



main()
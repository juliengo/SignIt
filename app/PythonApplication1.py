
import cv2
import sys
import os
import matplotlib
import numpy as np
import imutils 
import time

import tensorflow as tf

from run import * 
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

fileImage = dir_path+"/test.png"

import threading

loadingBuffer = 0

bg = None
sentence = ''
result = ''
characterHeld = False

num_frames = 0
prev_word_count = 0
prev_word = ''

busy = False
bgModel = False

def run_avg(image, aWeight):
    global bg
    # initialize the background
    if bg is None:
        bg = image.copy().astype("float")
        return

    # compute weighted average, accumulate it and update the background
    cv2.accumulateWeighted(image, bg, aWeight)


def segment(image, threshold=25):
    global bg
    # find the absolute difference between background and current frame
    diff = cv2.absdiff(bg.astype("uint8"), image)

    # threshold the diff image so that we get the foreground
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]

    # get the contours in the thresholded image
    cnts, ha= cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # return None, if no contours detected
    if len(cnts) == 0:
        return
    else:
        # based on contour area, get the maximum contour which is the hand
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)

def predictImage(roi):
    global busy
    cv2.imwrite(fileImage,roi)
    results = predict_local_image(fileImage)
    if(results):
        letter = results['predictedTagName']
        appendToSentence(letter)
        print(letter)
        
        busy = False


def createThread(roi):
    thr = threading.Thread(target=predictImage,args=(roi,),kwargs={})
    thr.start()

def appendToSentence(result):
    global sentence
    if(len(sentence) > 15):
        sentence = ''
    if result=="nothing":
        pass
    else:
        sentence += result

def removeBG(frame):
    fgmask = bgModel.apply(frame,learningRate=0)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # res = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    kernel = np.ones((0, 0), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res
    

if __name__ == "__main__":

    # initialize weight for running average
    aWeight = 0.5

    # get the reference to the webcam
    camera = cv2.VideoCapture(0)
   

    while(True): 
        rect, image = camera.read()
        #flip the image so that we can compare picture to picture
        image = cv2.flip(image, 1)

        #set the rectangle coordinates 
        if rect:
            top, right, bottom, left = 100,0,700,600 #100, 600, 700, 1200

        (height, width) = image.shape[:2]

        # get the ROI
        roi = image[top:bottom, right:left]

        '''if(bgModel):
            roi = removeBG(roi)
            cv2.imshow("mask",roi)'''

        # convert the roi to grayscale and blur it
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)


        if num_frames < 10:
            run_avg(gray, aWeight)
        else:
            # segment the hand region
            hand = segment(gray)

            # check whether hand region is segmented
            if hand is not None:
                # if yes, unpack the thresholded image and
                # segmented region
                (thresholded, segmented) = hand

                # draw the segmented region and display the frame
                cv2.drawContours(image, [segmented + (right, top)], -1, (0, 0, 255))
                #cv2.imshow("Thesholded", thresholded)

                

        # draw the segmented hand
        cv2.rectangle(image, (left, top), (right, bottom), (0,255,0), 2)


        keypress = cv2.waitKey(1)

        # if the user pressed "q", then stop looping
        if keypress == ord("q"):
            break
        elif keypress == ord('b'):  # press 'b' to capture the background
            bgModel = cv2.createBackgroundSubtractorMOG2(0, 50)
            isBgCaptured = 1
            print( '!!!Background Captured!!!')

        image_data = cv2.imencode('.jpg', image)[1].tostring()
        

        if busy==False:#num_frames >= 20:
            loadingBuffer += 1
            cv2.putText(image, "Set hand", (50,50), cv2.FONT_HERSHEY_TRIPLEX, 1, (255,255,255))
            if(loadingBuffer >= 20):
                loadingBuffer = 0
                busy=True
                createThread(roi)
                


            '''#buffer
            if prev_word == result:
                prev_word_count += 1
            else:
                prev_word_count = 0
            
            if prev_word_count == 2:
                #sentence += result
                characterHeld = True
                prev_word_count = 0
            prev_word = result'''
        else:
            cv2.putText(image, "analyzing", (50,50), cv2.FONT_HERSHEY_TRIPLEX, 1, (255,255,255))

       
        num_frames += 1
        score = 2
        #prev_word = result
        cv2.putText(image, "%s" % (result.upper()), (100,400), cv2.FONT_HERSHEY_TRIPLEX, 1, (255,255,255))
        #cv2.putText(image, '(score = %.5f)' % (float(score)), (100,450), cv2.FONT_HERSHEY_TRIPLEX, 1, (255,255,255))

        cv2.imshow("Video Feed", image)

        img_sequence = np.zeros((200,1200,3), np.uint8)
        cv2.putText(img_sequence, '%s' % (sentence.upper()), (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.imshow('sentence', img_sequence)
        

cv2.destroyAllWindows() 
cv2.VideoCapture(0).release()


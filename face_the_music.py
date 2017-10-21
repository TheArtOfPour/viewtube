import cv2
import sys
import os
import time
import signal
import subprocess
import numpy as np
from random import randint

scale = 1.1
imFlag = cv2.CASCADE_SCALE_IMAGE #cv2.cv.CV_HAAR_SCALE_IMAGE
minSize = 40
minNeighbors = 10

sentimental = True
emotions = ["neutral", "anger", "happy"]
loopsBeforeChange = 20
font = cv2.FONT_HERSHEY_SIMPLEX
videos = ["lX6JcybgDFo", "eBvm4FZF8L4", "7ZQLX4F_0Eg", "sUtS52lqL5w",
          "ZwL0t5kPf6E", "mkQ2pXkYjRM", "7A_jPky3jRY", "D-UmfqFjpl0"]

def openVideo(index):
    global videos
    video = videos[index]
    subprocess.Popen(["youtube-viewer https://www.youtube.com/watch?v="+video], shell=True)

def closeVideo():
    os.system("killall youtube-viewer")
    os.system("killall mpv")

cascDir = "/home/nvidia/opencv-2.4.9/data/haarcascades/"
cascPath = cascDir + "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

cascPath = cascDir + "haarcascade_frontalface_alt.xml"
faceCascade2 = cv2.CascadeClassifier(cascPath)

cascPath = cascDir + "haarcascade_frontalface_alt2.xml"
faceCascade3 = cv2.CascadeClassifier(cascPath)

cascPath = cascDir + "haarcascade_frontalface_alt_tree.xml"
faceCascade4 = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)
loopsWithoutFace = 0
videoIndex = randint(0, len(videos)-1)
score = 0
highScore = 1 
totalScore = 0
highScoreFace = []
highScoreVideo = ""
xFace = 5
yFace = 80
openVideo(videoIndex)
#time.sleep(5)
if sentimental:
    fishface = cv2.createFisherFaceRecognizer()
    fishface.load("fishface.yml")

while True:
    newHighScore = False
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
       
    faces = faceCascade.detectMultiScale(gray, scaleFactor=scale, minNeighbors=minNeighbors, minSize=(minSize, minSize), flags=imFlag)
    if len(faces) == 0:
        faces = faceCascade2.detectMultiScale(gray, scaleFactor=scale, minNeighbors=minNeighbors, minSize=(minSize, minSize), flags=imFlag)
        if len(faces) == 0:
            faces = faceCascade3.detectMultiScale(gray, scaleFactor=scale, minNeighbors=minNeighbors, minSize=(minSize, minSize), flags=imFlag)
            if len(faces) == 0:
               faces = faceCascade4.detectMultiScale(gray, scaleFactor=scale, minNeighbors=minNeighbors, minSize=(minSize, minSize), flags=imFlag)

    if len(faces) == 0:
        score = 0
        if loopsWithoutFace < loopsBeforeChange:
            loopsWithoutFace = loopsWithoutFace + 1
        else:
            #buffer to allow video to transition
            loopsWithoutFace = -30
            closeVideo()
            if videoIndex == len(videos)-1:
                videoIndex = 0
            else:
                videoIndex = videoIndex + 1
            openVideo(videoIndex)
    else:
        score = score + 1
        if score == highScore:
            newHighScore = True
            highScoreVideo = videos[videoIndex]
        if score > highScore:
            highScore = score
        totalScore = totalScore + 1
        loopsWithoutFace = 0

    # Draw a rectangle around the faces
    color = (0, 255, 0)
    if score*3 < 255:
        color = (0, score*3, 255-score*3)

    for (x, y, w, h) in faces:
        if newHighScore:
            highScoreFace = frame[y:y+h, x:x+w]
        if sentimental:
            img = frame[y:y+h, x:x+w]
            #image = cv2.imread(img)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            res = cv2.resize(gray,(350, 350))
            pred, conf = fishface.predict(res)
            cv2.putText(frame, emotions[pred] + " " + str(conf), (x+w+5, y-25), font, 0.8, color, 2, cv2.CV_AA)
            #print(emotions[pred])
            #print(conf)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, str(score), (x+w+5, y-5), font, 0.8, color, 2, cv2.CV_AA)    
    # Text
    totalColor = (0, 255, 0)
    if totalScore/2 < 255:
        totalColor = (0, int(totalScore/2), 255-int(totalScore/2))
    cv2.putText(frame, "Total Score: "+str(totalScore), (5, 25), font, 0.8, totalColor, 2, cv2.CV_AA)
    highColor = (0, 255, 0)
    if highScore < 255:
        highColor = (0, highScore, 255 - highScore)
    cv2.putText(frame, "High Score: "+str(highScore)+" "+highScoreVideo, (5, 60), font, 0.8, highColor, 2, cv2.CV_AA)    
    # High Score Face
    if len(highScoreFace) > 0:
        frame[yFace:yFace+highScoreFace.shape[0], xFace:xFace+highScoreFace.shape[1]] = highScoreFace
    # Display the resulting frame
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
closeVideo()

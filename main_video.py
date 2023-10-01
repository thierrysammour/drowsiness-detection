# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
from time import sleep
import platform
import os


if platform.system() == 'Windows':
    import pyaudio
    def play_sound():
        # Initialise the only _variable_ in use...
        n=0
        # Open the stream required, mono mode only...
        stream=pyaudio.PyAudio().open(format=pyaudio.paInt8,channels=1,rate=16000,output=True)
        for beep_num in range(0,3):
            # Now generate the 1KHz signal at the speakers/headphone output for about 0.1 seconds...
            # Sine wave, to 8 bit depth only...
            for n in range(0,100,1): stream.write("\x00\x30\x5a\x76\x7f\x76\x5a\x30\x00\xd0\xa6\x8a\x80\x8a\xa6\xd0")
            # Close the open _channel(s)_...
            sleep(0.1)
        stream.close()
        pyaudio.PyAudio().terminate()


def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])
	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)
	# return the eye aspect ratio
	return ear


# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold for to set off the
# alarm
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 25
# initialize the frame counter as well as a boolean used to
# indicate if the alarm is going off
COUNTER = 0
ALARM_ON = False

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

(jStart, jEnd) = face_utils.FACIAL_LANDMARKS_IDXS["jaw"]

cv2.namedWindow("Drowsiness Detection", 0)

# start the video stream thread
print("[INFO] starting video stream thread...")

# vs = VideoStream(src=0).start()
cap = cv2.VideoCapture(0)
# Define the codec and create VideoWriter object
# fourcc = cv2.VideoWriter_fourcc(*'MJPG')
# out = cv2.VideoWriter('result.avi', fourcc, 25.0, (640,  360))



while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break


    frame = imutils.resize(frame, width=640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# detect faces in the grayscale frame
    rects = detector(gray, 0)

    # loop over the face detections
    for rect in rects:
    # determine the facial landmarks for the face region, then
    # convert the facial landmark (x, y)-coordinates to a NumPy
    # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        # extract the left and right eye coordinates, then use the
        # coordinates to compute the eye aspect ratio for both eyes
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        # average the eye aspect ratio together for both eyes
        ear = (leftEAR + rightEAR) / 2.0

        # compute the convex hull for the left and right eye, then
        # visualize each of the eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)


        # cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        # cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
       

# check to see if the eye aspect ratio is below the blink
# threshold, and if so, increment the blink frame counter

        color = (0, 255, 0)

        if ear < EYE_AR_THRESH:

            color = (0, 255, 255)

            COUNTER += 1
            # if the eyes were closed for a sufficient number of
            # then sound the alarm
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                # if the alarm is not on, turn it on

                color = (0, 0, 255)

                if not ALARM_ON:
                    ALARM_ON = True

                    if platform.system() == 'Windows':
                        t = Thread(target=play_sound())
                        t.deamon = True
                        t.start()

          
                    # check to see if an alarm file was supplied,
                    # and if so, start a thread to have the alarm
                    # sound played in the background

                # draw an alarm on the frame
                cv2.putText(frame, "WAKE UP!", (int(frame.shape[1]/2) - int(frame.shape[1]/10), 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        # otherwise, the eye aspect ratio is not below the blink
        # threshold, so reset the counter and alarm
        else:
            COUNTER = 0
            ALARM_ON = False
            

        # delta = 7

        # rect = cv2.boundingRect(leftEyeHull)
        # cv2.rectangle(frame, (rect[0] - delta, rect[1] - delta), (rect[0] + rect[2] + delta, rect[1] + rect[3] + delta), color, 2)

        # rect = cv2.boundingRect(rightEyeHull)
        # cv2.rectangle(frame, (rect[0] - delta, rect[1] - delta), (rect[0] + rect[2] + delta, rect[1] + rect[3] + delta), color, 2)


        for landmark in shape[jStart:jEnd]:
            cv2.circle(frame, landmark, 1, color, -1)

        for landmark in shape[lStart:lEnd]:
            cv2.circle(frame, landmark, 1, color, -1)

        for landmark in shape[rStart:rEnd]:
            cv2.circle(frame, landmark, 1, color, -1)

		# draw the computed eye aspect ratio on the frame to help
		# with debugging and setting the correct eye aspect ratio
		# thresholds and frame counters
        # cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
        # cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
 

    # write the flipped frame
    # out.write(frame)

    # show the frame
    cv2.imshow("Drowsiness Detection", frame)
    key = cv2.waitKey(1) & 0xFF





    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
# do a bit of cleanup
cv2.destroyAllWindows()

# Release everything if job is finished
# cap.release()
# out.release()
# cv2.destroyAllWindows()
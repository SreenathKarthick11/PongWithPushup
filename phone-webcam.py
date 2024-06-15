import cv2
import numpy as np
# camera and videcapture
address1="http://100.75.***.***:8080/video"  # enter your ip address
video = cv2.VideoCapture(0) # number stands for which web cam u want to use,if file name we can see the video
video.open(address1)
while True:
    ret,frame = video.read() # ret returns false when camera is being used
                           # frames gives the current frame

    cv2.imshow("Frame",frame)

    if cv2.waitKey(1) == ord("q"):# when u press key q for 1 milli sec the loop will break
        break

video.release() # release the camera from use
cv2.destroyAllWindows()

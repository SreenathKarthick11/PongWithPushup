import cv2
import mediapipe as md
# mediapipe has different models for different parts like pos
# the classes we need
md_drawing=md.solutions.drawing_utils
md_drawing_styles=md.solutions.drawing_styles
md_pose=md.solutions.pose
# we count a push up when a shoulder is below the elbow

count=0
position1=None
position2=None
cap=cv2.VideoCapture(0)

# calling the pose model to recognise with an accuracy
with md_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success,image=cap.read()
        if not success:
            print("empty camera")
            break
        # media works on rgb and we flip the image so mirroring
        image=cv2.cvtColor(cv2.flip(image,1),cv2.COLOR_BGR2RGB)
        result=pose.process(image) # using model to identify in video

        lmList=[] # it will contain the position of all 32 points in the pos model

        if result.pose_landmarks: # if models is present
            md_drawing.draw_landmarks(image,result.pose_landmarks,md_pose.POSE_CONNECTIONS)
             # image=image on which we want to draw the landmark
             # result.pose_landmarks = these are the points by recoginzing with the model
             # md_pose.POSE_CONNECTION = these are the lines which connect the points

             # now we loop through the postion of each point on the body
            for id,lm in enumerate(result.pose_landmarks.landmark):
                # id=no id of the point and im (landmark) is the x and y ratio value of point
                h,w,_=image.shape # height and width
                x,y=int(lm.x*w),int(lm.y*h)
                lmList.append([id,x,y])
        if len(lmList) != 0: # if no model is detected
            
            if ((lmList[12][2] - lmList[14][2])>=5 and (lmList[11][2] - lmList[13][2])>=5):
                position = "down" # postion of both sholder below elbow
            if ((lmList[12][2] - lmList[14][2])<=5 and (lmList[11][2] - lmList[13][2])<=5) and position == "down":
                position = "up"  # postion of both sholder abow elbow
                count +=1 
                print(count)
            
                
        image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        cv2.imshow("Push up counter",image)
        
        if cv2.waitKey(1)==ord("q"):
            break

cap.release()

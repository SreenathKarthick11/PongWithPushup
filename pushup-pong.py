import cv2
import numpy as np
import cv2
import mediapipe as md
import pygame, sys, random

# mediapipe has different models for different parts like pos
# the classes we need
md_drawing=md.solutions.drawing_utils
md_drawing_styles=md.solutions.drawing_styles
md_pose=md.solutions.pose
# we count a push up when a shoulder is below the elbow

count1=0
count2=0
position1=None
position2=None
# camera and videcapture
address1="http://100.75.***.***:8080/video"  # every ip address in different so enter yours
# address2="http://100.91.139.67:8080/cap"
video1 = cv2.VideoCapture(0) # number stands for which web cam u want to use,if file name we can see the video
video2=cv2.VideoCapture(0) 
video1.open(address1)
# cap.open(address2)

#########################################
# General Setup
pygame.init()
clock=pygame.time.Clock()

# setting up with main window
screen_width = 800
screen_height = 600
screen = pygame.display.set_mode((screen_width,screen_height))
pygame.display.set_caption('PingPong')

# Game rectangles
ball = pygame.Rect(screen_width/2 - 15,screen_height/2 -15,30,30)
player = pygame.Rect(screen_width-20,screen_height/2-40,10,80)
opponent = pygame.Rect(10,screen_height/2-40,10,80)

bg_color = pygame.Color('grey12')
light_grey = (200,200,200)
light_blue  = (173,216,250)
light_green = (152,251,152)
ball_speed_x = 7 * random.choice((1,-1))
ball_speed_y = 7 * random.choice((1,-1))
player_speed = 0
opponent_speed = 0
left_score = 0
right_score = 0
Score_font = pygame.font.SysFont('arial', 50)
left_score_text = Score_font.render(f"{left_score}",True,light_grey)
right_score_text = Score_font.render(f"{right_score}",True,light_grey)
################################

##################
#ball animation function
def ball_animation():
    global ball_speed_x,ball_speed_y,left_score,right_score
    ball.x += ball_speed_x
    ball.y += ball_speed_y

    if ball.top <= 0 or ball.bottom >= screen_height:
        ball_speed_y *= -1
    if ball.left <= 0 :
        right_score += 1
    elif ball.right >= screen_width :
        left_score += 1
    if ball.left <= 0 or ball.right >= screen_width:
        ball_restart()
    
    if ball.colliderect(player) or ball.colliderect(opponent):
         ball_speed_x *= -1 
    
# player animation function
def player_animation():
    player.y += player_speed
    if player.top <= 0:
        player.top = 0
    if player.bottom >= screen_height:
        player.bottom = screen_height

# opponent animation function
def opponent_animation():
    opponent.y += opponent_speed
    if opponent.top <= 0:
        opponent.top = 0
    if opponent.bottom >= screen_height:
        opponent.bottom = screen_height

def ball_restart():
    global ball_speed_y,ball_speed_x
    ball.center = (screen_width/2,screen_height/2)
    ball_speed_y *= random.choice((1,-1))
    ball_speed_x *= random.choice((1,-1))
#####################



with md_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5) as pose:
    while video1.isOpened():
        ret1,frame1 = video1.read() # ret returns false when camera is being used, frames gives the current frame
        ret2,frame2=video2.read()
        frame1=cv2.resize(frame1,(300,200))
        frame2=cv2.resize(frame2,(300,200))
    
        if not ret1:
            print("empty camera")
            break
        # media works on rgb and we flip the image so mirroring
        frame1=cv2.cvtColor(cv2.flip(frame1,1),cv2.COLOR_BGR2RGB)
        frame2=cv2.cvtColor(cv2.flip(frame2,1),cv2.COLOR_BGR2RGB)
        result1=pose.process(frame1) # using model to identify in video
        result2=pose.process(frame2) # using model to identify in video
        

        lmList1=[] # it will contain the position of all 32 points in the pos model
        lmList2=[]
        #########
        if result1.pose_landmarks: # if models is present
            md_drawing.draw_landmarks(frame1,result1.pose_landmarks,md_pose.POSE_CONNECTIONS)

             # now we loop through the postion of each point on the body
            for id1,lm1 in enumerate(result1.pose_landmarks.landmark):
                # id=no id of the point and im (landmark) is the x and y ratio value of point
                h1,w1,_=frame1.shape # height and width
                x1,y1=int(lm1.x*w1),int(lm1.y*h1)
                lmList1.append([id1,x1,y1])
        
        #########
        if result2.pose_landmarks: # if models is present
            md_drawing.draw_landmarks(frame2,result2.pose_landmarks,md_pose.POSE_CONNECTIONS)

             # now we loop through the postion of each point on the body
            for id,lm in enumerate(result2.pose_landmarks.landmark):
                # id=no id of the point and im (landmark) is the x and y ratio value of point
                h2,w2,_=frame2.shape # height and width
                x2,y2=int(lm.x*w2),int(lm.y*h2)
                
                lmList2.append([id,x2,y2])
        
        ##########        
        frame1=cv2.cvtColor(frame1,cv2.COLOR_RGB2BGR)
        frame2=cv2.cvtColor(frame2,cv2.COLOR_RGB2BGR)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            
        # Update player1 speed based on template matching location
        if len(lmList1) != 0: # if no model is detected
            
            if ((lmList1[12][2] - lmList1[14][2])>=5 and (lmList1[11][2] - lmList1[13][2])>=5):
                player_speed = 7 # postion of both sholder below elbow
            
            if ((lmList1[12][2] - lmList1[14][2])<=5 and (lmList1[11][2] - lmList1[13][2])<=5) :
                 # postion of both sholder abow elbow
                player_speed = -7
                
        # Update player2 speed based on template matching location
        if len(lmList2) != 0: # if no model is detected
            
            if ((lmList2[12][2] - lmList2[14][2])>=5 and (lmList2[11][2] - lmList2[13][2])>=5):
                 # postion of both sholder below elbow
                opponent_speed = 7
            if ((lmList2[12][2] - lmList2[14][2])<=5 and (lmList2[11][2] - lmList2[13][2])<=5) :
             # postion of both sholder abow elbow
                opponent_speed = -7


        ball_animation()
        player_animation()
        opponent_animation()

        # Visuals
        screen.fill(bg_color)
        pygame.draw.rect(screen, light_blue, player)
        left_score_text = Score_font.render(f"Player 1: {left_score}", True, light_grey)
        right_score_text = Score_font.render(f"Player 2: {right_score}", True, light_grey)
        screen.blit(left_score_text, (screen_width / 4 - left_score_text.get_width() / 2, 20))
        screen.blit(right_score_text, (screen_width * 3 / 4 - right_score_text.get_width() / 2, 20))
        pygame.draw.rect(screen, light_green, opponent)
        pygame.draw.ellipse(screen, light_grey, ball)
        pygame.draw.aaline(screen, light_grey, (screen_width / 2, 0), (screen_width / 2, screen_height))

        # Updating the window
        pygame.display.flip()
        clock.tick(30)

        # Display the OpenCV windows for debugging
        cv2.imshow('Match1_BLUEFRAME', frame1)
        cv2.imshow('Match2_GREENFRAME', frame2)

    
        if cv2.waitKey(1) == ord("q"):# when u press key q for 1 milli sec the loop will break
            break

video1.release() # release the camera from use
video2.release()
cv2.destroyAllWindows()

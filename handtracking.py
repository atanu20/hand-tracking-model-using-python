import cv2
import time
import mediapipe as mp
from mediapipe.python.solutions import hands



cap = cv2.VideoCapture(0)

mphand=mp.solutions.hands
hands=mphand.Hands()
mpDraw = mp.solutions.drawing_utils
# static_image_mode=False,
#     max_num_hands=2,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5

ptime=0
ctime=0
while True:
    success, img = cap.read()
    imgRgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    result=hands.process(imgRgb)
    # print(result.multi_hand_landmarks)
    if(result.multi_hand_landmarks):
        for handlm in result.multi_hand_landmarks:
            for id, lm in enumerate(handlm.landmark):
                # print(id,lm)
                h, w, c = img.shape
                cx, cy = int(lm.x *w), int(lm.y*h)
               
                cv2.circle(img, (cx,cy), 10, (0,255,0), cv2.FILLED)

            mpDraw.draw_landmarks(img,handlm,mphand.HAND_CONNECTIONS)
    
    ctime=time.time()
    fps=1/ (ctime-ptime)
    ptime=ctime
    cv2.putText(img,str(int(fps)),(13,75),cv2.FONT_HERSHEY_PLAIN,2,(0,255,0),2)

    cv2.imshow("Image",img)
    if cv2.waitKey(1) == 13:
        break
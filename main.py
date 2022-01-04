import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
handLmsStyle = mpDraw.DrawingSpec(color=(0,0,255), thickness=5)
handConStyle = mpDraw.DrawingSpec(color=(0,255,0), thickness=10)

while True:
    ret, image = cap.read()
    if ret:
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = hands.process(imageRGB)
        # print(result.multi_hand_landmarks)
        if result.multi_hand_landmarks:
            for handsLms in result.multi_hand_landmarks:
                mpDraw.draw_landmarks(image, handsLms, mpHands.HAND_CONNECTIONS, handLmsStyle, handConStyle)
        cv2.imshow('image', image)

    if cv2.waitKey(1) == ord('q'):
        break
import cv2
import mediapipe as mp
from math import hypot
import screen_brightness_control as sbc
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75,
    max_num_hands=2)

draw_utils = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    landmarks = []
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            for id, lm in enumerate(hand_landmarks.landmark):
                height, width, _ = frame.shape
                x, y = int(lm.x * width), int(lm.y * height)
                landmarks.append([id, x, y])

            draw_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    if landmarks:
        thumb_x, thumb_y = landmarks[4][1], landmarks[4][2]
        index_x, index_y = landmarks[8][1], landmarks[8][2]
        cv2.circle(frame, (thumb_x, thumb_y), 7, (0, 255, 0), cv2.FILLED)
        cv2.circle(frame, (index_x, index_y), 7, (0, 255, 0), cv2.FILLED)
        cv2.line(frame, (thumb_x, thumb_y), (index_x, index_y), (0, 255, 0), 3)
        length = hypot(index_x - thumb_x, index_y - thumb_y)
        brightness = np.interp(length, [15, 220], [0, 100])
        sbc.set_brightness(int(brightness))

    cv2.imshow('Image', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

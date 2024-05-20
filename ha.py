import mediapipe as mp
import numpy as np
import requests
import cv2
import math


home_assistant_url = ""
access_token = ""

entity_id = "light.office_celling"

url = f"{home_assistant_url}/api/services/light/turn_on"

headers = {
    "Authorization": f"Bearer {access_token}",
    "Content-Type": "application/json",
}

data = {
    "entity_id": entity_id,
    "brightness": 150,
}

mpHands = mp.solutions.hands
mpDraw = mp.solutions.drawing_utils
drawing_styles = mp.solutions.drawing_styles
hands = mpHands.Hands()

FRAME_WIDTH = 3440
FRAME_HEIGHT = 1440

def scale_value(x, original_min, original_max, target_min, target_max):
    return (x - original_min) * (target_max - target_min) / (original_max - original_min) + target_min

def moving_average(data, window_size=5):
    cumsum = np.cumsum(np.insert(data, 0, 0, axis=0), axis=0)
    smoothed_data = (cumsum[window_size:] - cumsum[:-window_size]) / window_size
    return smoothed_data

cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()
    # img.flags.writeable = False
    if not success:
        break

    results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if results.multi_hand_landmarks: # Hands presents
        for hlm in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, hlm, )
            p1 = hlm.landmark[7]
            p2 = hlm.landmark[4]
            distance = math.sqrt((p1.x-p2.x)**2+(p1.y-p2.y)**2+(p1.z-p2.z)**2)
            # # print(distance)
            brightness = scale_value(distance, 0.04, 0.4,0,255)
            response = requests.post(url, headers=headers, json={ "entity_id": entity_id, "brightness": brightness})
            # # print(brightness)
            # print(response)

    cv2.imshow("Image", img)
    cv2.waitKey(1)


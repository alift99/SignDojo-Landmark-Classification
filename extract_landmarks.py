import os
import pandas as pd
import numpy as np
import cv2
import mediapipe as mp


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

hands = mp_hands.Hands()

base_path = 'American Sign Language Letters.v1-v1.yolov7pytorch'
folders = ['test', 'train', 'valid']
for folder in folders:
    data = []
    for filename in os.listdir(f'{base_path}/{folder}/images'):
        img = cv2.imread(f'{base_path}/{folder}/images/{filename}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img)
        x_coords = []
        y_coords = []
        x_final = []
        y_final = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                if len(hand_landmarks.landmark) == 21:
                    for i in range(len(hand_landmarks.landmark)):
                        x_coords.append(hand_landmarks.landmark[i].x)
                        y_coords.append(hand_landmarks.landmark[i].y)
                    break

            for i in range(len(x_coords)):
                x = (x_coords[i] - min(x_coords)) / (max(x_coords) - min(x_coords))
                x_final.append(x)

            for i in range(len(y_coords)):
                y = (y_coords[i] - min(y_coords)) / (max(y_coords) - min(y_coords))
                y_final.append(y)

        entry = [filename] + [filename[0]] + x_final + y_final
        if len(entry) == 44:
            data.append(entry)
        else:
            print(len(entry))

    print(len(data))
    x_columns = [f'x_{i}' for i in range(21)]
    y_columns = [f'y_{i}' for i in range(21)]
    cols = ['Filename', 'Label'] + x_columns + y_columns
    df = pd.DataFrame(data, columns=cols)
    df.to_csv(f'{folder}.csv', index=False)

        
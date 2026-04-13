import cv2
import mediapipe as mp
import random
import math
import time

# ---------------- HAND TRACKING SETUP ----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# ---------------- GAME VARIABLES ----------------
prev_x, prev_y = 0, 0
score = 0

fruits = []

# ---------------- MAIN LOOP ----------------
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    h, w, _ = img.shape

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    x, y = 0, 0

    # -------- HAND DETECTION --------
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            x = int(handLms.landmark[8].x * w)
            y = int(handLms.landmark[8].y * h)

            # Draw finger
            cv2.circle(img, (x, y), 10, (0, 255, 0), -1)

            # Draw slice trail
            if prev_x != 0 and prev_y != 0:
                cv2.line(img, (prev_x, prev_y), (x, y), (0, 0, 255), 5)

            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

    # -------- SPEED CALCULATION --------
    speed = math.hypot(x - prev_x, y - prev_y)

    # -------- SPAWN FRUITS --------
    if random.randint(1, 20) == 1:
        fruits.append({
            "x": random.randint(50, w-50),
            "y": h,
            "vx": random.randint(-3, 3),
            "vy": random.randint(-15, -10)
        })

    # -------- UPDATE FRUITS --------
    for fruit in fruits[:]:
        fruit["x"] += fruit["vx"]
        fruit["y"] += fruit["vy"]
        fruit["vy"] += 0.5  # gravity

        # Draw fruit
        cv2.circle(img, (int(fruit["x"]), int(fruit["y"])), 20, (0, 165, 255), -1)

        # -------- COLLISION DETECTION --------
        dist = math.hypot(fruit["x"] - x, fruit["y"] - y)

        if dist < 30 and speed > 20:
            fruits.remove(fruit)
            score += 1

        # Remove if out of screen
        if fruit["y"] > h:
            fruits.remove(fruit)

    # -------- UI --------
    cv2.putText(img, f"Score: {score}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    prev_x, prev_y = x, y

    cv2.imshow("Fruit Ninja Gesture", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
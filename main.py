import cv2
import mediapipe as mp
import random
import math
import os
import winsound

# ---------------- HAND TRACKING ----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# ---------------- LOAD IMAGES ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

apple = cv2.imread(os.path.join(BASE_DIR, "assets", "apple.png"), cv2.IMREAD_UNCHANGED)
bomb = cv2.imread(os.path.join(BASE_DIR, "assets", "bomb.png"), cv2.IMREAD_UNCHANGED)

if apple is None or bomb is None:
    print("❌ ERROR: Images not found!")
    exit()

apple = cv2.resize(apple, (80, 80))
bomb = cv2.resize(bomb, (80, 80))

# ---------------- GAME VARIABLES ----------------
prev_x, prev_y = 0, 0
score = 0
ai_score = 0
lives = 3

game_over = False
game_started = False

fruits = []

# 🤖 AI position
ai_x, ai_y = 300, 300

start_time = cv2.getTickCount()

# ---------------- DRAW IMAGE ----------------
def draw_image(frame, img, x, y):
    h, w = img.shape[:2]
    x1 = int(x - w / 2)
    y1 = int(y - h / 2)

    if x1 < 0 or y1 < 0 or x1 + w > frame.shape[1] or y1 + h > frame.shape[0]:
        return

    roi = frame[y1:y1+h, x1:x1+w]

    if img.shape[2] == 4:
        alpha = img[:, :, 3] / 255.0
        for c in range(3):
            roi[:, :, c] = (1 - alpha) * roi[:, :, c] + alpha * img[:, :, c]
    else:
        roi[:] = img[:, :, :3]

# ---------------- MAIN LOOP ----------------
while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    x, y = -100, -100

    # -------- HAND DETECTION --------
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            x = int(handLms.landmark[8].x * w)
            y = int(handLms.landmark[8].y * h)

            cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)

            if prev_x != 0 and prev_y != 0:
                cv2.line(frame, (prev_x, prev_y), (x, y), (0, 0, 255), 5)

            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

    speed = math.hypot(x - prev_x, y - prev_y)
    elapsed = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()

    # ================= START SCREEN =================
    if not game_started:
        cv2.putText(frame, "Press S to Start", (w//2 - 200, h//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    # ================= GAME =================
    if game_started and not game_over:

        # -------- SPAWN --------
        if random.randint(1, 8) == 1:
            fruits.append({
                "x": random.randint(50, w - 50),
                "y": h - 50,
                "vx": random.randint(-3, 3),
                "vy": random.randint(-18, -12),
                "type": random.choice(["apple", "bomb"])
            })

        # -------- UPDATE FRUITS --------
        for fruit in fruits[:]:
            fruit["x"] += fruit["vx"]
            fruit["y"] += fruit["vy"]
            fruit["vy"] += 0.5

            # DRAW
            if fruit["type"] == "apple":
                draw_image(frame, apple, fruit["x"], fruit["y"])
            else:
                draw_image(frame, bomb, fruit["x"], fruit["y"])

            # -------- PLAYER COLLISION --------
            dist = math.hypot(fruit["x"] - x, fruit["y"] - y)

            if x != -100 and elapsed > 2 and dist < 50 and speed > 40:

                if fruit["type"] == "apple":
                    score += 1
                    winsound.Beep(800, 80)
                else:
                    lives -= 1
                    winsound.Beep(300, 150)

                    if lives <= 0:
                        game_over = True

                fruits.remove(fruit)
                continue

            # remove if out
            if fruit["y"] > h:
                fruits.remove(fruit)

        # -------- SMART AI --------
        if len(fruits) > 0:
            target = min(fruits, key=lambda f: math.hypot(f["x"] - ai_x, f["y"] - ai_y))

            ai_speed = 0.18   # 🔥 increase for harder AI

            ai_x += (target["x"] - ai_x) * ai_speed
            ai_y += (target["y"] - ai_y) * ai_speed

            ai_dist = math.hypot(target["x"] - ai_x, target["y"] - ai_y)

            # AI reaction + randomness
            if ai_dist < 40 and random.random() > 0.2:

                if target["type"] == "apple":
                    ai_score += 1
                else:
                    # AI sometimes makes mistake
                    if random.random() < 0.3:
                        pass  # ignore bomb mostly

                if target in fruits:
                    fruits.remove(target)

    # -------- DRAW AI --------
    cv2.circle(frame, (int(ai_x), int(ai_y)), 12, (255, 0, 0), -1)
    cv2.putText(frame, "AI", (int(ai_x)-10, int(ai_y)-20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # -------- UI --------
    cv2.putText(frame, f"You: {score}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.putText(frame, f"AI: {ai_score}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.putText(frame, f"Lives: {lives}", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # -------- GAME OVER SCREEN --------
    if game_over:
        frame[:] = (0, 0, 100)

        result = "YOU WIN!" if score > ai_score else "AI WINS!"

        cv2.putText(frame, result, (w//2 - 150, h//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)

        cv2.putText(frame, f"You: {score}", (w//2 - 120, h//2 + 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.putText(frame, f"AI: {ai_score}", (w//2 - 120, h//2 + 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.putText(frame, "Press R to Restart", (w//2 - 200, h//2 + 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    prev_x, prev_y = x, y

    cv2.imshow("Gesture Fruit Ninja", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        game_started = True
        game_over = False
        score = 0
        ai_score = 0
        lives = 3
        fruits.clear()
        start_time = cv2.getTickCount()

    if key == ord('r') and game_over:
        game_over = False
        score = 0
        ai_score = 0
        lives = 3
        fruits.clear()
        start_time = cv2.getTickCount()

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
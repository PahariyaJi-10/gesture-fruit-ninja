import cv2
import mediapipe as mp
import random
import math
import os

# ---------------- HAND TRACKING ----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# ---------------- LOAD IMAGES ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

apple_path = os.path.join(BASE_DIR, "assets", "apple.png")
bomb_path = os.path.join(BASE_DIR, "assets", "bomb.png")

apple = cv2.imread(apple_path, cv2.IMREAD_UNCHANGED)
bomb = cv2.imread(bomb_path, cv2.IMREAD_UNCHANGED)

if apple is None or bomb is None:
    print("❌ ERROR: Images not found!")
    exit()

# Resize
apple = cv2.resize(apple, (80, 80))
bomb = cv2.resize(bomb, (80, 80))

# ---------------- GAME VARIABLES ----------------
prev_x, prev_y = 0, 0
score = 0
game_over = False
fruits = []
explosions = []

# ---------------- DRAW IMAGE (FIXED) ----------------
def draw_image(frame, img, x, y):
    h, w = img.shape[:2]
    x1 = int(x - w / 2)
    y1 = int(y - h / 2)

    if x1 < 0 or y1 < 0 or x1 + w > frame.shape[1] or y1 + h > frame.shape[0]:
        return

    # ✅ If image is empty (your case)
    if img.sum() == 0:
        cv2.circle(frame, (int(x), int(y)), 30, (0, 165, 255), -1)
        return

    roi = frame[y1:y1+h, x1:x1+w]

    # ✅ If PNG has alpha
    if img.shape[2] == 4:
        alpha = img[:, :, 3] / 255.0
        for c in range(3):
            roi[:, :, c] = (1 - alpha) * roi[:, :, c] + alpha * img[:, :, c]
    else:
        roi[:] = img
print("Apple sum:", apple.sum())
print("Bomb sum:", bomb.sum())

# ---------------- MAIN LOOP ----------------
while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    x, y = 0, 0

    # -------- HAND DETECTION --------
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            x = int(handLms.landmark[8].x * w)
            y = int(handLms.landmark[8].y * h)

            cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)

            if prev_x != 0 and prev_y != 0:
                cv2.line(frame, (prev_x, prev_y), (x, y), (0, 0, 255), 5)

            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

    # -------- SPEED --------
    speed = math.hypot(x - prev_x, y - prev_y)

    # -------- SPAWN --------
    if not game_over and random.randint(1,5) == 1:
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

        # COLLISION
        dist = math.hypot(fruit["x"] - x, fruit["y"] - y)

        if not game_over and dist < 40 and speed > 20:
            if fruit["type"] == "apple":
                score += 1
            else:
                game_over = True
                explosions.append({
                    "x": fruit["x"],
                    "y": fruit["y"],
                    "radius": 10
                })
            fruits.remove(fruit)

        # REMOVE IF OUT
        if fruit["y"] > h:
            fruits.remove(fruit)

    # -------- EXPLOSION EFFECT --------
    for exp in explosions[:]:
        cv2.circle(frame, (int(exp["x"]), int(exp["y"])),
                   int(exp["radius"]), (0, 0, 255), 3)
        exp["radius"] += 10

        if exp["radius"] > 100:
            explosions.remove(exp)

    # -------- UI --------
    cv2.putText(frame, f"Score: {score}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    if game_over:
        frame[:] = (0, 0, 100)
        cv2.putText(frame, "GAME OVER", (w//2 - 150, h//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

    prev_x, prev_y = x, y

    cv2.imshow("Gesture Fruit Ninja", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
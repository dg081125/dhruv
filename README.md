# dhruv
this is my first git repositroy
Auther Dhruv gupta
import cv2
import numpy as np
import time
import random

def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY)
    return thresh

def get_contour(thresh):
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        return max(contours, key=cv2.contourArea)
    return None

def compare_shapes(contour1, contour2):
    if contour1 is None or contour2 is None:
        return None
    return cv2.matchShapes(contour1, contour2, cv2.CONTOURS_MATCH_I1, 0.0)

# Load target challenge shape (example image)
target_img = cv2.imread("target_shape.png")
if target_img is None:
    print("Error: target_shape.png not found!")
    exit()

target_thresh = preprocess_image(target_img)
target_contour = get_contour(target_thresh)

cap = cv2.VideoCapture(0)

rounds = 3
previous_score = 0

for r in range(1, rounds+1):
    print(f"\nRound {r} starting... Prepare your movement!")

    start_time = time.time()
    captured_contour = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera not accessible")
            break

        elapsed = int(time.time() - start_time)
        remaining = max(0, 10 - elapsed)
        cv2.putText(frame, f"Round {r}: {remaining}s left",
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Movement Challenge", frame)

        if elapsed >= 10:
            hand_thresh = preprocess_image(frame)
            captured_contour = get_contour(hand_thresh)
            break

        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

    # Compare shapes
    similarity = compare_shapes(target_contour, captured_contour)
    if similarity is not None:
        video_score = max(0, int((1 - similarity) * 100))
    else:
        video_score = 0

    # Simulated PAM activity data
    steps = random.randint(50, 200)
    heart_rate = random.randint(80, 140)
    activity_score = int((steps/200)*50 + (heart_rate/140)*50)

    # Final score
    total_score = int((video_score + activity_score) / 2)

    print(f"Video Score: {video_score}/100")
    print(f"Activity Data → Steps: {steps}, HR: {heart_rate} bpm, Activity Score: {activity_score}/100")
    print(f"Total Round {r} Score: {total_score}/100")

    if total_score > previous_score:
        print("🎉 You beat your previous score!")
    else:
        print("Keep trying to improve!")

    previous_score = total_score

cap.release()
cv2.destroyAllWindows()

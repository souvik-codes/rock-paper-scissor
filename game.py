import cv2
import random
import time
from ultralytics import YOLO

# Load trained YOLO model
model = YOLO("runs/detect/train/weights/best.pt")  # Replace with your trained model path

# Define class names (must match your model)
class_names = ["paper", "stone", "scissors"]

# Game logic: who beats whom
def get_winner(player, computer):
    if player == computer:
        return "Draw"
    elif (player == "stone" and computer == "scissors") or \
         (player == "scissors" and computer == "paper") or \
         (player == "paper" and computer == "stone"):
        return "You Win!"
    else:
        return "Computer Wins!"

# Initialize webcam
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX

print("Get ready! Show your hand when prompted...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip for mirror view
    frame = cv2.flip(frame, 1)
    cv2.putText(frame, "Press SPACE to play!", (10, 30), font, 1, (0, 255, 255), 2)
    cv2.imshow("Rock Paper Scissors", frame)

    key = cv2.waitKey(1)
    if key == 27:  # ESC to quit
        break
    elif key == 32:  # SPACE to play
        countdown = 3
        while countdown > 0:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            cv2.putText(frame, f"Show your hand in... {countdown}", (50, 200), font, 1.5, (0, 0, 255), 3)
            cv2.imshow("Rock Paper Scissors", frame)
            cv2.waitKey(1000)
            countdown -= 1

        # Capture final frame
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        # Run YOLO detection
        results = model.predict(source=frame, conf=0.3, verbose=True)
        predicted_class = None

        for r in results:
            if len(r.boxes.cls) > 0:
                cls_id = int(r.boxes.cls[0])
                predicted_class = class_names[cls_id]
                break

        # Random computer move
        computer_choice = random.choice(class_names)

        if predicted_class:
            result_text = get_winner(predicted_class, computer_choice)
            print(f"You: {predicted_class}, Computer: {computer_choice} --> {result_text}")
        else:
            result_text = "No hand gesture detected!"
            print(result_text)

        # Show results on frame
        cv2.putText(frame, f"Your Move: {predicted_class}", (10, 50), font, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Computer: {computer_choice}", (10, 100), font, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Result: {result_text}", (10, 150), font, 1.2, (0, 255, 0), 3)

        # Display final result
        cv2.imshow("Rock Paper Scissors", frame)
        cv2.waitKey(3000)  # Wait 3 seconds before resuming

cap.release()
cv2.destroyAllWindows()

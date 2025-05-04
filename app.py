import time
import tkinter as tk
from tkinter import Label, Button
import cv2
import threading
from PIL import Image, ImageTk
import random
from ultralytics import YOLO

# Load your trained YOLOv8 model
model = YOLO("runs/detect/train/weights/best.pt")
class_names = ["paper", "stone", "scissors"]

# Game logic
def get_winner(player, computer):
    if player == computer:
        return "Draw"
    elif (player == "stone" and computer == "scissors") or \
         (player == "scissors" and computer == "paper") or \
         (player == "paper" and computer == "stone"):
        return "You Win!"
    else:
        return "Computer Wins!"

# Initialize GUI
root = tk.Tk()
root.title("Rock-Paper-Scissors with YOLO")
root.geometry("800x600")

video_label = Label(root)
video_label.pack()

result_label = Label(root, text="", font=("Arial", 18))
result_label.pack(pady=10)

cap = cv2.VideoCapture(0)

def update_video():
    ret, frame = cap.read()
    if ret:
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)
    root.after(10, update_video)

def play_game():
    countdown_label = Label(root, text="Get ready...", font=("Arial", 16))
    countdown_label.pack()

    def count_and_capture():
        for i in range(3, 0, -1):
            countdown_label.config(text=f"Show your hand in... {i}")
            time.sleep(1)

        ret, frame = cap.read()
        if not ret:
            result_label.config(text="Camera error!")
            return

        frame = cv2.flip(frame, 1)
        results = model.predict(source=frame, conf=0.3, verbose=False)
        predicted_class = None

        for r in results:
            if len(r.boxes.cls) > 0:
                cls_id = int(r.boxes.cls[0])
                predicted_class = class_names[cls_id]
                break

        computer_choice = random.choice(class_names)
        if predicted_class:
            result = get_winner(predicted_class, computer_choice)
            result_label.config(text=f"You: {predicted_class} | Computer: {computer_choice}\nResult: {result}")
        else:
            result_label.config(text="No hand detected!")

        countdown_label.destroy()

    threading.Thread(target=count_and_capture).start()

play_button = Button(root, text="Play", font=("Arial", 14), command=play_game)
play_button.pack(pady=10)

update_video()
root.protocol("WM_DELETE_WINDOW", lambda: (cap.release(), root.destroy()))
root.mainloop()

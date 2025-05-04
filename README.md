# 🪨📄✂️ Rock-Paper-Scissors Game using YOLOv11 and OpenCV

A real-time Rock-Paper-Scissors game built using **YOLO11 object detection**, **OpenCV**, and **Tkinter GUI**. The system captures your hand gesture using your webcam, detects whether you're showing "rock", "paper", or "scissors", and then plays the game against a random computer choice!


## 🚀 Features

- Real-time hand gesture detection using a custom-trained **YOLOv11** model.
- User interface built with **Tkinter**.
- Countdown timer before capturing gesture.
- Simple game logic with instant result display.
- Trained on a dataset sourced from **Roboflow**.

## 🧠 Model Info

- Model: YOLOv8n (nano)  
- Trained on: Rock-Paper-Scissors dataset from Roboflow  
- Classes: `paper`, `stone`, `scissors`  
- Format: PyTorch `.pt` file (exported from Ultralytics training run)

## 📁 Dataset

The dataset was sourced from [Roboflow](https://roboflow.com) and contains labeled images of hand gestures for rock, paper, and scissors.

## 📷 Requirements

- Python 3.8+
- Webcam

### Python Libraries

Install dependencies with:

```bash
pip install ultralytics opencv-python Pillow

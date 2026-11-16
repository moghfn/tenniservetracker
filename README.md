# 🎾 Real-Time Tennis Serve & Court Tracking System

An automated, computer vision-powered tennis tracking system that uses **YOLOv8**, **OpenCV (Optical Flow)**, and a **Raspberry Pi video stream** to track tennis balls in real time, detect bounces, classify serves (In/Out, Wide, Down the T, Deuce/Ad), and sync live match stats directly to a **Firebase Realtime Database**.

---

## 🚀 Features

* **🎥 Live Stream Integration:** Connects remotely to a Raspberry Pi camera stream via Tailscale with an automated retry loop.
* **📐 Interactive Court Calibration:** Click 4 court corners on the first frame to establish perspective transform boundaries (persisted locally in `court_config.json`).
* **🧠 YOLOv8 Ball Detection:** Real-time object detection tuned to isolate sports balls while filtering out non-ball objects.
* **🔄 Optical Flow & Anti-Rolling Filter:** Uses Lucas-Kanade optical flow for stable court tracking and filters out stationary or rolling balls to prevent false triggers.
* **🎯 Automated Serve & Bounce Solver:** Interpolates trajectories to pinpoint bounce locations and judge whether serves land **IN**, **WIDE**, or **LONG**, along with placement details (Down the T, Wide, Body).
* **☁️ Firebase Cloud Sync:** Instantly pushes live serve statistics and timestamps to your Firebase Realtime Database.

---

## 📦 Prerequisites & Dependencies

Make sure you have Python 3.8+ installed. You can install all required packages using pip:

```bash
pip install -r requirements.txt

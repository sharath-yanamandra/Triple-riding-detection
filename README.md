# Triple-riding-detection
Here’s a sample `README.md` file for your **Triple Riding Detection** project based on the code you've shared:

---

## 🚦 Triple Riding Detection

A computer vision project to detect triple riding (three people riding on a two-wheeler) using object detection models. The solution leverages pre-trained YOLO models and OpenCV for video analysis to enhance road safety and automate traffic rule enforcement.

---

### 📁 Project Structure

```
.
├── triple_riding.py        # Main script for inference using webcam or video
├── triple.ipynb            # Jupyter notebook for step-by-step visualization and testing
├── best.pt                 # Trained YOLO model for triple riding detection
├── sample_video.mp4        # (Optional) Input video for demo
└── README.md               # Project documentation
```

---

### 🚀 Features

- Detects people on two-wheelers in real-time
- Counts the number of riders per vehicle
- Alerts when more than two people are detected on a bike
- Displays bounding boxes and annotations on video frames

---

### 🛠️ Requirements

Install required libraries using pip:

```bash
pip install torch torchvision opencv-python ultralytics
```

---

### 📦 Setup & Usage

#### 1. Clone the repository

```bash
git clone https://github.com/your-username/triple-riding-detection.git
cd triple-riding-detection
```

#### 2. Add your trained YOLO model

Place the trained YOLO model file (`best.pt`) in the root directory.

#### 3. Run the main script

To use webcam input:

```bash
python triple_riding.py
```

Or use a video file:

```bash
python triple_riding.py --video_path sample_video.mp4
```

> Note: You can modify the script to read from CCTV feeds or IP cameras.

---

### 🧠 Model

This project uses a custom-trained YOLOv8 model fine-tuned on a dataset with annotated instances of triple riding. The model returns bounding boxes, class labels, and confidence scores.

---

### 🎯 Output

- Annotated video feed with:
  - Rider count per bike
  - "TRIPLE RIDING DETECTED" warning when more than two people are detected

---

### 📓 Notebook Version

If you prefer an interactive walkthrough, use `triple.ipynb` in Jupyter Notebook to visualize the detection pipeline step-by-step.

---

### ✅ To-Do

- [ ] Improve rider count accuracy
- [ ] Add license plate recognition
- [ ] Alert system integration (email, SMS)
- [ ] Deploy on edge devices (e.g., Jetson Nano)

---

### 📃 License

MIT License

---

Let me know if you want to include sections like **Dataset Info**, **Training Details**, or **Model Evaluation** too!

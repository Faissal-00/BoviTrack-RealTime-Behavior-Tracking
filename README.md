# 🐄 BoviTrack: Real-Time Cattle Behavior Tracking & Monitoring

An end-to-end, edge-optimized computer vision pipeline designed to track and classify cattle behavior in real-time from farm surveillance feeds. 

Instead of relying on a single heavy network, **BoviTrack** uses a decoupled **dual-stage architecture** to maintain high throughput and lightweight memory overhead suitable for edge deployment:
1. **Target Localization & Tracking:** A lightweight YOLO model detects cattle in the frame, paired with **ByteTrack** to maintain persistent IDs across frames despite occlusions and low-light transitions.
2. **Behavior Classification:** High-confidence detections are cropped and dynamically routed to a specialized YOLO classification head to identify target behaviors (`eat`, `drink`, `stand`, `lie`, `ruminate`).

---

## 📊 Dataset & Training

* **Dataset Source:** [Beef Cattle Behavior Dataset (Kaggle)](https://www.kaggle.com/datasets/lucyfirst/beef-cattle-behavior-data-set/data)
* **Training Pipeline:** All data auditing, video frame extraction (with strict video-level splits to eliminate data leakage), and model training were conducted on Kaggle using NVIDIA T4 GPU acceleration.
* **Pretrained Weights:** Both optimized Nano-scale models are pre-packaged in the `final_models/` directory for plug-and-play inference.

---

## 📁 Repository Structure

```text
BoviTrack-RealTime-Behavior-Tracking/
├── data/
│   └── bovine_data.yaml              # Dataset configuration
├── final_models/
│   ├── detector_best.pt              # Trained YOLO cattle detector
│   └── classifier_best.pt            # Trained YOLO behavior classifier
├── notebooks/
│   └── notebook58828f625d.ipynb      # Kaggle training & validation pipeline
├── scripts/
│   └── main.py                       # Real-time inference & video processing script
├── videos/                           # Directory for input and output videos
└── README.md
```

---

## 🚀 Quick Start & Inference

### 1. Clone the Repository
```bash
git clone [https://github.com/Faissal-00/BoviTrack-RealTime-Behavior-Tracking.git](https://github.com/Faissal-00/BoviTrack-RealTime-Behavior-Tracking.git)
cd BoviTrack-RealTime-Behavior-Tracking
```

### 2. Install Dependencies
```bash
pip install ultralytics opencv-python numpy
```

### 3. Add Input Footage
Create a `videos` directory in the project root if it does not already exist, and place your target `.mp4` video files inside it:
```bash
mkdir -p videos
# Copy your test video (e.g., day_video.mp4 / night_video.mp4) into the videos/ folder
```

### 4. Run the Pipeline
Execute the main deployment script:
```bash
python scripts/main.py
```

The script will automatically load the models from `final_models/`, process your video frame-by-frame with active ByteTrack ID persistence and behavior overlay badges, and export the final annotated video directly into the `videos/` folder.

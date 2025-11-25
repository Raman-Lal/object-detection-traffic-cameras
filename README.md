# Object Detection for Traffic Cameras

## Project Overview  
This project explores how computer vision can be used to support traffic monitoring by automatically detecting vehicles and pedestrians from video footage. The system uses the YOLO object detection model together with OpenCV to analyse video frames, count detected objects, and generate basic traffic behaviour insights.  
The goal is to create a simplified prototype that demonstrates how automated traffic analysis can support Smart City applications such as monitoring congestion, tracking flow patterns, and assisting road management.

## Objectives  
- Detect and classify common traffic objects such as cars, buses, trucks, motorcycles, bicycles, and pedestrians.  
- Measure system performance using FPS (processing speed) and estimated accuracy.  
- Count objects per frame to extract basic traffic behaviour.  
- Compare detection performance in three different environmental conditions (4K daytime, 1080p daytime, night-time rain).  
- Produce visual outputs including annotated video, screenshots, CSV logs, and detection graphs.

## Tools and Technologies  
- Python 3.10+  
- YOLOv8 (Ultralytics)  
- OpenCV  
- NumPy  
- Matplotlib  

## Phase Status  
- Phase 1 - Conception: Topic selection, abstract creation, and planning completed.  
- Phase 2 - Development Preparation: Folder structure, environment setup, and research overview completed.  
- Phase 3 - Development and Evaluation: Full implementation completed. Results analysed across three different video scenarios. All outputs generated for evaluation.

---

# Test Cases  
The following test cases were used to verify the functionality of the detection system. These cases ensure that each component behaves as expected under different conditions.

### 1. Video File Input Test  
**Purpose:** Verify that the script can successfully open and process a user-provided video.  
**Expected Result:** The script loads the video without errors and begins frame-by-frame processing.

### 2. Object Detection Test  
**Purpose:** Confirm that YOLOv8 correctly detects cars, buses, pedestrians, and other classes.  
**Expected Result:** Bounding boxes and labels appear on objects with non-zero confidence scores.

### 3. Counting Logic Test  
**Purpose:** Ensure that the per-frame object counts are correct and logged properly.  
**Expected Result:** The numbers in `results.csv` match the visible detections on the annotated video.

### 4. Screenshot Saving Test  
**Purpose:** Test whether screenshots are generated at the specified interval.  
**Expected Result:** Annotated frames are saved in `outputs/screenshots/` without errors.

### 5. Graph Generation Test  
**Purpose:** Validate that the system creates a graph showing total detections per frame.  
**Expected Result:** `detections_graph.png` is generated with a correct time-based detection trend.

### 6. Multi-Environment Performance Test  
**Purpose:** Compare system performance in different conditions:  
- Daytime 4K  
- Daytime 1080p  
- Night-time rain  
**Expected Result:** FPS and accuracy vary predictably, providing meaningful evaluation data.

---

# Detection Reliability Risks  
The system performs well under standard lighting and clear visibility; however, several risks affect detection reliability:

### 1. Low-Light Conditions  
Object boundaries become unclear, leading to missed detections or low-confidence predictions.

### 2. Rain and Weather Effects  
Raindrops, reflections, and blurred surfaces can confuse the model, resulting in false positives or incorrect labels.

### 3. Headlight Glare  
Night traffic produces bright glare that reduces contrast, making vehicles and pedestrians harder to detect.

### 4. Overlapping Objects  
In dense traffic scenes, multiple vehicles overlap, making accurate classification more difficult.

### 5. High Motion Blur  
Fast-moving vehicles or shaky cameras reduce the sharpness of objects, lowering detection quality.

### 6. Resolution Dependency  
Higher-resolution videos (like 4K) significantly reduce FPS due to increased computational load.

These risks are addressed in the system evaluation and will be further discussed in the final project report’s reflection section.

---

## Project Goal  
The purpose of the project is to build a small but complete traffic detection pipeline that can process video input, detect multiple objects in real time, track vehicle density, and generate analytical outputs.  
While the technology is not new, the project demonstrates how different tools can be combined into an educational, transparent, and reproducible prototype suitable for academic evaluation.

## Evaluation Summary (Phase 3)  
The system was tested on three different traffic videos to compare performance under different conditions:

| Video Type | Resolution | Avg FPS | Accuracy (Est.) | Total Detections | Notes |
|------------|------------|---------|------------------|------------------|-------|
| Daytime 4K | 3840×2160 | 9.44 | ~88% | 31,485 | High detail, many objects, slowest FPS |
| Daytime 1080p | 1920×1080 | 22.98 | ~90% | 17,485 | Best balance of speed and accuracy |
| Night + Rain 4K | 3840×2160 | 27.56 | ~75% | 7,271 | Faster FPS due to fewer visible objects, but accuracy reduced due to glare and reflections |

### Chart Summary  
A detections-per-frame chart is generated automatically after each run.  
It visualises how many objects were detected in each frame, allowing traffic density patterns and performance differences to be clearly observed across environments.

## Detection Reliability Risk  
YOLO performs well in clear, daylight conditions.  
However, reliability decreases in:  
- low light  
- rainy conditions  
- nighttime glare  
- frames with motion blur  
- scenes with many overlapping vehicles  

This aligns with known limitations in real-world traffic camera analysis.

## Tech Stack  
- YOLOv8 for object detection  
- OpenCV for image and video processing  
- Python for implementation  
- CUDA acceleration when available

## Folder Structure  
```
Objec-Detection-for-Traffic-Cameras/
│
├── src/
│   └── detect_phase3.py           # Main detection script
│
├── data/
│   └── README.md                  # Placeholder (videos not included)
│
├── outputs/
│   ├── README.md                  # outputs description
│   ├── results.csv                # Per-frame detections
│   ├── detections_graph.png       # Graph of total detections per frame
│   └── screenshots/
│       └── *.jpg                  # Automatically saved detection frames
│
├── venv/                          # Virtual environment (not uploaded)
│
├── README.md                      # Project documentation
└── requirements.txt               # Python dependencies
```

## Important Note About Video Files  
Traffic videos used for testing are not included in this repository due to copyright restrictions and file size limitations.  
Users can test the system using their own footage by placing video files in the `data/` folder.

## Author  
Raman Lal  
Master’s in Computer Science  
IU International University of Applied Sciences

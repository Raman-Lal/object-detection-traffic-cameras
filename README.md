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

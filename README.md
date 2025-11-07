# Object Detection for Traffic Cameras

## Project Overview
Traffic congestion and road safety remain major challenges in modern cities. Although traffic cameras are used widely for monitoring, manually checking video footage is slow and prone to human error.
This project develops an automated system that detects, labels, and counts vehicles from traffic camera footage in real time.

By using Python, OpenCV, and YOLOv8 (You Only Look Once) — a fast and efficient deep learning model — the system can recognize cars, buses, trucks, motorcycles, bicycles, and pedestrians. It also measures performance through Frames Per Second (FPS) and detection accuracy, while introducing a behavioral feature for vehicle counting to estimate traffic flow.

## Project goal
- Detect and label traffic-related objects (cars, buses, trucks, bikes, pedestrians) automatically.  
- Count vehicles passing through video frames to analyze road activity. 
- Measure detection speed (FPS) and accuracy for performance evaluation.
- Provide a foundation for future smart-city analytics such as traffic density or risk prediction.

## Tools and Technologies (Planned)
- **Python 3.10+**  
- **OpenCV** for video frame processing  
- **YOLOv8** for object detection  
## Environment
- Works on Windows, macOS, and Linux
- Supports CPU and GPU (GPU recommended for better FPS)

## Detection Reliability Risk
Detection results can vary depending on video conditions. The main reliability risks include:
- Lighting Conditions: Detection accuracy drops in dark or low-light footage.
- Weather Effects: Rain, fog, or glare can hide or blur vehicles.
- Camera Angle: Distant or side-angled views can reduce precision.
- Occlusion: Overlapping vehicles may cause missed detections or miscounts.
Possible Solutions:
- Use high-resolution videos for testing.
- Adjust YOLO confidence threshold (e.g., 0.3–0.5).
- Train or fine-tune YOLO on traffic-specific datasets for improved accuracy.

## Folder Structure
object-detection-traffic-cameras/
│
├── data/                     # csv files
│   ├── ideal.csv
│   ├── test.csv
│   └── train.csv
├── models/                     
│   ├── base.py
│   ├── data_loader.py        # CSV loading
│   ├── database.py           # SQLite integration
│   ├── deviation.py          # Test point deviation logic
│   ├── matcher.py            # Function matching logic
│   └── plotter.py            # Bokeh visualizations
├── tests/                     # Unit tests
│   ├── test_matcher.py
│   └── test_deviation.py
├── utils/                     
│  └── exception.py
├── main.py                    # Runs the complete pipeline
├── requirenments.txt          # Requirnments for running the program   
└── README.md

## Current Status
*Phase 1 – Conception Phase*  
- Project topic selected and abstract completed.
- Tools and methods defined (Python, OpenCV, YOLO).
*Phase 2 - Development Phase (Current)*
- Prototype implemented and tested.
- Measured FPS (~12 on CPU) and detection accuracy (~88%).
- Added a behavioral feature for vehicle counting to analyze traffic flow.
*Phase 3 - Final Phase(Planned)*
- Improve detection reliability under various conditions.
- Add simple visual charts (vehicle count vs time).
- Finalize the full project report and presentation.

## Author
**Raman Lal**  
Master’s in Computer Science, IU International University of Applied Sciences

---

*This repository is part of the “Project: Computer Science (CSEMCSPCSP01)” course portfolio.*

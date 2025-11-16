"""
detect_phase3.py
- Uses Ultralytics YOLOv8 for detection (GPU if available)
- Counts vehicles per frame and total
- Saves annotated video, CSV log, screenshots, and a graph of detections

Run (after installing requirements and activating venv):
for activating venv:
    cd "D:\Project Computer Science\Objec-Detection-for-Traffic-Cameras"
    python -m venv venv
    venv\Scripts\Activate.ps1
if Powershell blocks script:
    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
    venv\Scripts\Activate.ps1
    
then run this in powershell:
    python src\detect_phase3.py --source data/traffic.mp4 --output outputs/output_video.mp4 --screenshots_every 50
"""

import os
import time
import argparse
from collections import defaultdict

import cv2
import pandas as pd
import matplotlib.pyplot as plt

# try to import torch to check for GPU; if not present, script will still run with CPU
try:
    import torch
except Exception:
    torch = None

# simple list of classes we consider "vehicles" (COCO names)
VEHICLE_CLASSES = {'car', 'bus', 'truck', 'motorcycle', 'bicycle'}
OTHER_CLASSES = {'person'}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Traffic detection + counting (Phase 3)')
    parser.add_argument('--source', type=str, default='../data/traffic.mp4',
                        help='Path to input video file (or camera index like 0)')
    parser.add_argument('--output', type=str, default='../outputs/output_video.mp4',
                        help='Path for saving annotated output video')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                        help='YOLOv8 model file, e.g. yolov8n.pt or yolov8s.pt')
    parser.add_argument('--conf', type=float, default=0.35,
                        help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='NMS IoU threshold')
    parser.add_argument('--screenshots_every', type=int, default=50,
                        help='Save one annotated screenshot every N frames (0 = disabled)')
    parser.add_argument('--save_csv', action='store_true',
                        help='Save per-frame results to CSV')
    args = parser.parse_args()
    return args


def ensure_folders(output_path):
    out_dir = os.path.dirname(output_path) or '.'
    screenshots_dir = os.path.join(out_dir, 'screenshots')
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(screenshots_dir, exist_ok=True)
    return screenshots_dir


def get_device_name():
    if torch is not None and torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def draw_box(frame, x1, y1, x2, y2, label, conf):
    color = (0, 200, 0) if label in VEHICLE_CLASSES else (50, 150, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    text = f"{label} {conf:.2f}"
    cv2.putText(frame, text, (x1, max(15, y1-7)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)


def main():
    args = parse_args()

    # local import of ultralytics (delayed so script can show helpful error)
    try:
        from ultralytics import YOLO
    except Exception as e:
        print("Please install the 'ultralytics' package. See requirements.txt or use pip.")
        raise e

    # prepare folders
    screenshots_dir = ensure_folders(args.output)

    # decide device (GPU if available)
    device = get_device_name()
    print(f"[INFO] Using device: {device}")

    # load model (will download if not present)
    model = YOLO(args.model)
    # set model to run on chosen device (ultralytics auto handles device selection,
    # but we print this for clarity)
    try:
        model.to(device)
    except Exception:
        # Some ultralytics versions ignore .to - that's okay
        pass

    # open video
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video source: {args.source}")

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(
        f"[INFO] Input video: {args.source} ({width}x{height} @ {fps_in:.2f} FPS)")

    # prepare writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps_in, (width, height))

    total_frames = 0
    total_time = 0.0
    per_frame_counts = []  # list of dicts for csv/plot
    total_counts = defaultdict(int)  # overall counts per class

    # main loop
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        total_frames += 1
        t0 = time.time()

        # run detection - pass the numpy frame directly
        results = model.predict(
            source=frame, conf=args.conf, iou=args.iou, verbose=False, max_det=300)

        t1 = time.time()
        elapsed = t1 - t0
        total_time += elapsed
        fps_current = 1.0 / elapsed if elapsed > 0 else 0.0

        # parse results (ultralytics returns a list; we take first)
        frame_counts = {'frame': total_frames,
                        'fps': fps_current, 'vehicles': 0, 'people': 0}
        if len(results) > 0:
            res = results[0]
            # boxes may be empty
            if hasattr(res, 'boxes') and res.boxes is not None:
                for box in res.boxes:
                    try:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        name = model.names.get(cls, str(cls))
                    except Exception:
                        # fallback to safe parsing
                        continue

                    # count vehicles vs people
                    if name in VEHICLE_CLASSES:
                        frame_counts['vehicles'] += 1
                        total_counts[name] += 1
                    elif name in OTHER_CLASSES:
                        frame_counts['people'] += 1
                        total_counts[name] += 1

                    # draw box on frame
                    draw_box(frame, x1, y1, x2, y2, name, conf)

        # save annotated frame to output video
        out.write(frame)

        # save a screenshot every N frames if requested
        if args.screenshots_every and (total_frames % args.screenshots_every == 0):
            shot_path = os.path.join(
                screenshots_dir, f"frame_{total_frames:06d}.jpg")
            cv2.imwrite(shot_path, frame)

        per_frame_counts.append(frame_counts)

        # small status update every 100 frames
        if total_frames % 100 == 0:
            avg_fps = (total_frames / total_time) if total_time > 0 else 0.0
            print(
                f"[INFO] Frames: {total_frames} | Avg FPS: {avg_fps:.2f} | Last frame FPS: {fps_current:.2f}")

    # cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    avg_fps = (total_frames / total_time) if total_time > 0 else 0.0
    print("=== Run summary ===")
    print(f"Frames processed: {total_frames}")
    print(f"Average processing FPS: {avg_fps:.2f}")
    print("Detections total by class:")
    for k, v in total_counts.items():
        print(f"  {k}: {v}")

    # save csv if requested (default: save)
    df = pd.DataFrame(per_frame_counts)
    csv_path = os.path.join(os.path.dirname(args.output), 'results.csv')
    df.to_csv(csv_path, index=False)
    print(f"[INFO] Per-frame results saved to: {csv_path}")

    # simple plot: vehicles per frame (smoothed)
    try:
        plt.figure(figsize=(10, 4))
        plt.plot(df['frame'], df['vehicles'], label='Vehicles/frame')
        plt.xlabel('Frame')
        plt.ylabel('Vehicles detected')
        plt.title('Vehicle detections per frame')
        plt.legend()
        plt.grid(True)
        graph_path = os.path.join(os.path.dirname(
            args.output), 'detections_graph.png')
        plt.savefig(graph_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"[INFO] Graph saved to: {graph_path}")
    except Exception as e:
        print("Warning: could not create plot (matplotlib may be missing).", e)

    print("All done. You can see the CSV, graph, screenshots, and annotated video.")


if __name__ == "__main__":
    main()

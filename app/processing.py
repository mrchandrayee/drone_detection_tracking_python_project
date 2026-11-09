from pathlib import Path
import cv2
from .detector import DroneDetector

def process_video(source, destination, model_path="yolo11n.pt", device="cpu",
                  confidence=.25, imgsz=640, tracker="bytetrack.yaml",
                  draw_trail=True, box_only=False):
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video source: {source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        raise ValueError("Invalid video dimensions")
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width,height))
    detector = DroneDetector(model_path,device,confidence,imgsz,tracker,draw_trail=draw_trail,box_only=box_only)
    frame_count, total_detections, seen_ids = 0, 0, set()
    try:
        while True:
            ok, frame = cap.read()
            if not ok: break
            annotated, detections = detector.process_frame(frame)
            total_detections += len(detections)
            for d in detections:
                if d["track_id"] is not None: seen_ids.add(d["track_id"])
            cv2.putText(annotated, f"DETECTIONS: {len(detections)}  TRACKED IDS: {len(seen_ids)}",
                        (12,28),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,255,80),2,cv2.LINE_AA)
            writer.write(annotated)
            frame_count += 1
    finally:
        cap.release(); writer.release()
    return {"frames_processed": frame_count, "duration_seconds": round(frame_count/fps,2),
            "detection_instances": total_detections, "unique_track_ids": len(seen_ids)}


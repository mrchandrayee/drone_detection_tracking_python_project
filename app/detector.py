from pathlib import Path
from collections import defaultdict, deque
import cv2
from ultralytics import YOLO

class DroneDetector:
    """YOLO inference wrapper with persistent multi-object tracking."""
    def __init__(self, model_path="yolo11n.pt", device="cpu", confidence=0.25,
                 imgsz=640, tracker="bytetrack.yaml", classes=None,
                 draw_trail=True, box_only=False):
        if not Path(model_path).exists() and not str(model_path).endswith(".pt"):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        self.model = YOLO(str(model_path))
        self.device = device
        self.confidence = float(confidence)
        self.imgsz = int(imgsz)
        self.tracker = tracker
        self.classes = classes
        self.draw_trail = draw_trail
        self.box_only = box_only
        self.trails = defaultdict(lambda: deque(maxlen=30))

    def process_frame(self, frame):
        results = self.model.track(
            source=frame, persist=True, tracker=self.tracker,
            conf=self.confidence, imgsz=self.imgsz,
            classes=self.classes, device=self.device, verbose=False
        )
        result = results[0]
        annotated = frame.copy()
        if result.boxes is None:
            return annotated, []
        names = result.names
        detections = []
        boxes = result.boxes
        ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(boxes)
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.int().cpu().tolist()
        for box, score, cls_id, track_id in zip(xyxy, confs, clss, ids):
            x1,y1,x2,y2 = map(int, box)
            label = names.get(cls_id, str(cls_id))
            color = ((37 * (track_id or cls_id+1)) % 255,
                     (97 * (track_id or cls_id+1)) % 255,
                     (173 * (track_id or cls_id+1)) % 255)
            if not self.box_only:
                cv2.rectangle(annotated, (x1,y1), (x2,y2), color, 2)
                tag = f"ID {track_id} | {label} {score:.2f}" if track_id is not None else f"{label} {score:.2f}"
                cv2.rectangle(annotated, (x1, max(0,y1-24)), (x1+min(260, max(100,len(tag)*8)), y1), color, -1)
                cv2.putText(annotated, tag, (x1+4,max(16,y1-7)), cv2.FONT_HERSHEY_SIMPLEX, .48, (255,255,255), 1, cv2.LINE_AA)
            if track_id is not None:
                cx, cy = (x1+x2)//2, (y1+y2)//2
                self.trails[track_id].append((cx,cy))
                if self.draw_trail and len(self.trails[track_id]) > 1:
                    pts = list(self.trails[track_id])
                    for a,b in zip(pts,pts[1:]):
                        cv2.line(annotated,a,b,color,2)
            detections.append({"track_id": track_id, "class_id": cls_id,
                               "label": label, "confidence": round(float(score),4),
                               "bbox": [x1,y1,x2,y2]})
        return annotated, detections

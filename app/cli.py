import argparse
from .processing import process_video

def main():
    p=argparse.ArgumentParser(description="YOLO drone detection and tracking")
    p.add_argument("--source",required=True,help="Video file or webcam index (0)")
    p.add_argument("--model",default="models/drone.pt",help="Drone-trained .pt/.onnx model")
    p.add_argument("--tracker",default="bytetrack.yaml")
    p.add_argument("--confidence",type=float,default=.25)
    p.add_argument("--imgsz",type=int,default=640)
    p.add_argument("--device",default="cpu",help="cpu, 0, cuda:0")
    p.add_argument("--output",default="outputs/tracked.mp4")
    p.add_argument("--show",action="store_true")
    a=p.parse_args()
    source=int(a.source) if a.source.isdigit() else a.source
    print(process_video(source,a.output,a.model,a.device,a.confidence,a.imgsz,a.tracker))
if __name__=="__main__": main()

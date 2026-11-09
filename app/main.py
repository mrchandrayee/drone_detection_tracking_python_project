from pathlib import Path
import os, shutil, uuid
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from .processing import process_video

BASE=Path(__file__).resolve().parent.parent
UPLOADS=BASE/"uploads"; OUTPUTS=BASE/"outputs"
UPLOADS.mkdir(exist_ok=True); OUTPUTS.mkdir(exist_ok=True)
app=FastAPI(title="Drone Detection & Tracking")
app.mount("/static",StaticFiles(directory=str(BASE/"app"/"static")),name="static")

@app.get("/",response_class=HTMLResponse)
def home(): return (BASE/"app"/"static"/"index.html").read_text(encoding="utf-8")

@app.post("/api/analyze")
async def analyze(file: UploadFile=File(...), model_path: str=Form("yolo11n.pt"),
                  confidence: float=Form(.25), tracker: str=Form("bytetrack.yaml"),
                  device: str=Form("cpu")):
    ext=Path(file.filename or "").suffix.lower()
    if ext not in {".mp4",".avi",".mov",".mkv"}: raise HTTPException(400,"Upload MP4, AVI, MOV or MKV.")
    if not (model_path=="yolo11n.pt" or Path(model_path).is_file()):
        raise HTTPException(400,"Model path not found. Upload/place a trained model and enter its path.")
    job=uuid.uuid4().hex[:12]; src=UPLOADS/f"{job}{ext}"; out=OUTPUTS/f"{job}_tracked.mp4"
    with src.open("wb") as f: shutil.copyfileobj(file.file,f)
    try:
        stats=process_video(src,out,model_path=model_path,device=device,confidence=confidence,tracker=tracker)
        return {"job_id":job,"stats":stats,"result_url":f"/api/result/{job}_tracked.mp4",
                "model":model_path,"warning":None if model_path!="yolo11n.pt" else "Generic COCO model selected; it does not have a dedicated drone class. Use a drone-trained model for actual drone detection."}
    except Exception as e: raise HTTPException(500,str(e)) from e
    finally: src.unlink(missing_ok=True)

@app.get("/api/result/{filename}")
def result(filename:str):
    path=OUTPUTS/filename
    if not path.exists() or path.parent!=OUTPUTS: raise HTTPException(404,"Result not found")
    return FileResponse(path,media_type="video/mp4",filename=filename)

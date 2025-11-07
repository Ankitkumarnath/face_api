from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from PIL import Image
from io import BytesIO
from typing import Optional

from face_service import recognize_faces, register_face, auto_register_if_new

app = FastAPI(title="Face Detection & Registration API (M1-safe)")

def _read_image(upload: UploadFile) -> Image.Image:
    raw = upload.file.read()
    return Image.open(BytesIO(raw)).convert("RGB")

@app.get("/")
def root():
    return {"ok": True, "service": "face-api", "endpoints": ["/detect", "/register", "/detect-or-register"]}

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    img = _read_image(file)
    result = recognize_faces(img)
    return JSONResponse({"ok": True, "results": result})

@app.post("/register")
async def register(
    file: UploadFile = File(...),
    note: Optional[str] = Form(None)
):
    img = _read_image(file)
    res = register_face(img, note=note)
    return JSONResponse({"ok": True, **res})

@app.post("/detect-or-register")
async def detect_or_register(
    file: UploadFile = File(...),
    auto_register: bool = Form(True)
):
    img = _read_image(file)
    res = auto_register_if_new(img, auto_register=auto_register)
    return JSONResponse({"ok": True, **res})

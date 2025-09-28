# app.py
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional
from pathlib import Path
import io

from PIL import Image
import torch
from torch import nn
from torchvision import transforms as T
import torchvision

# --------- Paths ---------
CKPT_DIR = Path("checkpoints_multiclass_strong")
DATA_DIR = Path("disaster-ai/data/xbd/tier1")  # used when a relative path is sent

# --------- Model helpers ---------
def build_model(backbone: str, n_classes: int):
    if backbone == "resnet18":
        m = torchvision.models.resnet18(weights=None); m.fc = nn.Linear(m.fc.in_features, n_classes)
    elif backbone == "resnet50":
        m = torchvision.models.resnet50(weights=None); m.fc = nn.Linear(m.fc.in_features, n_classes)
    elif backbone == "efficientnet_b0":
        m = torchvision.models.efficientnet_b0(weights=None); m.classifier[1] = nn.Linear(m.classifier[1].in_features, n_classes)
    elif backbone == "vit_b_16":
        m = torchvision.models.vit_b_16(weights=None); m.heads.head = nn.Linear(m.heads.head.in_features, n_classes)
    else:
        raise ValueError(f"Unknown backbone: {backbone}")
    return m

def load_pipeline(ckpt_dir: Path = CKPT_DIR):
    ckpt = torch.load(ckpt_dir / "best.pt", map_location="cpu")
    classes  = ckpt["classes"]
    img_size = ckpt["img_size"]
    backbone = ckpt["backbone"]

    model = build_model(backbone, len(classes))
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    tform = T.Compose([
        T.Resize(int(img_size*1.2)), T.CenterCrop(img_size),
        T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    return model, tform, classes

MODEL, TFORM, CLASSES = load_pipeline()

def _predict_image_pil(img: Image.Image):
    x = TFORM(img).unsqueeze(0)
    with torch.no_grad():
        logits = MODEL(x)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        top = int(torch.argmax(probs).item())
    return CLASSES[top], float(probs[top])

def _resolve_path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (DATA_DIR / p)

# --------- FastAPI app ---------
app = FastAPI(title="Damage Multiclass API", version="1.1")

class PredictResponse(BaseModel):
    image_path: Optional[str] = None
    pred: str
    conf: float

class PathsRequest(BaseModel):
    image_paths: List[str]

@app.get("/health")
def health():
    return {"status": "ok", "num_classes": len(CLASSES)}

@app.get("/labels")
def labels():
    return {"classes": CLASSES}

# Single file
@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    pred, conf = _predict_image_pil(img)
    return {"image_path": file.filename, "pred": pred, "conf": conf}

# Multiple uploaded files
@app.post("/predict-multi", response_model=List[PredictResponse])
async def predict_multi(files: List[UploadFile] = File(...)):
    out: List[PredictResponse] = []
    for f in files:
        b = await f.read()
        img = Image.open(io.BytesIO(b)).convert("RGB")
        pred, conf = _predict_image_pil(img)
        out.append(PredictResponse(image_path=f.filename, pred=pred, conf=conf))
    return out

# Paths on disk (absolute or relative to DATA_DIR)
@app.post("/predict-paths", response_model=List[PredictResponse])
def predict_paths(req: PathsRequest):
    out: List[PredictResponse] = []
    for p in req.image_paths:
        path = _resolve_path(p)
        img = Image.open(path).convert("RGB")
        pred, conf = _predict_image_pil(img)
        out.append(PredictResponse(image_path=str(path), pred=pred, conf=conf))
    return out

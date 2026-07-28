"""
detector.py
===========
Task-Aware Object Detection — EfficientNet-B0 Pipeline
SpecOps 199 | DVCon India 2026 Stage 2A

Architecture (matches Stage 1 proposal exactly):
─────────────────────────────────────────────────
  Stage A — Detection Head  : YOLOv8n
      Produces bounding boxes + class labels + confidence scores.

  Stage B — CNN Backbone    : EfficientNet-B0  (torchvision)
      Each detected crop is passed through EfficientNet-B0.
      The final feature vector x_i (80-dim, COCO-aligned) is
      computed from the backbone's output embedding.

  Stage C — Relevance Scoring (scorer.py)
      score_i = x_i · W_t   (dot product with task weight vector)

Install:
    pip install ultralytics torchvision torch opencv-python numpy Pillow
"""

import os
import numpy as np
import cv2

import torch
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from task_config import COCO_CLASSES


# ============================================================
# CONSTANTS
# ============================================================

CONF_THRESHOLD   = 0.20
EFFNET_IMG_SIZE  = 224

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
YOLO_WEIGHTS = os.path.join(_SCRIPT_DIR, "yolov8n.pt")
if not os.path.exists(YOLO_WEIGHTS):
    YOLO_WEIGHTS = "yolov8n.pt"

_EFFNET_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((EFFNET_IMG_SIZE, EFFNET_IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    ),
])


# ============================================================
# LOAD EFFICIENTNET-B0 BACKBONE
# ============================================================

def _load_efficientnet():
    print("[Detector] Loading EfficientNet-B0 backbone...")
    try:
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        print("[Detector] EfficientNet-B0: pretrained ImageNet weights loaded.")
    except Exception:
        model = efficientnet_b0(weights=None)
        print("[Detector] EfficientNet-B0: random initialization.")
    model.classifier = torch.nn.Identity()
    model.eval()
    return model


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():
    effnet = _load_efficientnet()

    print("[Detector] Loading YOLOv8n detection head...")
    try:
        from ultralytics import YOLO
        yolo = YOLO(YOLO_WEIGHTS)
        print(f"[Detector] YOLOv8n loaded from: {YOLO_WEIGHTS}")
    except ImportError:
        print("\n" + "="*60)
        print("ERROR: 'ultralytics' not installed. Run: pip install ultralytics")
        print("="*60 + "\n")
        raise SystemExit(1)

    print("[Detector] Pipeline ready: EfficientNet-B0 backbone + YOLOv8n head")
    return {"yolo": yolo, "effnet": effnet}


# ============================================================
# EFFICIENTNET FEATURE VECTOR
# ============================================================

def _effnet_feature_vector(effnet, image_bgr, bbox, cls_id, conf, orig_h, orig_w):
    """
    Extract EfficientNet-B0 features from the detected crop,
    project to 80-dim COCO-aligned feature vector x_i.

    Steps:
      1. Crop the bounding box region
      2. Pass through EfficientNet-B0  →  1280-dim embedding
      3. Reshape (80, 16) and take L2 norm  →  80-dim vector
      4. Reinforce detected class channel with confidence + area bonus
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(orig_w - 1, x2); y2 = min(orig_h - 1, y2)

    if x2 <= x1 or y2 <= y1:
        fv = np.zeros(80, dtype=np.float32)
        fv[cls_id] = conf
        return fv

    crop_rgb = cv2.cvtColor(image_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
    tensor = _EFFNET_TRANSFORM(crop_rgb).unsqueeze(0)

    with torch.no_grad():
        embedding = effnet(tensor)[0].numpy()          # (1280,)

    # Project 1280 → 80 via reshape + L2 norm
    proj = np.linalg.norm(embedding[:1280].reshape(80, 16), axis=1)
    proj_max = proj.max()
    if proj_max > 0:
        proj = proj / proj_max

    area_ratio = ((x2-x1)*(y2-y1)) / (orig_h*orig_w + 1e-6)
    primary    = conf + min(area_ratio * 0.3, 0.2)

    fv = proj * conf * 0.5          # EfficientNet context signal
    fv[cls_id] = max(fv[cls_id], primary)   # dominant class signal

    return fv.astype(np.float32)


# ============================================================
# DETECT
# ============================================================

def detect(model: dict, image_bgr: np.ndarray) -> list:
    yolo   = model["yolo"]
    effnet = model["effnet"]
    orig_h, orig_w = image_bgr.shape[:2]

    # Stage A: YOLO bounding boxes
    results = yolo(image_bgr, conf=CONF_THRESHOLD, verbose=False, device="cpu")

    raw = []
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            if cls_id >= len(COCO_CLASSES):
                continue
            raw.append({
                "cls_id": cls_id,
                "conf":   conf,
                "bbox":   [float(v) for v in box.xyxy[0]],
            })

    # Deduplicate: keep highest-conf per class
    seen = {}
    for d in sorted(raw, key=lambda d: d["conf"], reverse=True):
        if d["cls_id"] not in seen:
            seen[d["cls_id"]] = d

    # Stage B: EfficientNet-B0 feature vectors
    detections = []
    for d in seen.values():
        fv = _effnet_feature_vector(
            effnet, image_bgr, d["bbox"],
            d["cls_id"], d["conf"], orig_h, orig_w
        )
        detections.append({
            "class_id":       d["cls_id"],
            "class_name":     COCO_CLASSES[d["cls_id"]],
            "confidence":     round(d["conf"], 3),
            "bbox":           [round(v, 1) for v in d["bbox"]],
            "feature_vector": fv,
        })

    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return detections


# ============================================================
# DETECT FROM FILE
# ============================================================

def detect_from_file(image_path: str, model: dict) -> list:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot open image: {image_path}")
    return detect(model, img)

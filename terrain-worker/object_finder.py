"""
object_finder.py — Open-Vocabulary Object Detection using OWL-ViT.
Returns 2D bounding boxes with labels and confidence scores, which the
frontend maps onto the 3D terrain using its own depth math.
"""

import io
import base64
import logging
import time

import numpy as np
from PIL import Image
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from transformers import pipeline

logger = logging.getLogger("depthwizard.object_finder")
router = APIRouter()

# Lazy-loaded singleton
_detector = None
_load_time = None

def _get_detector():
    global _detector, _load_time
    if _detector is None:
        logger.info("Loading OWL-ViT model...")
        t0 = time.time()
        _detector = pipeline(
            task="zero-shot-object-detection",
            model="google/owlv2-base-patch16-ensemble",
            device=-1,  # CPU. Use 0 for CUDA
        )
        _load_time = time.time() - t0
        logger.info(f"OWL-ViT loaded in {_load_time:.2f}s")
    return _detector

@router.post("/detect-objects")
async def detect_objects(
    file: UploadFile = File(...),
    query: str = Form(...),
    threshold: float = Form(default=0.15),
):
    raw = await file.read()
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Could not read image: {e}")

    detector = _get_detector()
    
    t0 = time.time()
    try:
        results = detector(img, candidate_labels=[query], threshold=threshold)
    except Exception as e:
        logger.exception("OWL-ViT inference failed")
        raise HTTPException(500, f"Model inference error: {e}")

    inference_ms = (time.time() - t0) * 1000
    
    w, h = img.size
    objects = []
    for res in results:
        box = res["box"]
        cx = (box["xmin"] + box["xmax"]) / 2 / w
        cy = (box["ymin"] + box["ymax"]) / 2 / h
        
        objects.append({
            "label": res["label"],
            "score": round(res["score"], 3),
            "center_x": round(cx, 5),
            "center_y": round(cy, 5),
            "inference_ms": round(inference_ms, 1),
        })

    logger.info(f"Detected {len(objects)} objects for query: {query}")
    
    # CREATE THE SHOWCASE MESSAGE
    message = f"OWL-ViT loaded in {_load_time:.2f}s | Detected {len(objects)} objects for query: '{query}' in {inference_ms:.0f}ms"
    
    return {"objects": objects, "message": message}
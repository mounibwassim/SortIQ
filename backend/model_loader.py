import sys
import os
import io
import base64
import json
import logging
import gc
import builtins
from typing import Any, List, Dict, Optional, Tuple

import numpy as np
from PIL import Image
import cv2
from ultralytics import YOLO

# Set TF to use legacy Keras 2 logic
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from logger import logger

# Initialize global storage in builtins for truly global state across any module reloads
if not hasattr(builtins, '_sortiq_model_instance'):
    builtins._sortiq_model_instance = None

# Constants
HARD_BLOCK = ["person", "face"]

# Monkey patch torch.load before any ultralytics imports if possible, 
# though ultralytics might have already imported torch. 
# We'll do it here to be sure.
try:
    import torch
    if not hasattr(builtins, '_torch_load_patched'):
        original_load = torch.load
        def patched_load(*args, **kwargs):
            if 'weights_only' not in kwargs:
                kwargs['weights_only'] = False
            return original_load(*args, **kwargs)
        torch.load = patched_load
        builtins._torch_load_patched = True
        logger.info("Torch.load patched to allow all globals (needed for YOLO)")
except Exception as e:
    logger.warning(f"Failed to patch torch.load: {e}")

def is_face_or_skin(crop: np.ndarray) -> bool:
    if crop is None or crop.size == 0: return False
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    lower = np.array([0, 20, 70], dtype=np.uint8)
    upper = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    skin_ratio = np.sum(mask > 0) / (crop.shape[0] * crop.shape[1])
    return skin_ratio > 0.4

def is_background(box, frame_w, frame_h, crop) -> bool:
    x1, y1, x2, y2 = box
    box_area = (x2 - x1) * (y2 - y1)
    if box_area < 800: return True
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if var < 15: return True
    return False

class SortIQModel:
    def __init__(self):
        self.version = os.getenv("MODEL_VERSION", "v1.0")
        self.threshold = float(os.getenv("CONFIDENCE_THRESHOLD", 0.30))
        self.bin_mapping = {
            "Plastic": {"bin": "Blue", "colorHex": "#3b82f6"},
            "Paper": {"bin": "Green", "colorHex": "#22c55e"},
            "Metal": {"bin": "Yellow", "colorHex": "#eab308"},
            "Glass": {"bin": "Red", "colorHex": "#ef4444"}
        }
        self.model = None
        self.yolo_model = None
        self.classes = {}
        self.loaded = False

    def load(self):
        # 1. Load YOLOv8n
        try:
            logger.info("Loading YOLOv8n model...")
            self.yolo_model = YOLO("yolov8n.pt")
            logger.info("YOLOv8n loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load YOLO: {e}")

        # 2. Load MobileNetV2
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.getenv("MODEL_PATH", os.path.join(base_dir, "model", "sortiq_model.h5")).replace("\\", "/")
        
        if not os.path.exists(model_path):
             logger.error(f"MODEL FILE MISSING: {model_path}")
             model_path = "./model/sortiq_model.h5"
             
        import tensorflow as tf
        import tf_keras
        
        logger.info(f"Loading MobileNetV2 from {model_path}...")
        try:
            from tensorflow.keras.layers import BatchNormalization
            self.model = tf_keras.models.load_model(
                model_path, 
                compile=False,
                custom_objects={'BatchNormalization': BatchNormalization}
            )
            self.loaded = True
            logger.info(f"Model loaded successfully in PID {os.getpid()}")
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            raise RuntimeError(f"Failed to load model: {e}")

        # 3. Load Classes
        classes_path = os.path.join(os.path.dirname(model_path), "classes.json").replace("\\", "/")
        if os.path.exists(classes_path):
            try:
                with open(classes_path, "r") as f:
                    raw = json.load(f)
                cl = {int(k): v for k, v in raw.items()}
                cl.update({str(k): v for k, v in raw.items()})
                self.classes = cl
            except:
                self.classes = {0: "Glass", 1: "Metal", 2: "Paper", 3: "Plastic"}
        else:
            self.classes = {0: "Glass", 1: "Metal", 2: "Paper", 3: "Plastic"}

    def predict_scene(self, img_pil: Image.Image, color_overrides: Dict[str, str] = None) -> List[Dict[str, Any]]:
        m = self.model
        y = self.yolo_model
        if m is None or y is None:
            return []
        img_rgb = np.array(img_pil)
        frame_h, frame_w = img_rgb.shape[:2]

        def run_mobilenet(crop_rgb, box, yolo_hint: str = None):
            try:
                crop_resized = cv2.resize(crop_rgb, (224, 224))
                crop_batch = np.expand_dims(crop_resized.astype(np.float32) / 255.0, axis=0)
                preds = m.predict(crop_batch, verbose=0)
                
                # Get Glass (0) and Metal (1) probabilities
                glass_prob = float(preds[0][0])
                metal_prob = float(preds[0][1])
                
                cls_idx = int(np.argmax(preds[0]))
                conf = float(preds[0][cls_idx])
                cls_label = self.classes.get(cls_idx, "Unknown")

                # Glass vs Metal Boost: If YOLO sees a bottle/cup/glass, 
                # but MobileNet says Metal, check if Glass is also likely.
                glass_hints = ["bottle", "wine glass", "cup"]
                if yolo_hint in glass_hints and cls_label == "Metal":
                    if glass_prob > 0.15: # Significant glass probability
                        logger.info(f"[BIAS] YOLO saw {yolo_hint}, MobileNet said Metal. Flipped to Glass (prob {glass_prob:.2f})")
                        cls_label = "Glass"
                        conf = glass_prob

                if conf < self.threshold:
                    return None
                    
                mapped = self.bin_mapping.get(cls_label, {"bin": "Recycling", "colorHex": "#22c55e"})
                final_color = color_overrides.get(cls_label, mapped["colorHex"]) if color_overrides else mapped["colorHex"]
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                h_pos = "Left" if cx < frame_w/3 else ("Right" if cx > 2*frame_w/3 else "Center")
                v_pos = "Top" if cy < frame_h/3 else ("Bottom" if cy > 2*frame_h/3 else "Middle")
                
                return {
                    "is_waste": True,
                    "label": cls_label,
                    "raw_label": cls_label.lower(),
                    "confidence": conf,
                    "bin_color": mapped["bin"],
                    "color_hex": final_color,
                    "box_color": mapped["bin"],
                    "box_color_hex": final_color,
                    "box": [x1, y1, x2, y2],
                    "location": f"{v_pos}-{h_pos}",
                    "message": self._waste_message(cls_label),
                    "tip": self._waste_tip(cls_label),
                    "interaction_type": "waste",
                }
            except Exception as e:
                logger.error(f"MobileNetV2 error: {e}")
                return None

        final_results = []
        results = y.predict(img_pil, conf=0.20, verbose=False)

        for res in results:
            if len(res.boxes) == 0:
                continue
            boxes = res.boxes.xyxy.cpu().numpy()
            classes = res.boxes.cls.cpu().numpy()
            for i in range(len(boxes)):
                box = boxes[i].astype(int)
                yolo_label = y.names[int(classes[i])]
                if yolo_label in HARD_BLOCK:
                    continue
                x1, y1, x2, y2 = box
                crop = img_rgb[max(0,y1):min(frame_h,y2), max(0,x1):min(frame_w,x2)]
                if crop.size == 0:
                    continue
                if is_face_or_skin(crop):
                    continue
                if is_background(box, frame_w, frame_h, crop):
                    continue
                det = run_mobilenet(crop, box, yolo_hint=yolo_label)
                if det:
                    final_results.append(det)

        # Fallback: YOLO found nothing useful — run MobileNetV2 on full image
        if not final_results:
            logger.info("[FALLBACK] No YOLO detections — running MobileNetV2 on full image")
            full_crop = img_rgb
            full_box = [0, 0, frame_w, frame_h]
            if not is_face_or_skin(full_crop):
                det = run_mobilenet(full_crop, full_box)
                if det:
                    det["box"] = [10, 10, frame_w-10, frame_h-10]
                    final_results.append(det)

        return final_results

    def _waste_message(self, cls: str) -> str:
        msgs = {
            "Plastic": "♻️ Plastic detected! Place in the plastic recycling bin. Remember to rinse and remove caps!",
            "Glass":   "♻️ Glass detected! Place in the glass recycling bin. Handle with care!",
            "Metal":   "♻️ Metal detected! Place in the metal recycling bin. Great job recycling!",
            "Paper":   "♻️ Paper detected! All kinds of paper can be recycled. Great job!",
        }
        return msgs.get(cls, f"♻️ {cls} detected! Place in the recycling bin.")

    def _waste_tip(self, cls: str) -> str:
        tips = {
            "Plastic": "Rinse before placing. Remove caps and labels.",
            "Glass":   "Glass is 100% recyclable forever!",
            "Metal":   "Aluminum cans can be recycled infinitely!",
            "Paper":   "Most paper products are highly recyclable!",
        }
        return tips.get(cls, "Place in the correct recycling bin.")

def get_model():
    if not builtins._sortiq_model_instance:
        builtins._sortiq_model_instance = SortIQModel()
    return builtins._sortiq_model_instance

def ensure_models_loaded():
    model = get_model()
    if not model.loaded or model.model is None:
        logger.info(f"Initializing models globally in PID {os.getpid()}...")
        model.load()

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

# ── Complete item lists per category ──────────────────────────

GLASS_ITEMS = {
    "wine glass", "cup", "vase", "jar", "bowl",
    "bottle", "glass bottle", "wine bottle",
    "beer bottle", "juice bottle", "sauce bottle",
    "medicine bottle", "perfume bottle",
    "glass cup", "drinking glass", "champagne glass",
    "shot glass", "glass mug", "glass jar",
    "jam jar", "pickle jar", "honey jar",
    "pasta sauce jar", "mason jar", "glass bowl",
    "glass vase", "glass candle", "glass plate",
    "glass dish", "broken glass", "light bulb",
    "glass figurine", "glass frame",
}

PLASTIC_ITEMS = {
    "bottle", "plastic bottle", "water bottle",
    "soda bottle", "juice bottle", "milk jug",
    "shampoo bottle", "detergent bottle",
    "plastic bag", "grocery bag", "ziplock bag",
    "plastic container", "takeaway container",
    "yogurt cup", "plastic cup", "plastic lid",
    "plastic tray", "styrofoam box", "foam cup",
    "bottle cap", "straw", "plastic straw",
    "plastic spoon", "plastic fork",
    "plastic wrap", "food packaging", "bag",
    "suitcase", "frisbee", "toothbrush",
    "cell phone", "remote", "clock",
}

METAL_ITEMS = {
    "can", "tin", "fork", "knife", "spoon",
    "scissors", "aluminum can", "soda can",
    "beer can", "energy drink can", "food tin",
    "canned food", "paint can", "aerosol can",
    "metal fork", "metal knife", "metal spoon",
    "metal straw", "metal bowl", "aluminum foil",
    "foil tray", "metal bottle cap", "metal key",
    "metal coin", "metal ruler", "paper clip",
    "staple", "metal wire", "metal pipe",
    "oven", "microwave", "toaster", "sink",
}

PAPER_ITEMS = {
    "book", "newspaper", "envelope", "cardboard",
    "box", "paper", "white paper", "lined paper",
    "printed paper", "magazine", "notebook",
    "notepad", "textbook", "comic book",
    "paperback", "diary", "journal",
    "cardboard box", "cardboard sheet",
    "pizza box", "cereal box", "shoe box",
    "delivery box", "egg carton", "paper bag",
    "paper cup", "paper plate", "paper tray",
    "gift wrap", "tissue paper", "receipt",
    "ticket", "brochure", "flyer", "poster",
    "sticky note", "calendar", "greeting card",
    "keyboard", "mouse", "remote", "laptop",
}

# YOLO label → material hint mapping
YOLO_MATERIAL_HINTS = {
    # Glass hints — YOLO says these → boost glass
    "wine glass": "Glass",
    "cup":        "Glass",
    "vase":       "Glass",
    "jar":        "Glass",
    "bowl":       "Glass",
    # Plastic hints
    "bottle":     "Plastic",
    "bag":        "Plastic",
    "suitcase":   "Plastic",
    "frisbee":    "Plastic",
    "toothbrush": "Plastic",
    "cell phone": "Plastic",
    "remote":     "Plastic",
    # Metal hints
    "can":        "Metal",
    "tin":        "Metal",
    "fork":       "Metal",
    "knife":      "Metal",
    "spoon":      "Metal",
    "scissors":   "Metal",
    "oven":       "Metal",
    "microwave":  "Metal",
    "sink":       "Metal",
    # Paper hints
    "book":       "Paper",
    "newspaper":  "Paper",
    "envelope":   "Paper",
    "cardboard":  "Paper",
    "keyboard":   "Paper",
    # Food override — run metal check first
    "donut":      "Metal",
    "sandwich":   "Metal",
    "pizza":      "Paper",
}

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

def detect_glass_signals(crop_rgb: np.ndarray) -> float:
    """
    Returns a glass confidence boost score 0.0 to 1.0.
    Uses 6 computer vision techniques to detect glass.
    """
    try:
        crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
        hsv      = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        gray     = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

        sat  = float(np.mean(hsv[:,:,1]))
        bri  = float(np.mean(hsv[:,:,2]))
        var  = float(np.var(gray))

        r = float(np.mean(crop_rgb[:,:,0]))
        g = float(np.mean(crop_rgb[:,:,1]))
        b = float(np.mean(crop_rgb[:,:,2]))
        channel_diff = max(abs(r-g), abs(g-b), abs(r-b))

        _, bright_mask = cv2.threshold(
            gray, 220, 255, cv2.THRESH_BINARY
        )
        bright_ratio = float(np.sum(bright_mask>0)) / bright_mask.size

        edges      = cv2.Canny(gray, 50, 150)
        edge_d     = float(np.sum(edges>0)) / edges.size

        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_var = float(np.var(np.sqrt(sobelx**2+sobely**2)))

        score = 0
        # Transparency — RGB channels similar = transparent
        if channel_diff < 20: score += 3
        elif channel_diff < 35: score += 1
        # Specular highlights
        if 0.02 < bright_ratio < 0.30: score += 2
        # Hard sharp edges
        if edge_d > 0.06: score += 2
        # Refraction gradient
        if gradient_var > 3000: score += 2
        # Low saturation = not colored
        if sat < 30: score += 2
        elif sat < 50: score += 1
        # High brightness = light passes through
        if bri > 170: score += 2
        elif bri > 140: score += 1
        # High variance = specular reflections
        if var > 2500: score += 2
        elif var > 1500: score += 1

        glass_conf = min(score / 17.0, 1.0)
        logger.info(f"[GLASS_DETECT] score={score}/17 conf={glass_conf:.2f} sat={sat:.0f} bri={bri:.0f} ch_diff={channel_diff:.0f}")
        return glass_conf

    except Exception as e:
        logger.warning(f"[GLASS_DETECT] failed: {e}")
        return 0.0

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

        def run_mobilenet(crop_rgb, box, yolo_label=""):
            try:
                crop_resized = cv2.resize(crop_rgb, (224, 224))
                crop_batch   = np.expand_dims(
                    crop_resized.astype(np.float32)/255.0,
                    axis=0
                )
                preds = m.predict(crop_batch, verbose=0)
                probs = preds[0].copy()

                # Get class indices
                rev = {
                    v.lower(): int(k)
                    for k,v in self.classes.items()
                    if isinstance(k, int)
                }
                glass_idx   = rev.get("glass")
                plastic_idx = rev.get("plastic")
                metal_idx   = rev.get("metal")
                paper_idx   = rev.get("paper")

                logger.info(f"[MOBILENET] raw={dict(zip(self.classes.values(), probs.tolist()))}")

                # Apply YOLO material hint boost
                hint = YOLO_MATERIAL_HINTS.get(
                    yolo_label.lower(), ""
                )
                if hint == "Glass" and glass_idx is not None:
                    probs[glass_idx] *= 3.0
                    if plastic_idx is not None:
                        probs[plastic_idx] *= 0.5
                    probs = probs / probs.sum()
                    logger.info(f"[HINT] YOLO '{yolo_label}' → glass boost 3x")

                elif hint == "Metal" and metal_idx is not None:
                    probs[metal_idx] *= 2.5
                    probs = probs / probs.sum()
                    logger.info(f"[HINT] YOLO '{yolo_label}' → metal boost 2.5x")

                elif hint == "Paper" and paper_idx is not None:
                    probs[paper_idx] *= 2.5
                    probs = probs / probs.sum()
                    logger.info(f"[HINT] YOLO '{yolo_label}' → paper boost 2.5x")

                elif hint == "Plastic" and plastic_idx is not None:
                    probs[plastic_idx] *= 1.5
                    probs = probs / probs.sum()

                # Glass detection using CV signals
                if glass_idx is not None and plastic_idx is not None:
                    glass_cv_conf = detect_glass_signals(crop_rgb)
                    if glass_cv_conf > 0.45:
                        boost = 1.0 + glass_cv_conf * 3.0
                        probs[glass_idx]   *= boost
                        probs[plastic_idx] *= 0.4
                        probs = probs / probs.sum()
                        logger.info(f"[GLASS_BOOST] cv_conf={glass_cv_conf:.2f} boost={boost:.1f}x")

                # Metal vs Glass separator
                if metal_idx is not None and glass_idx is not None:
                    hsv_crop = cv2.cvtColor(
                        cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR),
                        cv2.COLOR_BGR2HSV
                    )
                    sat = float(np.mean(hsv_crop[:,:,1]))
                    bri_var = float(np.var(hsv_crop[:,:,2]))
                    # Metal = high brightness variance (specular)
                    if bri_var > 2500 and sat < 40:
                        probs[metal_idx]  *= 2.0
                        probs[glass_idx]  *= 0.6
                        probs = probs / probs.sum()
                        logger.info(f"[METAL_BOOST] bri_var={bri_var:.0f}")

                cls_idx   = int(np.argmax(probs))
                conf      = float(probs[cls_idx])
                cls_label = self.classes.get(cls_idx, "Unknown")

                logger.info(f"[FINAL] {cls_label} {conf*100:.1f}%")

                if conf < self.threshold:
                    return None

                mapped = self.bin_mapping.get(
                    cls_label,
                    {"bin": "Recycling", "colorHex": "#22c55e"}
                )
                final_color = (
                    color_overrides.get(cls_label, mapped["colorHex"])
                    if color_overrides else mapped["colorHex"]
                )
                x1,y1,x2,y2 = int(box[0]),int(box[1]),int(box[2]),int(box[3])
                cx = (x1+x2)//2
                cy = (y1+y2)//2
                h_pos = "Left" if cx < frame_w/3 else ("Right" if cx > 2*frame_w/3 else "Center")
                v_pos = "Top"  if cy < frame_h/3 else ("Bottom" if cy > 2*frame_h/3 else "Middle")

                return {
                    "is_waste":         True,
                    "label":            cls_label,
                    "raw_label":        cls_label.lower(),
                    "confidence":       conf,
                    "bin_color":        mapped["bin"],
                    "color_hex":        final_color,
                    "box_color":        mapped["bin"],
                    "box_color_hex":    final_color,
                    "box":              [x1,y1,x2,y2],
                    "location":         f"{v_pos}-{h_pos}",
                    "message":          self._waste_message(cls_label),
                    "tip":              self._waste_tip(cls_label),
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
                det = run_mobilenet(crop, box, yolo_label)
                if det:
                    final_results.append(det)

        # Fallback: YOLO found nothing useful — run MobileNetV2 on full image
        if not final_results:
            logger.info("[FALLBACK] No YOLO detections — running MobileNetV2 on full image")
            full_crop = img_rgb
            full_box = [0, 0, frame_w, frame_h]
            if not is_face_or_skin(full_crop):
                det = run_mobilenet(full_crop, full_box, "")
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

import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
import io
import base64
import json
import numpy as np  # pyre-ignore
import tensorflow as tf  # pyre-ignore
from tensorflow.keras.layers import BatchNormalization as KerasBatchNormalization  # pyre-ignore
from ultralytics import YOLO  # pyre-ignore
import cv2  # pyre-ignore
from PIL import Image  # pyre-ignore
from logger import logger  # pyre-ignore
from typing import List, Dict, Any, Optional

# Load face detector once at startup
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + 
    'haarcascade_frontalface_default.xml'
)

# Handle Keras axis-as-list bug
class BatchNormalization(KerasBatchNormalization):
    def __init__(self, **kwargs):
        if 'axis' in kwargs and isinstance(kwargs['axis'], list):
            kwargs['axis'] = kwargs['axis'][0]
        super().__init__(**kwargs)

# NEVER run MobileNetV2 on these YOLO labels
VALID_WASTE = {"plastic", "metal", "paper", "glass"}
HARD_BLOCK = {
    "person","face","man","woman","child",
    "boy","girl","baby","human",
    "cat","dog","bird","horse","cow",
    "elephant","bear","sheep","giraffe","zebra",
    "tree","plant","flower","grass","sky",
    "bed","couch","sofa","toilet",
    "car","truck","bus","motorcycle",
    "airplane","boat","train",
}

# ONLY run MobileNetV2 on these YOLO labels
WASTE_ONLY = {
    "bottle","wine glass","cup","bowl","vase","jar",
    "can","tin","fork","knife","spoon","scissors",
    "book","newspaper","cardboard","box","backpack",
    "bag","suitcase","toothbrush","umbrella","handbag",
    "cell phone","remote","clock","frisbee",
    "baseball bat","tennis racket","sports ball",
}

def is_face_or_skin(crop_rgb: np.ndarray) -> bool:
    """
    Returns True if crop contains a human face or skin.
    If True → block MobileNetV2 completely.
    """
    try:
        crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
        gray     = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

        # Check 1: Haar face detector
        faces = FACE_CASCADE.detectMultiScale(
            gray, scaleFactor=1.1,
            minNeighbors=4, minSize=(30,30)
        )
        if len(faces) > 0:
            logger.info("[GATE] Face detected → blocking MobileNetV2")
            return True

        # Check 2: Skin tone detection
        hsv  = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv,
            np.array([0,  20,  70], dtype=np.uint8),
            np.array([25, 255, 255], dtype=np.uint8)
        )
        skin_ratio = np.sum(mask > 0) / mask.size
        if skin_ratio > 0.35:
            logger.info(f"[GATE] Skin {skin_ratio:.0%} → blocking MobileNetV2")
            return True

        return False

    except Exception as e:
        logger.warning(f"[GATE] face/skin check failed: {e}")
        return False

def is_background(box, frame_w, frame_h, crop_rgb):
    """
    Returns True if this detection is background/surface.
    Background = floor, wall, table, ceiling.
    Real objects are smaller and more distinct.
    """
    x1, y1, x2, y2 = box
    box_w = x2 - x1
    box_h = y2 - y1
    box_area   = box_w * box_h
    frame_area = frame_w * frame_h
    area_ratio = box_area / frame_area

    # Rule 1: Box covers more than 60% of frame = background
    if area_ratio > 0.60:
        logger.info(f"[BG_REJECT] area_ratio={area_ratio:.2f} > 0.60 → background")
        return True

    # Rule 2: Box touches 3 or more frame edges = background
    margin = 10
    touches_left   = x1 < margin
    touches_right  = x2 > frame_w - margin
    touches_top    = y1 < margin
    touches_bottom = y2 > frame_h - margin
    edges_touched  = sum([touches_left, touches_right,
                          touches_top, touches_bottom])
    if edges_touched >= 3:
        logger.info(f"[BG_REJECT] touches {edges_touched} edges → background")
        return True

    # Rule 3: Very low texture variance = flat surface
    # Walls and floors are very uniform
    gray = cv2.cvtColor(
        cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR),
        cv2.COLOR_BGR2GRAY
    )
    variance = float(np.var(gray))
    if variance < 120 and area_ratio > 0.30:
        logger.info(f"[BG_REJECT] var={variance:.0f} < 120 + large area → flat surface")
        return True

    # Rule 4: Dominant single color = wall or floor
    # Real objects have more color variation
    hsv = cv2.cvtColor(
        cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR),
        cv2.COLOR_BGR2HSV
    )
    hue_std = float(np.std(hsv[:,:,0]))
    sat_std = float(np.std(hsv[:,:,1]))
    if hue_std < 8 and sat_std < 15 and area_ratio > 0.25:
        logger.info(f"[BG_REJECT] hue_std={hue_std:.1f} sat_std={sat_std:.1f} → single color surface")
        return True

    logger.info(f"[BG_REJECT] ✅ NOT background — area={area_ratio:.2f} var={variance:.0f}")
    return False

# --- COMPLETE ITEM LISTS BY CATEGORY ---
PLASTIC_ITEMS = {
    "water bottle", "plastic bottle", "soda bottle",
    "juice bottle", "milk jug", "shampoo bottle",
    "conditioner bottle", "detergent bottle",
    "cleaning spray bottle", "ketchup bottle",
    "plastic bag", "grocery bag", "ziplock bag",
    "cling wrap", "bubble wrap", "plastic wrap",
    "food packaging", "snack wrapper",
    "plastic container", "takeaway container",
    "yogurt cup", "plastic cup", "plastic lid",
    "plastic tray", "styrofoam box", "foam cup",
    "bottle cap", "plastic cap", "straw",
    "plastic straw", "plastic spoon",
    "plastic fork", "plastic knife",
    "plastic chair", "plastic toy",
    "cd case", "dvd case",
}

YOLO_TO_PLASTIC = {
    "bottle", "cup", "bowl", "bag", "suitcase",
    "frisbee", "toothbrush", "cell phone",
    "remote", "clock", "umbrella",
}

GLASS_ITEMS = {
    "glass bottle", "wine bottle", "beer bottle",
    "juice glass bottle", "sauce bottle",
    "medicine bottle", "perfume bottle",
    "glass cup", "drinking glass", "wine glass",
    "champagne glass", "shot glass",
    "glass mug", "glass jar",
    "jam jar", "pickle jar", "honey jar",
    "pasta sauce jar", "glass container",
    "mason jar", "glass bowl",
    "glass vase", "glass candle holder",
    "glass plate", "glass dish",
    "broken glass", "glass panel",
    "mirror", "glass frame",
    "light bulb", "glass figurine",
}

YOLO_TO_GLASS = {
    "wine glass", "cup", "vase", "bowl",
    "bottle", "jar",
}

METAL_ITEMS = {
    "aluminum can", "soda can", "beer can",
    "energy drink can", "food tin", "canned food",
    "tin can", "paint can", "aerosol can",
    "spray can",
    "metal fork", "metal knife", "metal spoon",
    "metal chopsticks", "metal straw",
    "metal bowl", "metal pot", "metal pan",
    "metal lid", "bottle opener",
    "aluminum foil", "foil tray", "foil wrap",
    "metal packaging", "metal bottle cap",
    "metal screw", "metal nail", "metal wire",
    "metal pipe", "metal bracket",
    "metal hinge", "metal clip",
    "metal ruler", "scissors",
    "metal key", "metal coin",
    "staple", "paper clip",
    "metal cable", "metal tube",
}

YOLO_TO_METAL = {
    "can", "tin", "fork", "knife", "spoon",
    "scissors", "oven", "microwave",
    "toaster", "sink", "bowl",
}

PAPER_ITEMS = {
    "paper sheet", "white paper", "lined paper",
    "printed paper", "newspaper", "magazine",
    "notebook", "notepad",
    "book", "textbook", "comic book",
    "paperback", "hardcover book",
    "notebook", "diary", "journal",
    "cardboard box", "cardboard sheet",
    "pizza box", "cereal box", "shoe box",
    "delivery box", "packaging box",
    "egg carton", "cardboard tube",
    "envelope", "paper bag", "paper cup",
    "paper plate", "paper tray",
    "gift wrap", "tissue paper",
    "paper napkin", "tissue box",
    "receipt", "ticket", "brochure",
    "flyer", "poster", "sticky note",
    "calendar", "greeting card",
}

YOLO_TO_PAPER = {
    "book", "newspaper", "envelope",
    "cardboard", "box", "keyboard",
    "mouse", "remote", "laptop",
}

def get_material_hint(yolo_label: str) -> str:
    label = yolo_label.lower().strip()
    if label in YOLO_TO_GLASS:   return "glass"
    if label in YOLO_TO_METAL:   return "metal"
    if label in YOLO_TO_PAPER:   return "paper"
    if label in YOLO_TO_PLASTIC: return "plastic"
    return ""

def find_foreground_object(detections, frame_w, frame_h):
    """
    From all YOLO detections, return only the ones
    that are real objects (not background/surface).
    Scores each detection and filters background.
    """
    foreground = []

    for det in detections:
        if "box" not in det or "crop_rgb" not in det or "label" not in det:
            logger.warning(f"[FOCUS] Skipping invalid detection: missing keys {set(['box','crop_rgb','label']) - set(det.keys())}")
            continue
        box = det["box"]
        crop_rgb = det["crop_rgb"]
        label = det["label"]

        x1, y1, x2, y2 = box

        # Skip if background
        if is_background([x1,y1,x2,y2], frame_w, frame_h, crop_rgb):
            logger.info(f"[FOCUS] Rejected background: {label}")
            continue

        # Score how "foreground" this object is
        box_w = x2 - x1
        box_h = y2 - y1
        area_ratio = (box_w * box_h) / (frame_w * frame_h)

        # Center distance (objects near center = foreground)
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        center_dist = abs(cx - frame_w/2) / frame_w + \
                      abs(cy - frame_h/2) / frame_h

        # Edge sharpness (real objects are sharp)
        gray = cv2.cvtColor(
            cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR),
            cv2.COLOR_BGR2GRAY
        )
        edges      = cv2.Canny(gray, 50, 150)
        sharpness  = float(np.sum(edges > 0)) / edges.size

        # Foreground score:
        # smaller area = better (not background)
        # closer to center = better
        # sharper edges = better
        fg_score = (
            (1.0 - area_ratio) * 3.0 +   # small = foreground
            (1.0 - center_dist) * 2.0 +  # centered = foreground
            sharpness * 10.0              # sharp = real object
        )

        det["fg_score"] = fg_score
        logger.info(f"[FOCUS] {label}: fg_score={fg_score:.2f} area={area_ratio:.2f} sharp={sharpness:.3f}")
        foreground.append(det)

    # Sort by foreground score (highest first)
    foreground.sort(key=lambda d: d["fg_score"], reverse=True)
    return foreground

def should_run_mobilenet(crop_rgb: np.ndarray, 
                          yolo_label: str) -> tuple[bool, str]:
    """
    Master gate. Returns (True, "ok") ONLY if crop
    should be classified as waste material.
    """
    label = yolo_label.lower().strip()
    
    # Gate 1: Hard block list
    if label in HARD_BLOCK:
        return False, "hard_block"

    # Gate 2: Face and skin detector
    if is_face_or_skin(crop_rgb):
        return False, "face_detected"

    # Gate 3: Only whitelist OR unknown labels pass
    # (unknown = YOLO found something not in either list)
    if label in WASTE_ONLY or label not in HARD_BLOCK:
        return True, "ok"

    return False, "restricted"

# --- UNIFIED CONFIDENCE THRESHOLDS ---
THRESHOLDS = {
    "yolo_min":            0.15,
    "mobilenet_uncertain": 0.40,
    "mobilenet_confirmed": 0.55,
    "mobilenet_fallback":  0.45,
    "mobilenet_override":  0.55,
    "auto_save":           0.65,
    "robot_override":      5,
}

# --- PIPELINE CATEGORIZATION LISTS & MAPS ---
WASTE_CANDIDATES = {
    "bottle", "water bottle", "wine glass", "cup", "bowl", "vase", "jar", "can", "tin", "fork", "knife", "spoon", "scissors",
    "book", "newspaper", "paper", "cardboard", "box", "envelope",
    "bag", "container", "tube", "cylinder", "plastic bag", "straw", "plastic",
    "object", "item", "thing", "product", "package"
}

YOLO_TO_WASTE_HINTS = {
    # Glass hints
    "wine glass": "glass", "cup": "glass",
    "vase": "glass", "jar": "glass",
    "bowl": "glass", "glass": "glass",
    # Plastic hints
    "bottle": "plastic", "bag": "plastic",
    "container": "plastic", "suitcase": "plastic",
    "frisbee": "plastic", "toothbrush": "plastic",
    "cell phone": "plastic", "remote": "plastic",
    # Metal hints
    "can": "metal", "tin": "metal",
    "fork": "metal", "knife": "metal",
    "spoon": "metal", "scissors": "metal",
    "sink": "metal", "oven": "metal",
    # Paper hints
    "book": "paper", "newspaper": "paper",
    "envelope": "paper", "cardboard": "paper",
    "keyboard": "paper", "mouse": "paper",
    # Food = run metal check first
    "donut": "metal", "sandwich": "metal",
    "pizza": "paper",  # pizza box = cardboard
    # Misidentified items — still run material check
    "cat": "",   "dog": "",
    "chair": "", "table": "",
    "person": "", "laptop": "",
    "tv": "",    "clock": "",
}

HUMAN_LIST = {"person", "face", "human", "man", "woman", "child"}
NATURE_LIST = {"tree", "plant", "flower", "grass", "sky", "mountain", "river", "cloud"}
ANIMAL_LIST = {"animal", "dog", "cat", "bird"}
STRUCTURE_LIST = {"building", "house", "school", "road", "car", "bus", "truck", "chair", "table", "desk", "sofa", "bed", "tv", "laptop", "phone", "keyboard", "bicycle"}
FOOD_LIST = {"banana", "apple", "sandwich", "pizza", "donut", "cake", "hot dog", "broccoli", "carrot", "orange"}

EMOJI_MAP = {
  "person": "👤", "face": "😄",
  "tree": "🌳", "plant": "🌿", "flower": "🌸",
  "grass": "🌱", "sky": "☁️", "mountain": "⛰️",
  "dog": "🐕", "cat": "🐈", "bird": "🐦",
  "car": "🚗", "bus": "🚌", "bicycle": "🚲",
  "building": "🏢", "house": "🏠", "school": "🏫",
  "book": "📚", "chair": "🪑", "table": "🪵",
  "phone": "📱", "laptop": "💻", "tv": "📺",
  "banana": "🍌", "apple": "🍎", "pizza": "🍕",
  "sandwich": "🥪", "donut": "🍩", "cake": "🎂",
  "bottle": "🍾", "cup": "☕", "bowl": "🥣",
  "scissors": "✂️", "knife": "🔪",
  "sports ball": "⚽", "umbrella": "☂️",
  "backpack": "🎒", "handbag": "👜",
  "clothing": "👕", "tie": "👔",
  "plastic": "♻️", "glass": "♻️",
  "metal": "♻️", "paper": "♻️",
  "cardboard": "♻️", "can": "♻️",
  "default": "📦"
}

# --- BIN LABELS & COLOR MAPPINGS ---
BIN_LABELS = {
    "Plastic": "Blue",
    "Glass":   "Green",
    "Metal":   "Yellow",
    "Paper":   "Blue",
}

COLOR_NAMES = {
    "#ef4444": "Red",    "#dc2626": "Red",
    "#3b82f6": "Blue",   "#2563eb": "Blue",
    "#22c55e": "Green",  "#16a34a": "Green",
    "#eab308": "Yellow", "#ca8a04": "Yellow",
    "#f97316": "Orange", "#ea580c": "Orange",
    "#a855f7": "Purple", "#9333ea": "Purple",
    "#ec4899": "Pink",   "#db2777": "Pink",
    "#000000": "Black",  "#1e293b": "Black",
    "#ffffff": "White",  "#f8fafc": "White",
    "#6b7280": "Gray",   "#64748b": "Gray",
}

DEFAULT_BINS = {
    "Plastic": {"label": "Blue",   "hex": "#3b82f6"},
    "Glass":   {"label": "Green",  "hex": "#22c55e"},
    "Metal":   {"label": "Yellow", "hex": "#eab308"},
    "Paper":   {"label": "Blue",   "hex": "#f97316"},
}

# --- EXACT MESSAGES FROM SPEC ---
TREE_MSG = "🌳 Whoa, beautiful trees! I love nature \nbut my specialty is waste sorting. \nShow me a bottle, can, paper or glass \nand I'll tell you exactly how to recycle it!"
PLANT_MSG = "🌿 I see plants! Great for the environment. \nI'm a waste sorting robot though — \npoint me at some recyclable material!"
SKY_MSG = "☁️ All I see is sky! \nBring a waste item closer to the camera."
GRASS_MSG = "🌱 Nice green grass! I'm better at detecting \nwhat's ON the grass though. \nAny bottles or cans around?"

PERSON_MSG = "👋 Hey there! I'm SortIQ — a waste detection \nrobot, not a people detector! \nShow me Glass, Plastic, Paper or Metal \nand I'll do my magic! ♻️"
FACE_MSG = "😄 I see a face! Hello! I'm better at \nidentifying recyclable materials. \nPoint me at some waste to classify!"

BUILDING_MSG = "🏢 I can see a building! I work best \nwith close-up shots of waste items. \nTry holding a bottle or paper \nup to the camera!"
SCHOOL_MSG = "🏫 Looks like a school! Education is \nimportant — so is recycling! \nShow me some waste to sort ♻️"
ROAD_MSG = "🛣️ I see a road. I'm not a navigation app! \nShow me waste items to classify."

BANANA_MSG = "🍌 A banana! Organic waste — \ngoes in the compost bin, not recycling. \nI specialize in Glass, Plastic, \nPaper and Metal though!"
APPLE_MSG = "🍎 An apple! Compost bin for this one. \nShow me something made of \nPlastic, Glass, Metal or Paper!"
FOOD_GENERAL_MSG = "🍽️ I see food! I classify materials \nfor recycling, not food items. \nFood waste goes to organic/compost bins. \nShow me a bottle, can, or paper!"
PIZZA_MSG = "🍕 Pizza?! I'm hungry but I'm also a robot \nso I don't eat. I DO sort waste though — \nshow me the pizza box and I'll \nclassify that cardboard for you! 📦"

DOG_MSG = "🐕 Aww a dog! Not my area of expertise. \nI detect recyclable waste — \nGlass, Plastic, Paper, Metal. \nCan you show me some?"
CAT_MSG = "🐈 A cat! Cute but not recyclable 😄 \nShow me something made of \nGlass, Plastic, Paper or Metal!"
ANIMAL_GENERAL_MSG = "🐾 I see an animal! I'm a waste \nsorting robot, not a wildlife expert. \nShow me recyclable items!"

CAR_MSG = "🚗 That's a car! Big items like vehicles \nneed special recycling facilities. \nFor everyday recycling, show me \nbottles, cans, paper or glass!"
BICYCLE_MSG = "🚲 A bicycle! Metal frame — could be recycled \nat a scrap metal facility. \nFor everyday items, show me \ncans, bottles, or paper!"

BOOK_MSG = "📚 That's a book! Books are made of paper \nbut should be donated or reused \nbefore recycling. \nIf damaged beyond use → Paper recycling bin! \nShow me the book cover for a better scan."
CHAIR_MSG = "🪑 A chair! Furniture recycling depends on \nthe material. \nFor small everyday items, show me \nbottles, cans, or paper!"
TABLE_MSG = "🪵 I see a table! Large furniture goes to \nspecial collection points. \nShow me small recyclable items for \nmy best analysis!"
PHONE_MSG = "📱 A phone! E-waste — very important to \nrecycle properly at e-waste centers. \nDon't throw in regular bins!"
LAPTOP_MSG = "💻 A laptop! E-waste — take to an \nauthorized e-waste recycling center. \nNever in regular trash bins!"
SCISSORS_MSG = "✂️ Scissors! Metal item — \nwrap sharp edges before placing \nin the Metal recycling bin! ⚠️"

CLOTHING_MSG = "👕 Clothing! Donate if wearable, \ntextile recycling bin if worn out. \nNot in regular recycling bins!"
SPORTS_BALL_MSG = "⚽ A ball! Sporting goods go to \nspecial collection or donation. \nShow me bottles, cans or paper!"

PLASTIC_MSG = "♻️ Plastic detected! Place in the \nBLUE recycling bin. \nRemember to rinse and remove caps! \nEvery plastic saved = less ocean pollution 🌊"
GLASS_MSG = "♻️ Glass detected! Place in the \nGREEN glass recycling bin. \nHandle with care — \nglass is 100% recyclable forever! ✨"
METAL_MSG = "♻️ Metal detected! Place in the \nYELLOW metal recycling bin. \nAluminum cans can be recycled \ninfinitely — great job! 💪"
PAPER_MSG = "♻️ Paper detected! Place in the \nBLUE paper recycling bin. \nKeep it dry — wet paper \ncannot be recycled! 💧"

UNCERTAIN_MSG = "🤔 I think I see something recyclable \nbut I'm not 100% sure. \nTips: Move closer, better lighting, \nplace item on a flat surface!"
NOTHING_MSG = "📷 I can't see anything clearly! \nLet me help you: \n• Move the item closer to camera \n• Find better lighting \n• Place item on a white/flat surface \n• Hold the camera steady \nI'm looking for: Glass 🟢 Plastic 🔵 \nPaper 📄 Metal ⚡"

def get_nonwaste_color(label: str) -> str:
    l = label.lower()
    if l in {"person","face","man","woman","child",
             "boy","girl","baby","human"}:
        return "#ef4444"   # RED
    if l in {"cat","dog","bird","horse","cow",
             "elephant","bear","sheep"}:
        return "#ec4899"   # PINK
    if l in {"tree","plant","flower","grass",
             "sky","mountain"}:
        return "#3b82f6"   # BLUE
    if l in {"bed","couch","sofa","toilet",
             "chair","dining table"}:
        return "#a855f7"   # PURPLE
    if l in {"car","truck","bus","motorcycle",
             "airplane","train"}:
        return "#6b7280"   # GRAY
    return "#6b7280"       # GRAY default

def get_display_label(label: str) -> str:
    EMOJI = {
        "person":"👤 Person",   "face":"😊 Face",
        "cat":"🐱 Cat",         "dog":"🐕 Dog",
        "bird":"🐦 Bird",       "tree":"🌳 Tree",
        "plant":"🌿 Plant",     "flower":"🌸 Flower",
        "bed":"🛏️ Bed",         "couch":"🛋️ Sofa",
        "chair":"🪑 Chair",
        "car":"🚗 Car",         "truck":"🚚 Truck",
    }
    return EMOJI.get(label.lower(), f"📦 {label.capitalize()}")

def get_message_for_yolo_label(label: str) -> str:
    l = label.lower()
    if l == "tree": return TREE_MSG
    if l in ("plant", "flower"): return PLANT_MSG
    if l == "sky" or l == "cloud": return SKY_MSG
    if l == "grass": return GRASS_MSG
    if l in HUMAN_LIST: return FACE_MSG if l == "face" else PERSON_MSG
    if l in FOOD_LIST:
        if l == "banana": return BANANA_MSG
        if l == "apple": return APPLE_MSG
        if l == "pizza": return PIZZA_MSG
        return FOOD_GENERAL_MSG
    if l in ANIMAL_LIST:
        if l == "dog": return DOG_MSG
        if l == "cat": return CAT_MSG
        return ANIMAL_GENERAL_MSG
    if l == "building" or l == "house": return BUILDING_MSG
    if l == "school": return SCHOOL_MSG
    if l == "road": return ROAD_MSG
    if l in ("car", "bus", "truck"): return CAR_MSG
    if l == "bicycle": return BICYCLE_MSG
    if "book" in l: return BOOK_MSG
    if l == "chair": return CHAIR_MSG
    if l in ("table", "desk"): return TABLE_MSG
    if l == "phone" or l == "cell phone": return PHONE_MSG
    if l == "laptop" or l == "tv": return LAPTOP_MSG
    if "scissors" in l: return SCISSORS_MSG
    if "clothing" in l or "tie" in l: return CLOTHING_MSG
    if "sports ball" in l: return SPORTS_BALL_MSG
    return f"I see a {label}! Try scanning a recyclable item."

def robot_glass(crop_rgb, yolo_hint=""):
    score = 0
    crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    sat = float(np.mean(hsv[:,:,1]))
    bri = float(np.mean(hsv[:,:,2]))
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    var = float(np.var(gray))
    edges = cv2.Canny(gray, 50, 150)
    edge_d = float(np.sum(edges>0)) / edges.size
    h, w = crop_rgb.shape[:2]
    aspect = h / max(w, 1)
    
    r_mean = float(np.mean(crop_rgb[:,:,0]))
    g_mean = float(np.mean(crop_rgb[:,:,1]))
    b_mean = float(np.mean(crop_rgb[:,:,2]))
    channel_diff = max(abs(r_mean-g_mean), 
                       abs(g_mean-b_mean), 
                       abs(r_mean-b_mean))
    
    if sat < 30:     score += 4
    elif sat < 50:   score += 2
    if bri > 190:    score += 3
    elif bri > 160:  score += 1
    if var > 3000:   score += 3
    elif var > 2000: score += 2
    if edge_d > 0.08: score += 2
    if aspect > 1.5:  score += 2
    if channel_diff < 15: score += 2  # transparent = similar channels
    
    GLASS_HINTS = {"wine glass","cup","vase","jar","bowl",
                   "glass","bottle"}
    if yolo_hint.lower() in GLASS_HINTS:
        score += 5
    
    logger.info(f"[ROBOT_GLASS] score={score} sat={sat:.0f} bri={bri:.0f} var={var:.0f}")
    return score

def robot_plastic(crop_rgb, yolo_hint=""):
    score = 0
    crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    sat = float(np.mean(hsv[:,:,1]))
    bri = float(np.mean(hsv[:,:,2]))
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    var = float(np.var(gray))
    edges = cv2.Canny(gray, 50, 150)
    edge_d = float(np.sum(edges>0)) / edges.size
    h, w = crop_rgb.shape[:2]
    aspect = h / max(w, 1)
    
    r_mean = float(np.mean(crop_rgb[:,:,0]))
    g_mean = float(np.mean(crop_rgb[:,:,1]))
    b_mean = float(np.mean(crop_rgb[:,:,2]))
    channel_diff = max(abs(r_mean-g_mean),
                       abs(g_mean-b_mean),
                       abs(r_mean-b_mean))
    
    if 30 < sat < 80:   score += 3
    elif sat >= 80:     score += 2
    if 600 < var < 2500: score += 2
    if 0.03 < edge_d < 0.09: score += 2
    if bri > 100 and sat > 25: score += 2  # colored + bright
    if aspect > 1.5 and sat > 30: score += 3  # plastic bottle
    if channel_diff > 20: score += 2  # colored = not transparent
    
    PLASTIC_HINTS = {"bottle","bag","container","plastic",
                     "cup","suitcase","frisbee","toothbrush",
                     "cell phone","remote","clock"}
    if yolo_hint.lower() in PLASTIC_HINTS:
        score += 4
    
    logger.info(f"[ROBOT_PLASTIC] score={score} sat={sat:.0f}")
    return score

def robot_metal(crop_rgb, yolo_hint=""):
    score = 0
    crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    sat = float(np.mean(hsv[:,:,1]))
    bri = float(np.mean(hsv[:,:,2]))
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    var = float(np.var(gray))
    bri_var = float(np.var(hsv[:,:,2]))
    edges = cv2.Canny(gray, 50, 150)
    edge_d = float(np.sum(edges>0)) / edges.size
    
    r_mean = float(np.mean(crop_rgb[:,:,0]))
    g_mean = float(np.mean(crop_rgb[:,:,1]))
    b_mean = float(np.mean(crop_rgb[:,:,2]))
    gray_dominance = 1.0 - (max(abs(r_mean-g_mean),
                                abs(g_mean-b_mean)) / 255.0)
    
    if sat < 30:       score += 3
    elif sat < 50:     score += 2
    if bri > 150:      score += 2
    if var > 3000:     score += 3
    elif var > 2000:   score += 2
    if bri_var > 1500: score += 3  # specular highlights
    if edge_d > 0.07:  score += 2
    if gray_dominance > 0.85: score += 2  # gray = metallic
    
    METAL_HINTS = {"can","tin","fork","knife","spoon",
                   "scissors","bowl","cup","oven","microwave",
                   "refrigerator","toaster","sink"}
    FOOD_HINTS  = {"donut","sandwich","pizza","hot dog",
                   "cake","banana","apple","orange","broccoli"}
    
    if yolo_hint.lower() in METAL_HINTS:
        score += 5
    if yolo_hint.lower() in FOOD_HINTS:
        # YOLO said food but image is metallic
        if sat < 50 and var > 1500:
            score += 4
            logger.info(f"[ROBOT_METAL] Food override: {yolo_hint} looks metallic")
    
    logger.info(f"[ROBOT_METAL] score={score} sat={sat:.0f} var={var:.0f} bri_var={bri_var:.0f}")
    return score

def robot_paper(crop_rgb, yolo_hint=""):
    score = 0
    crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    hue = float(np.mean(hsv[:,:,0]))
    sat = float(np.mean(hsv[:,:,1]))
    bri = float(np.mean(hsv[:,:,2]))
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    var = float(np.var(gray))
    bri_var = float(np.var(hsv[:,:,2]))
    edges = cv2.Canny(gray, 50, 150)
    edge_d = float(np.sum(edges>0)) / edges.size
    h, w = crop_rgb.shape[:2]
    aspect = w / max(h, 1)  # wider than tall = flat sheet
    
    if var < 500:       score += 3
    elif var < 1000:    score += 2
    elif var < 2000:    score += 1
    if 10 < hue < 40:   score += 3  # warm = paper/cardboard
    elif hue < 10:      score += 1  # white paper
    if sat < 40:        score += 2  # low saturation = white/beige
    if 100 < bri < 220: score += 2  # medium brightness = not shiny
    if edge_d < 0.04:   score += 2  # soft edges
    if aspect > 1.2:    score += 2  # wider than tall
    if bri_var < 500:   score += 2  # no specular = matte
    
    PAPER_HINTS = {"book","newspaper","envelope","cardboard",
                   "box","remote","keyboard","mouse",
                   "laptop","magazine"}
    if yolo_hint.lower() in PAPER_HINTS:
        score += 5
    
    logger.info(f"[ROBOT_PAPER] score={score} hue={hue:.0f} sat={sat:.0f} var={var:.0f}")
    return score

def robot_glass_detector(crop_rgb):
    """
    Dedicated glass detection using 6 techniques.
    Returns (is_glass: bool, glass_confidence: float)
    """
    try:
        crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
        hsv      = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        gray     = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

        sat = float(np.mean(hsv[:,:,1]))
        bri = float(np.mean(hsv[:,:,2]))
        var = float(np.var(gray))

        # Technique 1: Transparency check
        # Glass = RGB channels are very similar
        # (transparent = sees through to same color)
        r = float(np.mean(crop_rgb[:,:,0]))
        g = float(np.mean(crop_rgb[:,:,1]))
        b = float(np.mean(crop_rgb[:,:,2]))
        channel_diff = max(abs(r-g), abs(g-b), abs(r-b))
        is_transparent = channel_diff < 20
        logger.info(f"[GLASS] channel_diff={channel_diff:.1f} transparent={is_transparent}")

        # Technique 2: Specular highlight detection
        # Glass has bright white spots (light reflection)
        _, bright_mask = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
        bright_ratio   = float(np.sum(bright_mask > 0)) / bright_mask.size
        has_highlights = bright_ratio > 0.02 and bright_ratio < 0.30
        logger.info(f"[GLASS] bright_ratio={bright_ratio:.3f} highlights={has_highlights}")

        # Technique 3: Edge sharpness pattern
        # Glass has very sharp hard edges
        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(np.sum(edges > 0)) / edges.size
        # Dilate edges to find edge thickness
        kernel      = np.ones((3,3), np.uint8)
        thick_edges = cv2.dilate(edges, kernel, iterations=1)
        thick_ratio = float(np.sum(thick_edges > 0)) / thick_edges.size
        edge_sharpness = edge_density / (thick_ratio + 0.001)
        has_sharp_edges = edge_sharpness > 0.5
        logger.info(f"[GLASS] edge_density={edge_density:.3f} sharpness={edge_sharpness:.2f}")

        # Technique 4: Refraction pattern
        # Glass creates gradient transitions at edges
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(sobelx**2 + sobely**2)
        gradient_var = float(np.var(gradient_mag))
        has_refraction = gradient_var > 5000
        logger.info(f"[GLASS] gradient_var={gradient_var:.0f} refraction={has_refraction}")

        # Technique 5: Low saturation check
        # Glass is not colored (mostly)
        is_low_sat = sat < 50
        logger.info(f"[GLASS] sat={sat:.1f} low_sat={is_low_sat}")

        # Technique 6: Brightness check
        # Glass reflects light = bright regions
        is_bright = bri > 140
        logger.info(f"[GLASS] bri={bri:.1f} is_bright={is_bright}")

        # Score all techniques
        score = sum([
            is_transparent  * 3,   # strongest signal
            has_highlights  * 2,   # specular = glass
            has_sharp_edges * 2,   # hard edges
            has_refraction  * 2,   # gradient variation
            is_low_sat      * 1,   # not colored
            is_bright       * 1,   # reflective
        ])

        max_score    = 11
        glass_conf   = score / max_score
        is_glass     = score >= 5   # need at least 5/11

        logger.info(f"[GLASS] score={score}/{max_score} conf={glass_conf:.2f} is_glass={is_glass}")
        return is_glass, glass_conf

    except Exception as e:
        logger.warning(f"[GLASS] detector failed: {e}")
        return False, 0.0

def separate_glass_from_metal(crop_rgb):
    """
    When both glass and metal are possible,
    use these signals to decide which one.
    Returns "glass", "metal", or "uncertain"
    """
    crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
    hsv      = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    gray     = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

    sat       = float(np.mean(hsv[:,:,1]))
    bri       = float(np.mean(hsv[:,:,2]))
    bri_var   = float(np.var(hsv[:,:,2]))
    gray_var  = float(np.var(gray))

    # RGB channel similarity
    r = float(np.mean(crop_rgb[:,:,0]))
    g = float(np.mean(crop_rgb[:,:,1]))
    b = float(np.mean(crop_rgb[:,:,2]))
    channel_diff = max(abs(r-g), abs(g-b), abs(r-b))

    # Bright spots ratio
    _, bright = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
    bright_ratio = float(np.sum(bright > 0)) / bright.size

    glass_score = 0
    metal_score = 0

    # Glass signals
    if channel_diff < 20:   glass_score += 4  # transparent
    if bright_ratio > 0.03 \
    and bright_ratio < 0.25: glass_score += 3  # controlled highlights
    if sat < 30:             glass_score += 2  # very low sat
    if bri_var < 2000:       glass_score += 2  # smooth brightness

    # Metal signals
    if channel_diff > 25:    metal_score += 2  # slight color tint
    if bri_var > 2500:       metal_score += 3  # high brightness variation
    if bright_ratio > 0.25:  metal_score += 3  # many bright spots
    if gray_var > 3000:      metal_score += 2  # rough texture
    if sat > 30 and sat < 60: metal_score += 2  # slight metallic color

    logger.info(
        f"[GLASS_vs_METAL] glass={glass_score} metal={metal_score} "
        f"ch_diff={channel_diff:.1f} bright={bright_ratio:.3f} "
        f"bri_var={bri_var:.0f}"
    )

    if glass_score > metal_score + 2:
        return "glass", glass_score
    elif metal_score > glass_score + 2:
        return "metal", metal_score
    else:
        return "uncertain", max(glass_score, metal_score)

def run_material_robots(crop_rgb, yolo_hint=""):
    """
    All 4 robots run simultaneously on same crop.
    Highest score wins. Score is combined with
    MobileNetV2 probability for final decision.
    """
    scores = {
        "Glass":   robot_glass(crop_rgb, yolo_hint),
        "Plastic": robot_plastic(crop_rgb, yolo_hint),
        "Metal":   robot_metal(crop_rgb, yolo_hint),
        "Paper":   robot_paper(crop_rgb, yolo_hint),
    }
    
    logger.info(f"[MATERIAL_ROBOTS] scores={scores}")
    
    # Winner by score
    robot_winner = max(scores, key=lambda k: scores[k])
    robot_score  = scores[robot_winner]
    
    # Must win by clear margin (at least 3 points ahead)
    scores_sorted = sorted(scores.values(), reverse=True)
    margin = scores_sorted[0] - scores_sorted[1]
    
    logger.info(f"[MATERIAL_ROBOTS] winner={robot_winner} margin={margin}")
    
    return robot_winner, robot_score, margin, scores

class SortIQModel:
    _instance = None
    model = None
    yolo_model = None
    classes: dict = {}
    version: str = "v1.0"
    threshold: float = 0.30
    bin_mapping: dict = {
        "Plastic": {"bin": "Blue", "colorHex": "#3b82f6"},
        "Paper": {"bin": "Green", "colorHex": "#22c55e"},
        "Metal": {"bin": "Yellow", "colorHex": "#eab308"},
        "Glass": {"bin": "Red", "colorHex": "#ef4444"}
    }
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SortIQModel, cls).__new__(cls)
            cls._instance.model = None
            cls._instance.yolo_model = None
            cls._instance.classes = {}
            cls._instance.version = os.getenv("MODEL_VERSION", "v1.0")
            cls._instance.threshold = float(os.getenv("CONFIDENCE_THRESHOLD", 0.30))
            cls._instance.bin_mapping = cls.bin_mapping
        return cls._instance
        
    def load(self):
        import gc
        import os
        
        # Free memory before loading models
        gc.collect()
        
        # Set TF to use minimal memory
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        os.environ["TF_USE_LEGACY_KERAS"]   = "1"
        
        # Limit TensorFlow memory growth
        import tensorflow as tf
        gpus = tf.config.experimental.list_physical_devices("GPU")
        if gpus:
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except:
                    pass
        
        # Configure TF to use minimal CPU memory
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)

        # 1. Load YOLOv8n (Lightweight)
        try:
            logger.info("Loading YOLOv8n model...")
            self.yolo_model = YOLO("yolov8n.pt")
            gc.collect()  # free memory after YOLO loads
            logger.info("YOLOv8n loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load YOLO: {e}")
            self.yolo_model = None

        # 2. Load MobileNetV2 (Waste Classification)
        model_path = os.getenv("MODEL_PATH", "./model/sortiq_model.h5")
        classes_path = os.path.join(os.path.dirname(model_path), "classes.json")
        
        os.environ['TF_USE_LEGACY_KERAS'] = '1'
        logger.info(f"Loading MobileNetV2 with tf_keras from {model_path}...")
        try:
            import tf_keras  # pyre-ignore
            self.model = tf_keras.models.load_model(
                model_path, 
                compile=False,
                custom_objects={'BatchNormalization': KerasBatchNormalization}
            )
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            raise RuntimeError(f"Failed to load model from {model_path}: {e}")
        
        # Validate with warm-up
        if self.model is not None:
            logger.info("Validating model with warm-up...")
            try:
                dummy_input = np.zeros((1, 224, 224, 3), dtype=np.float32)
                _ = self.model.predict(dummy_input, verbose=0)
                logger.info("Model warm-up successful.")
            except Exception as e:
                logger.warning(f"Warm-up failed (non-critical): {e}")
        else:
            logger.warning("Main model is None - skipping warm-up")
        
        gc.collect()  # free memory after both models load
            
        # Load class mapping
        if self.model is not None:
            if os.path.exists(classes_path):
                try:
                    with open(classes_path, "r") as f:
                        raw_classes = json.load(f)
                    
                    # Normalize keys to both int and str for bulletproof lookup
                    self.classes = {}
                    for k, v in raw_classes.items():
                        self.classes[int(k)] = v
                        self.classes[str(k)] = v
                        
                    logger.info("Class mapping loaded successfully.")
                except Exception as e:
                    logger.error(f"Failed to parse classes.json: {e}")
                    raise RuntimeError(f"Classes mapping required at {classes_path}")
            else:
                logger.warning(f"Class mapping not found at {classes_path}. Using fallback.")
                self.classes = {
                    0: "Glass", "0": "Glass",
                    1: "Metal", "1": "Metal",
                    2: "Paper", "2": "Paper",
                    3: "Plastic", "3": "Plastic"
                }
                
            # Warm-up disabled to prevent startup crash
            # logger.info("Performing warm-up inference...")
            # dummy_input = np.zeros((1, 224, 224, 3), dtype=np.float32)
            # keras_model = self.model
            # if keras_model is not None:
            #     keras_model.predict(dummy_input)
            # logger.info("Warm-up complete.")
            
    def get_crops(self, input_data: np.ndarray) -> np.ndarray:
        """
        Creates 5 augmented views for Test-Time Augmentation using numpy/OpenCV.
        input_data is shape (1, 224, 224, 3)
        Returns shape (5, 224, 224, 3)
        """
        img = input_data[0].copy()  # (224, 224, 3)
        crops = []
        
        # 1. Original
        crops.append(img)
        
        # 2. Flipped Horizontal (numpy)
        crops.append(np.fliplr(img))
        
        # 3. Brightness slightly up (clip to float [0,1])
        crops.append(np.clip(img + 0.05, 0.0, 1.0))
        
        # 4. Brightness slightly down
        crops.append(np.clip(img - 0.05, 0.0, 1.0))
        
        # 5. Mild center crop then resize back to 224x224
        h, w = img.shape[:2]
        margin = int(h * 0.05)
        center_crop = img[margin:h-margin, margin:w-margin]
        # Convert to uint8 for cv2 resize then back
        center_uint8 = (center_crop * 255).astype(np.uint8)
        zoomed_uint8 = cv2.resize(center_uint8, (224, 224), interpolation=cv2.INTER_LINEAR)
        crops.append(zoomed_uint8.astype(np.float32) / 255.0)
        
        return np.array(crops)

    def is_likely_metal(self, crop_rgb: np.ndarray) -> bool:
        """
        Pre-check: detect metallic texture regardless of YOLO label.
        Metal signature: low saturation + high brightness + high gray variance.
        """
        try:
            crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
            hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
            saturation = float(np.mean(hsv[:, :, 1]))
            brightness  = float(np.mean(hsv[:, :, 2]))
            gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
            variance = float(np.var(gray))
            result = saturation < 50 and brightness > 120 and variance > 1500
            if result:
                logger.info(f"is_likely_metal triggered — sat:{saturation:.1f} bright:{brightness:.1f} var:{variance:.0f}")
            return result
        except Exception as e:
            logger.warning(f"is_likely_metal() check failed: {e}")
            return False
            
    def is_likely_paper(self, crop_rgb: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> bool:
        """Paper: high brightness, low sat, warm hue, and flat aspect ratio rules."""
        try:
            crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
            hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
            sat = float(np.mean(hsv[:, :, 1]))
            bri = float(np.mean(hsv[:, :, 2]))
            hue = float(np.mean(hsv[:, :, 0]))
            
            gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
            variance = float(np.var(gray))
            
            w = max(1, x2 - x1)
            h = max(1, y2 - y1)
            aspect = w / h
            
            # Paper is generally matte (low variance), warm/neutral hue
            if sat < 40 and bri > 150 and (10 < hue < 50 or hue > 160):
                 if variance < 1000:
                      logger.info(f"is_likely_paper triggered -> sat:{sat:.1f} bri:{bri:.1f} var:{variance:.0f}")
                      return True
                      
            # Cardboard is often flat and warm
            if aspect > 1.2 and variance < 800 and (15 < hue < 40):
                 logger.info(f"is_likely_paper (cardboard) triggered -> aspect:{aspect:.2f} hue:{hue:.1f}")
                 return True
                 
            return False
        except Exception:
            return False

    def is_likely_glass(self, crop_rgb: np.ndarray) -> bool:
        """Glass: extremely low sat, extremely high brightness, high variance (refractions)"""
        try:
            crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
            hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
            sat = float(np.mean(hsv[:, :, 1]))
            bri = float(np.mean(hsv[:, :, 2]))
            
            gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
            variance = float(np.var(gray))
            
            # Very specular transparent behavior
            if sat < 25 and bri > 170 and variance > 2000:
                logger.info(f"is_likely_glass triggered -> sat:{sat:.1f} bri:{bri:.1f} var:{variance:.0f}")
                return True
            return False
        except Exception:
            return False
            
    def is_likely_bottle(self, x1: int, y1: int, x2: int, y2: int) -> bool:
        """Bottles are typically tall and thin (aspect ratio < 0.8)"""
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        aspect = w / h
        return aspect < 0.85

    def glass_vs_plastic_check(self, crop_rgb):
        try:
            crop_bgr   = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
            hsv        = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
            sat        = float(np.mean(hsv[:,:,1]))
            bri        = float(np.mean(hsv[:,:,2]))
            gray       = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
            var        = float(np.var(gray))
            edges      = cv2.Canny(gray, 50, 150)
            edge_d     = float(np.sum(edges>0)) / edges.size
            
            gs, ps = 0, 0
            
            # Saturation (glass = low = transparent)
            if sat < 20:  gs += 4
            elif sat < 35: gs += 3
            elif sat < 50: gs += 1
            if sat > 45:  ps += 2
            if sat > 60:  ps += 2
            
            # Brightness (glass = high = light passes through)
            if bri > 200: gs += 3
            elif bri > 170: gs += 2
            elif bri > 140: gs += 1
            
            # Variance (glass = high specular highlights)
            if var > 3000:  gs += 3
            elif var > 2000: gs += 2
            elif var > 1200: gs += 1
            if var < 800:   ps += 2
            
            # Edges (glass = sharper edges)
            if edge_d > 0.08: gs += 2
            if edge_d < 0.03: ps += 1
            
            logger.info(f"glass_check: sat={sat:.0f} bri={bri:.0f} var={var:.0f} edge={edge_d:.3f} → G={gs} P={ps}")
            return gs, ps
        except Exception as e:
            logger.warning(f"glass_check failed: {e}")
            return 0, 0

    def analyze_waste_crop_v2(self, crop_array, crop_rgb, yolo_hint=""):
        """
        ANALYST robot: MobileNetV2 TTA ensemble
        """
        # Guard: if model not loaded, return equal probabilities (no crash)
        if self.model is None:
            logger.warning("robot_analyst: MobileNetV2 not loaded, attempting reload...")
            self.load()
        if self.model is None:
            logger.error("robot_analyst: MobileNetV2 still not available — returning neutral probabilities")
            nclasses = len([k for k in self.classes if isinstance(k, int)]) or 4
            return np.ones(nclasses) / nclasses, {v.lower(): int(k) for k,v in self.classes.items() if isinstance(k, int)}

        crops_batch = self.get_crops(crop_array)
        predictions = self.model.predict(crops_batch, verbose=0)
        probs = np.mean(predictions, axis=0).copy()
        
        rev = {v.lower(): int(k) for k,v in self.classes.items()
               if isinstance(k, int)}
        
        glass_idx   = rev.get("glass")
        metal_idx   = rev.get("metal")
        plastic_idx = rev.get("plastic")
        paper_idx   = rev.get("paper")

        # ── GLASS ROBOT ─────────────────────────────
        is_glass, glass_conf = robot_glass_detector(crop_rgb)
        if is_glass and glass_idx is not None:
            logger.info(f"[ANALYST] Glass robot confirmed → boosting glass {glass_conf:.2f}")
            boost = 1.5 + glass_conf * 3.0   # up to 4.5× boost
            probs[glass_idx]   *= boost
            if metal_idx is not None: probs[metal_idx]   *= 0.3    # push metal down hard
            if plastic_idx is not None: probs[plastic_idx] *= 0.5
            probs = probs / probs.sum()

        # ── GLASS vs METAL SEPARATOR ─────────────────
        if glass_idx is not None and metal_idx is not None:
            winner, score = separate_glass_from_metal(crop_rgb)
            if winner == "glass":
                logger.info(f"[SEPARATOR] Glass wins over metal")
                probs[glass_idx] *= 2.5
                probs[metal_idx] *= 0.4
                probs = probs / probs.sum()
            elif winner == "metal":
                logger.info(f"[SEPARATOR] Metal wins over glass")
                probs[metal_idx] *= 2.5
                probs[glass_idx] *= 0.4
                probs = probs / probs.sum()

        try:
            logger.info(f"[ANALYST] boosted MobileNetV2 probs: {dict(zip(['Glass','Metal','Paper','Plastic'], probs.tolist()))}")
        except:
            pass
            
        return probs, rev

    def robot_verifier_v2(self, probs, rev, crop_rgb, yolo_hint=""):
        """
        VERIFIER robot: combines MobileNetV2 + material robots
        Final material decision made here.
        """
        # Run all 4 material robots
        robot_winner, robot_score, margin, all_scores = \
            run_material_robots(crop_rgb, yolo_hint)
        
        # Get MobileNetV2 winner
        mob_winner_idx  = int(np.argmax(probs))
        mob_winner      = self.classes.get(mob_winner_idx, "Unknown")
        mob_confidence  = float(np.max(probs))
        
        logger.info(f"[VERIFIER] MobileNetV2 says: {mob_winner} ({mob_confidence*100:.1f}%)")
        logger.info(f"[VERIFIER] Robots say: {robot_winner} (margin={margin})")
        
        # DECISION RULES:
        
        # Rule 1: Both agree → high confidence result
        if mob_winner.lower() == robot_winner.lower():
            final_class = mob_winner
            final_conf  = min(mob_confidence * 1.2, 0.99)
            logger.info(f"[VERIFIER] ✅ AGREEMENT → {final_class} {final_conf*100:.1f}%")
            return final_class, final_conf
        
        # Rule 2: Robots have strong signal (margin >= 5)
        # AND MobileNetV2 confidence is not overwhelming (< 0.80)
        # → Trust the robots over MobileNetV2
        if margin >= 5 and mob_confidence < 0.80:
            # Apply robot winner boost to probs
            winner_idx = rev.get(robot_winner.lower())
            if winner_idx is not None:
                probs[winner_idx] *= 3.0
                probs = probs / probs.sum()
            final_class = robot_winner
            final_conf  = float(np.max(probs))
            logger.info(f"[VERIFIER] 🔄 ROBOT OVERRIDE → {final_class} {final_conf*100:.1f}%")
            return final_class, final_conf
        
        # Rule 3: MobileNetV2 is very confident (>= 0.80)
        # → Trust MobileNetV2 even if robots disagree
        if mob_confidence >= 0.80:
            logger.info(f"[VERIFIER] 💪 MOBILENET CONFIDENT → {mob_winner}")
            return mob_winner, mob_confidence
        
        # Rule 4: Neither is confident → apply partial robot boost
        if margin >= 3:
            winner_idx = rev.get(robot_winner.lower())
            if winner_idx is not None:
                probs[winner_idx] *= 1.8
                probs = probs / probs.sum()
        
        final_idx = int(np.argmax(probs))
        final_class = self.classes.get(final_idx) or self.classes.get(str(final_idx)) or "Unknown"
        final_conf  = float(np.max(probs))
        
        if final_class.lower() not in VALID_WASTE:
            logger.info(f"[VERIFIER] Invalid category '{final_class}' → Unknown")
            return "Unknown", 0.0

        logger.info(f"[VERIFIER] 🤔 PARTIAL → {final_class} {final_conf*100:.1f}%")
        return final_class, final_conf


    def get_object_location(self, x1, y1, x2, y2, frame_w, frame_h):
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        h_pos = "Left" if cx < frame_w/3 else ("Right" if cx > 2*frame_w/3 else "Center")
        v_pos = "Top" if cy < frame_h/3 else ("Bottom" if cy > 2*frame_h/3 else "Middle")
        return f"{v_pos}-{h_pos}"

    def get_mobilenet_crop_detections(self, crop_pil, img_pil, x1, y1, x2, y2, yolo_hint=None, metal_boost=False, paper_boost=False, glass_boost=False, bottle_boost=False):
        if crop_pil.mode != "RGB":
            crop_pil = crop_pil.convert("RGB")
        crop_pil_resized = crop_pil.resize((224, 224))
        crop_array = np.array(crop_pil_resized, dtype=np.float32) / 255.0
        crop_array = np.expand_dims(crop_array, axis=0)
        original_crop_rgb = np.array(crop_pil)
        
        probs, rev = self.analyze_waste_crop_v2(
             crop_array, original_crop_rgb, 
             yolo_hint=yolo_hint
        )
        pred_class, mobilenet_conf = self.robot_verifier_v2(
             probs, rev, original_crop_rgb, yolo_hint
        )
        location = self.get_object_location(x1, y1, x2, y2, img_pil.width, img_pil.height)
        return pred_class, mobilenet_conf, location

    def format_box_label(self, label: str, conf: float) -> str:
        emoji = EMOJI_MAP.get(label.lower(), EMOJI_MAP["default"])
        return f"{emoji} {label.capitalize()} {int(conf * 100)}%"

    def format_waste_message(self, waste_class: str,
                              bin_label: Optional[str] = None,
                              bin_color_hex: Optional[str] = None) -> str:
        """
        ROBOT NARRATOR: Generate dynamic waste message without hardcoded colors.
        """
        w = waste_class.lower()
        if w == "plastic":
            return (
                "♻️ Plastic detected! Place in the plastic "
                "recycling bin. Remember to rinse and remove "
                "caps! Every plastic saved = less ocean "
                "pollution 🌊"
            )
        if w == "glass":
            return (
                "♻️ Glass detected! Place in the glass "
                "recycling bin. Handle with care — "
                "glass is 100% recyclable forever! ✨"
            )
        if w == "metal":
            return (
                "♻️ Metal detected! Place in the metal "
                "recycling bin. Aluminum cans can be "
                "recycled infinitely — great job! 💪"
            )
        if w == "paper":
            return (
                "♻️ Paper detected! Place in the paper "
                "recycling bin. Keep it dry — wet paper "
                "cannot be recycled! 💧"
            )
        return f"♻️ {waste_class} detected! Place in the recycling bin."

    def robot_scout(self, img_pil):
        """ROBOT SCOUT: Ask YOLO to find bounding boxes"""
        yolo_net = self.yolo_model
        if yolo_net is None:
            raise RuntimeError("YOLO model not loaded.")
        return yolo_net(img_pil, verbose=False)

    def predict_scene(self, img_pil: Image.Image, color_overrides: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
        """
        MASTER ORCHESTRATOR: 7-Robot Architecture
        1. Scout (YOLO) points 👉
        2. Analyst (MobileNet) looks 👁️
        3. Verifier corrects 🧠
        4. Narrator speaks 💬
        """
        try:
            if color_overrides is None:
                 color_overrides = {}
                 
            detections: List[Dict[str, Any]] = []
            frame_w, frame_h = img_pil.width, img_pil.height
            
            # ── SCOUT ROBOT ──────────────────────────────
            yolo_results = self.robot_scout(img_pil)
            raw_boxes = []
            
            if yolo_results and len(yolo_results[0].boxes) > 0:
                for box in yolo_results[0].boxes:
                    conf = float(box.conf[0])
                    if conf < THRESHOLDS["yolo_min"]:
                        continue
                    cls_id = int(box.cls[0])
                    label  = yolo_results[0].names[cls_id].lower()  # pyre-ignore
                    x1,y1,x2,y2 = map(int, box.xyxy[0])
                    # Ensure box is within frame boundaries
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame_w, x2), min(frame_h, y2)
                    
                    crop_arr = np.array(img_pil)[y1:y2, x1:x2]
                    if crop_arr.size == 0: continue

                    raw_boxes.append({
                        "label": label,
                        "conf":  conf,
                        "box":   [x1,y1,x2,y2],
                        "crop_rgb": crop_arr
                    })
                logger.info(f"[SCOUT] found {len(raw_boxes)} objects: {[b['label'] for b in raw_boxes]}")
            
            # ── PROCESS EACH BOX ─────────────────────────
            if raw_boxes:
                # ── FOCUS FILTER: remove background ──────
                foreground = find_foreground_object(raw_boxes, frame_w, frame_h)

                if not foreground:
                    logger.info("[FOCUS] All detections were background — trying full image")
                else:
                    for raw in foreground:
                        if "label" not in raw or "box" not in raw or "conf" not in raw or "crop_rgb" not in raw:
                            logger.warning(f"[ORCHESTRATOR] Skipping invalid foreground: missing keys {set(['label','box','conf','crop_rgb']) - set(raw.keys())}")
                            continue
                        label = raw["label"]
                        box = raw["box"]
                        yolo_conf = raw["conf"]
                        crop_rgb = raw["crop_rgb"]

                        x1, y1, x2, y2 = box
                        
                        yolo_hint = get_material_hint(label)
                        
                        # ── GATE CHECK ──────────────────────────────
                        logger.info(f"[GATE] checking label='{label}'")
                        run_it, reason = should_run_mobilenet(crop_rgb, label)
                        logger.info(f"[GATE] result={run_it} reason={reason}")

                        if not run_it:
                            message = get_message_for_yolo_label(label)
                            det = {
                                "is_waste": False,
                                "label": get_display_label(label),
                                "message": message,
                                "box": [x1, y1, x2, y2],
                                "box_color_hex": get_nonwaste_color(label),
                                "confidence": float(yolo_conf),
                                "interaction_type": "non_waste",
                                "location": self.get_object_location(x1, y1, x2, y2, frame_w, frame_h),
                                "raw_label": label
                            }
                            detections.append(det)
                            continue 
        
                        # ── ANALYST & VERIFIER ─────────────────────
                        crop_pil = Image.fromarray(crop_rgb)
                        crop_resized = crop_pil.resize((224,224))
                        crop_array   = np.expand_dims(np.array(crop_resized, dtype=np.float32)/255.0, axis=0)

                        probs, rev = self.analyze_waste_crop_v2(crop_array, crop_rgb, yolo_hint or label)
                        final_class, final_conf = self.robot_verifier_v2(probs, rev, crop_rgb, yolo_hint)
                        
                        # Fix 2: Handle Unknown or low confidence
                        if final_class.lower() not in VALID_WASTE or final_conf < 0.45:
                            det = {
                                "is_waste": False,
                                "label": "❓ Unknown",
                                "message": (
                                    "🤔 I can see something but I cannot "
                                    "identify it as Glass, Plastic, Metal "
                                    "or Paper. Try moving closer or "
                                    "improving the lighting!"
                                ) if final_class.lower() not in VALID_WASTE else (
                                    "🤔 Not sure what this is. "
                                    "Try: move closer, better lighting, "
                                    "place item on flat surface."
                                ),
                                "box": [x1, y1, x2, y2],
                                "box_color_hex": "#6b7280",
                                "confidence": final_conf,
                                "interaction_type": "unknown",
                                "location": self.get_object_location(x1, y1, x2, y2, frame_w, frame_h),
                                "raw_label": final_class.lower()
                            }
                            detections.append(det)
                            continue

                        cls_title = final_class.title()
                        override_hex = color_overrides.get(cls_title)
                        default = DEFAULT_BINS.get(cls_title, {"label": "Recycling", "hex": "#6b7280"})
                        
                        bin_hex = override_hex or default["hex"]
                        bin_label = default["label"]
                        
                        if override_hex:
                            bin_label = COLOR_NAMES.get(override_hex.lower(), bin_label)

                        det = {
                            "box": [x1, y1, x2, y2],
                            "location": self.get_object_location(x1,y1,x2,y2, frame_w, frame_h),
                            "is_waste": final_conf >= THRESHOLDS["mobilenet_confirmed"],
                            "label": self.format_box_label(final_class, final_conf),
                            "confidence": final_conf,
                            "raw_label": final_class.lower(),
                            "interaction_type": "waste",
                            "color_hex": bin_hex,
                            "bin_color": bin_label.lower()
                        }
                        
                        if final_conf >= THRESHOLDS["mobilenet_confirmed"]:
                            det["message"] = self.format_waste_message(final_class, bin_label, bin_hex)
                            det["box_color"], det["box_color_hex"] = "green", "#22c55e"
                        else:
                            det["message"] = UNCERTAIN_MSG
                            det["box_color"], det["box_color_hex"] = "yellow", "#eab308"
                            det["color_hex"] = "#eab308"
                            det["bin_color"] = "gray"
                        
                        detections.append(det)
            
            # ── FALLBACK: FULL IMAGE ────────────────────────
            if not detections:
                full_rgb = np.array(img_pil.convert("RGB"))
                if is_face_or_skin(full_rgb):
                    logger.info("[GATE] Full image is a face — skip")
                    return [{
                        "is_waste": False,
                        "label": "👤 Person",
                        "message": PERSON_MSG,
                        "box": [10, 10, frame_w-10, frame_h-10],
                        "box_color_hex": "#ef4444",
                        "interaction_type": "human",
                        "location": "Middle-Center",
                        "raw_label": "person"
                    }]

                full_resized = img_pil.resize((224,224))
                full_array = np.expand_dims(np.array(full_resized, dtype=np.float32)/255.0, axis=0)
                
                probs, rev = self.analyze_waste_crop_v2(full_array, full_rgb, "")
                final_class, final_conf = self.robot_verifier_v2(probs, rev, full_rgb, "")
                
                if final_conf >= THRESHOLDS["mobilenet_fallback"] and final_class.lower() in VALID_WASTE:
                    cls_title = final_class.title()
                    override_hex = color_overrides.get(cls_title)
                    default = DEFAULT_BINS.get(cls_title, {"label": "Recycling", "hex": "#6b7280"})
                    
                    bin_hex = override_hex or default["hex"]
                    bin_label = default["label"]
                    
                    if override_hex:
                        bin_label = COLOR_NAMES.get(override_hex.lower(), bin_label)

                    det = {
                        "box": [10, 10, frame_w-10, frame_h-10],
                        "location": "Middle-Center",
                        "is_waste": True,
                        "label": self.format_box_label(final_class, final_conf),
                        "confidence": final_conf,
                        "message": self.format_waste_message(final_class, bin_label, bin_hex),
                        "bin_color": bin_label.lower(),
                        "color_hex": bin_hex,
                        "box_color": "green",
                        "box_color_hex": "#22c55e",
                        "raw_label": final_class.lower(),
                        "interaction_type": "waste"
                    }
                    detections.append(det)
                else:
                    det = {
                        "is_waste": False,
                        "label": "❓ Unknown",
                        "message": (
                            "🤔 Not sure what this is. "
                            "Try: move closer, better lighting, "
                            "place item on flat surface."
                        ),
                        "box": [10, 10, frame_w-10, frame_h-10],
                        "box_color_hex": "#6b7280",
                        "confidence": final_conf,
                        "interaction_type": "unknown",
                        "location": "Middle-Center",
                        "raw_label": final_class.lower()
                    }
                    detections.append(det)
            
            return detections

        except (ValueError, RuntimeError, cv2.error, tf.errors.OpError) as e:
            logger.error(f"predict_scene failed ({type(e).__name__}): {e}")
            raise RuntimeError(f"Prediction error: {e}") from e
        except Exception as e:
            import traceback
            logger.error(f"Unexpected error in predict_scene: {e}")
            logger.error(traceback.format_exc())
            raise RuntimeError("Unexpected prediction failure") from e


# Lazy loading state
_models_loaded = False

def ensure_models_loaded():
    global _models_loaded
    if not _models_loaded:
        logger.info("Lazy loading models on first request...")
        model = get_model()
        model.load()
        _models_loaded = True

# Singleton instance access
model_instance = SortIQModel()

def get_model():
    return model_instance

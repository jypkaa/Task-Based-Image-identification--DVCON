"""
Task Weight Vectors — SpecOps 199 | DVCon India 2026
=====================================================
Implements Stage 1 Proposal Section 3 (Trained Weights):
  "For each task, a task weight vector (W1...W14) is predefined during training.
   These weight vectors encode the relationship between tasks and object categories.
   All task weight vectors are stored in DDR memory before inference begins."

80 COCO classes -> each W_t is an 80-dim vector.
Weight value = relevance of that COCO class to the task.
Normalized to unit vector so dot-product scoring is comparable across tasks.
"""

import numpy as np

# ── COCO 80 class names (0-indexed, matching EfficientDet pipeline output) ──
CLASS_NAMES_COCO = {
    0:"person", 1:"bicycle", 2:"car", 3:"motorcycle", 4:"airplane",
    5:"bus", 6:"train", 7:"truck", 8:"boat", 9:"traffic light",
    10:"fire hydrant", 11:"stop sign", 12:"parking meter", 13:"bench",
    14:"bird", 15:"cat", 16:"dog", 17:"horse", 18:"sheep", 19:"cow",
    20:"elephant", 21:"bear", 22:"zebra", 23:"giraffe", 24:"backpack",
    25:"umbrella", 26:"handbag", 27:"tie", 28:"suitcase", 29:"frisbee",
    30:"skis", 31:"snowboard", 32:"sports ball", 33:"kite", 34:"baseball bat",
    35:"baseball glove", 36:"skateboard", 37:"surfboard", 38:"tennis racket",
    39:"bottle", 40:"wine glass", 41:"cup", 42:"fork", 43:"knife",
    44:"spoon", 45:"bowl", 46:"banana", 47:"apple", 48:"sandwich",
    49:"orange", 50:"broccoli", 51:"carrot", 52:"hot dog", 53:"pizza",
    54:"donut", 55:"cake", 56:"chair", 57:"couch", 58:"potted plant",
    59:"bed", 60:"dining table", 61:"toilet", 62:"tv", 63:"laptop",
    64:"mouse", 65:"remote", 66:"keyboard", 67:"cell phone", 68:"microwave",
    69:"oven", 70:"toaster", 71:"sink", 72:"refrigerator", 73:"book",
    74:"clock", 75:"vase", 76:"scissors", 77:"teddy bear", 78:"hair drier",
    79:"toothbrush"
}

# Reverse: name -> class_id
CLASS_ID = {v: k for k, v in CLASS_NAMES_COCO.items()}

N_CLASSES = 80

# ── 14 Task names (Task 0 = general, Tasks 1-13 = contest queries) ──
TASK_NAMES = {
    0:  "general detection",
    1:  "serve wine",
    2:  "eat food",
    3:  "drink water",
    4:  "sit down",
    5:  "travel",
    6:  "play sport",
    7:  "cook",
    8:  "write",
    9:  "read",
    10: "carry bag",
    11: "extinguish fire",
    12: "cut",
    13: "clean",
}

# Keywords to help identify task from free-text query
TASK_KEYWORDS = {
    1:  ["wine", "serve wine", "pour wine", "glass"],
    2:  ["eat", "food", "meal", "lunch", "dinner", "breakfast", "hungry"],
    3:  ["drink", "water", "thirsty", "beverage"],
    4:  ["sit", "seat", "rest", "relax"],
    5:  ["travel", "commute", "go", "transport", "ride", "fly", "drive"],
    6:  ["sport", "play", "game", "ball", "match", "exercise"],
    7:  ["cook", "bake", "heat", "boil", "kitchen", "prepare food"],
    8:  ["write", "pen", "draw", "note", "sign"],
    9:  ["read", "book", "study", "newspaper"],
    10: ["bag", "carry", "luggage", "backpack", "handbag"],
    11: ["fire", "extinguish", "firefight", "hydrant"],
    12: ["cut", "slice", "chop", "trim", "scissors"],
    13: ["clean", "wash", "brush", "teeth", "hygiene"],
}


def _make_vector(class_weights: dict) -> np.ndarray:
    """Build normalized 80-dim task weight vector from {class_name: weight}."""
    vec = np.zeros(N_CLASSES, dtype=np.float32)
    for name, w in class_weights.items():
        cid = CLASS_ID.get(name)
        if cid is not None:
            vec[cid] = w
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


# ── 14 Task Weight Vectors W_1 ... W_14 ──
# Stored in memory (simulates DDR memory in Stage 1 proposal §3)
TASK_WEIGHT_VECTORS = {

    # W_0: General detection — all classes equal weight
    0: _make_vector({name: 1.0 for name in CLASS_NAMES_COCO.values()}),

    # W_1: Serve wine
    # Primary: wine glass. Also: bottle, cup, dining table
    1: _make_vector({
        "wine glass":   1.00,
        "bottle":       0.75,
        "cup":          0.50,
        "dining table": 0.35,
        "bowl":         0.20,
    }),

    # W_2: Eat food
    # Primary: fork, spoon. Also: food items, bowl, knife, dining table
    2: _make_vector({
        "fork":         1.00,
        "spoon":        1.00,
        "knife":        0.80,
        "bowl":         0.75,
        "dining table": 0.65,
        "sandwich":     0.55,
        "pizza":        0.55,
        "hot dog":      0.55,
        "banana":       0.45,
        "apple":        0.45,
        "orange":       0.45,
        "cake":         0.45,
        "donut":        0.45,
        "broccoli":     0.35,
        "carrot":       0.35,
        "cup":          0.30,
    }),

    # W_3: Drink water
    # Primary: cup. Also: bottle, bowl, sink
    3: _make_vector({
        "cup":    1.00,
        "bottle": 0.90,
        "bowl":   0.40,
        "sink":   0.30,
    }),

    # W_4: Sit down
    # Primary: chair, couch. Also: bench, bed, toilet
    4: _make_vector({
        "chair":  1.00,
        "couch":  0.95,
        "bench":  0.80,
        "bed":    0.60,
        "toilet": 0.40,
    }),

    # W_5: Travel
    # Primary: car, airplane, train. Also: suitcase, motorcycle, bus
    5: _make_vector({
        "airplane":   1.00,
        "car":        0.95,
        "train":      0.90,
        "bus":        0.85,
        "motorcycle": 0.70,
        "bicycle":    0.65,
        "boat":       0.60,
        "truck":      0.50,
        "suitcase":   0.80,
        "backpack":   0.50,
    }),

    # W_6: Play sport
    # Primary: sports ball, tennis racket. Also: baseball bat, frisbee, surfboard
    6: _make_vector({
        "sports ball":    1.00,
        "tennis racket":  0.95,
        "baseball bat":   0.90,
        "baseball glove": 0.85,
        "frisbee":        0.75,
        "skateboard":     0.70,
        "surfboard":      0.70,
        "skis":           0.70,
        "snowboard":      0.65,
        "kite":           0.55,
        "bicycle":        0.40,
    }),

    # W_7: Cook
    # Primary: oven, microwave, knife. Also: bowl, sink, refrigerator
    7: _make_vector({
        "oven":         1.00,
        "microwave":    0.95,
        "knife":        0.90,
        "toaster":      0.75,
        "bowl":         0.70,
        "fork":         0.65,
        "spoon":        0.65,
        "sink":         0.60,
        "dining table": 0.55,
        "refrigerator": 0.55,
        "bottle":       0.35,
    }),

    # W_8: Write
    # Primary: book, keyboard. Also: laptop, cell phone, scissors
    8: _make_vector({
        "book":       1.00,
        "keyboard":   0.85,
        "laptop":     0.75,
        "scissors":   0.55,
        "cell phone": 0.45,
        "mouse":      0.40,
    }),

    # W_9: Read
    # Primary: book. Also: laptop, tv, cell phone
    9: _make_vector({
        "book":       1.00,
        "laptop":     0.65,
        "tv":         0.55,
        "cell phone": 0.55,
        "remote":     0.35,
    }),

    # W_10: Carry bag
    # Primary: backpack, handbag, suitcase
    10: _make_vector({
        "backpack":  1.00,
        "handbag":   1.00,
        "suitcase":  0.95,
        "umbrella":  0.35,
    }),

    # W_11: Extinguish fire
    # Primary: fire hydrant (closest COCO class). Also: bottle (spray proxy)
    11: _make_vector({
        "fire hydrant": 1.00,
        "bottle":       0.40,
        "sink":         0.30,
    }),

    # W_12: Cut
    # Primary: knife, scissors. Also: fork (cutting action)
    12: _make_vector({
        "knife":    1.00,
        "scissors": 1.00,
        "fork":     0.50,
    }),

    # W_13: Clean
    # Primary: toothbrush, sink. Also: hair drier, toilet
    13: _make_vector({
        "toothbrush": 1.00,
        "sink":       0.95,
        "hair drier": 0.75,
        "toilet":     0.65,
        "bottle":     0.30,
    }),
}

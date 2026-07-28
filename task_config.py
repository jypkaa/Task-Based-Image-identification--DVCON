"""
task_config.py
==============
14 Task definitions from COCO-Tasks paper.
Each task has a weight vector W_t (80-dim, one per COCO class).
These are the task weight vectors W1...W14 from the Stage 1 proposal.
"""

COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
    'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# 14 tasks with relevance weights per COCO class (0.0 = irrelevant, 1.0 = most relevant)
TASK_DEFINITIONS = {
    1: {
        "name": "step on something",
        "weights": {
            "skateboard": 1.0, "surfboard": 0.9, "snowboard": 0.9,
            "skis": 0.8, "frisbee": 0.7, "sports ball": 0.6,
            "bench": 0.5, "chair": 0.4, "couch": 0.3, "bed": 0.3,
            "dining table": 0.4, "suitcase": 0.4, "book": 0.3,
        }
    },
    2: {
        "name": "sit comfortably",
        "weights": {
            "couch": 1.0, "chair": 0.95, "bench": 0.85, "bed": 0.8,
            "toilet": 0.5, "motorcycle": 0.3, "bicycle": 0.25,
            "boat": 0.35, "dining table": 0.2, "suitcase": 0.2,
        }
    },
    3: {
        "name": "place flowers",
        "weights": {
            "vase": 1.0, "potted plant": 0.9, "bowl": 0.7,
            "cup": 0.65, "bottle": 0.6, "wine glass": 0.6,
            "sink": 0.4, "dining table": 0.3,
        }
    },
    4: {
        "name": "get potatoes out of fire",
        "weights": {
            "fork": 1.0, "knife": 0.95, "spoon": 0.9,
            "scissors": 0.7, "baseball bat": 0.5,
            "oven": 0.8, "microwave": 0.5, "toaster": 0.4,
        }
    },
    5: {
        "name": "water plant",
        "weights": {
            "bottle": 1.0, "cup": 0.7, "bowl": 0.5,
            "wine glass": 0.7, "vase": 0.6, "sink": 0.3,
            "umbrella": 0.4, "potted plant": 0.3,
        }
    },
    6: {
        "name": "get lemon out of tea",
        "weights": {
            "spoon": 1.0, "fork": 0.95, "knife": 0.85,
            "scissors": 0.5, "toothbrush": 0.3, "cup": 0.4,
        }
    },
    7: {
        "name": "dig hole",
        "weights": {
            "spoon": 1.0, "fork": 0.9, "knife": 0.8,
            "scissors": 0.7, "baseball bat": 0.65,
            "tennis racket": 0.5, "skateboard": 0.4,
        }
    },
    8: {
        "name": "open bottle of beer",
        "weights": {
            "knife": 1.0, "scissors": 0.9, "fork": 0.75,
            "spoon": 0.6, "bottle": 0.5, "baseball bat": 0.4,
        }
    },
    9: {
        "name": "open parcel",
        "weights": {
            "scissors": 1.0, "knife": 0.95, "fork": 0.6,
            "spoon": 0.4, "baseball bat": 0.35,
        }
    },
    10: {
        "name": "serve wine",
        "weights": {
            "wine glass": 1.0, "cup": 0.85, "bowl": 0.6,
            "bottle": 0.5, "vase": 0.4, "fork": 0.2,
        }
    },
    11: {
        "name": "pour sugar",
        "weights": {
            "bowl": 1.0, "cup": 0.9, "wine glass": 0.8,
            "bottle": 0.85, "vase": 0.6, "spoon": 0.5,
        }
    },
    12: {
        "name": "smear butter",
        "weights": {
            "knife": 1.0, "spoon": 0.85, "fork": 0.7,
            "scissors": 0.4, "toothbrush": 0.35,
        }
    },
    13: {
        "name": "extinguish fire",
        "weights": {
            "fire hydrant": 1.0, "bottle": 0.9, "cup": 0.8,
            "bowl": 0.75, "wine glass": 0.65, "umbrella": 0.5,
            "vase": 0.55, "sink": 0.85,
        }
    },
    14: {
        "name": "pound carpet",
        "weights": {
            "baseball bat": 1.0, "tennis racket": 0.9,
            "skateboard": 0.75, "surfboard": 0.7,
            "snowboard": 0.65, "skis": 0.6,
            "umbrella": 0.45, "scissors": 0.5,
        }
    },
}


def get_task_weight_vector(task_id: int) -> list:
    """Return 80-dim weight vector W_t for the given task."""
    task = TASK_DEFINITIONS[task_id]
    vector = [0.0] * len(COCO_CLASSES)
    for cls_name, score in task["weights"].items():
        if cls_name in COCO_CLASSES:
            vector[COCO_CLASSES.index(cls_name)] = score
    return vector


def match_task_by_name(task_str: str) -> int:
    """Fuzzy-match task string to task ID (1-14)."""
    task_str = task_str.lower().strip()
    for tid, tdef in TASK_DEFINITIONS.items():
        if task_str == tdef["name"].lower():
            return tid
    # word overlap
    best_id, best_score = 1, -1
    for tid, tdef in TASK_DEFINITIONS.items():
        overlap = len(set(task_str.split()) & set(tdef["name"].lower().split()))
        if overlap > best_score:
            best_score, best_id = overlap, tid
    return best_id

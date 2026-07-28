# SpecOps 199 — DVCon India 2026 Stage 2A
## Task-Aware Object Detection Pipeline — EfficientDet-D0

**Team:** K Jyothsna Padma | B Mythri Reddy | MSN Subhiksha

> Please refer to the report for final results.

**Dataset used:** [COCO Dataset](https://cocodataset.org/#home)

---

## Architecture

```
Input Image + Task Description
        │
        ▼
  EfficientDet-D0  (FPGA-targeted detector)
  ├── Backbone : EfficientNet-B0  (compound-scaled CNN)
  ├── Neck     : BiFPN            (bidirectional feature pyramid)
  └── Head     : Class + Box prediction
  → bounding boxes + class labels + confidence scores
        │
        ▼
  Task Identification
  "serve wine" → Task ID 10 → load W_10 from memory
        │
        ▼
  Relevance Scoring  (VEGA CPU stage)
  score_i = x_i · W_t   (dot product)
        │
        ▼
  Relative Thresholding
  keep objects with score ≥ 0.70 × max_score
        │
        ▼
  Output: Most relevant object + bounding box
```

### Why EfficientDet?

EfficientDet (Tan et al., 2020) uses compound scaling across depth, width, and resolution, achieving a better accuracy/efficiency trade-off than YOLO for FPGA deployment after INT8 quantization. The BiFPN neck fuses multi-scale features bidirectionally, improving small-object detection.

---

## All 14 Supported Tasks

| ID | Task Name | Best Object In Image |
|----|-----------|------------------------|
| 1  | Step on something | Skateboard, surfboard, snowboard |
| 2  | Sit comfortably | Couch, chair, bench |
| 3  | Place flowers | Vase, bowl, cup |
| 4  | Get potatoes out of fire | Fork, knife, spoon |
| 5  | Water plant | Bottle, cup, sink |
| 6  | Get lemon out of tea | Spoon, fork, knife |
| 7  | Dig hole | Spoon, fork, knife |
| 8  | Open bottle of beer | Knife, scissors |
| 9  | Open parcel | Scissors, knife |
| 10 | Serve wine | Wine glass, cup |
| 11 | Pour sugar | Bowl, bottle, cup |
| 12 | Smear butter | Knife, spoon |
| 13 | Extinguish fire | Fire hydrant, bottle, sink |
| 14 | Pound carpet | Baseball bat, tennis racket |

---

## File Structure

```
specops_final/
├── run.py           # Main script (single image + task)
├── batch_run.py     # Run all 14 tasks at once
├── detector.py      # EfficientDet-D0 object detection   ← CHANGED
├── scorer.py        # Dot product relevance scoring
├── task_config.py   # 14 task weight vectors W1...W14
├── visualizer.py    # Bounding box drawing
├── requirements.txt # pip dependencies                   ← CHANGED
└── README.md        # This file
```

---

## Scoring Formula

For each detected object `i` in the image, with task `t`:

```
x_i[c]    = confidence_i + area_bonus          (80-dim feature vector)
W_t[c]    = task relevance weight for class c  (from task_config.py)
dot_score = x_i · W_t
class_wt  = W_t[class_of_i]
relevance = 0.5 × dot_score + 0.5 × (class_wt × confidence_i)
```

Objects with `relevance ≥ 0.70 × max_relevance` are kept (top-5 max).

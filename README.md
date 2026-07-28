# SpecOps 199 — Task-Oriented Object Identification
## DVCon India 2026 | Design Contest Stage 2A

**Team:** K Jyothsna Padma, B Mythri Reddy, MSN Subhiksha  
**Institute:** NIT Warangal | Team ID: 199

---

## How This Maps to Your Stage 1 Proposal

```
Stage 1 Proposal Block Diagram          This Code
────────────────────────────            ─────────────────────────────────────
DataSet (COCO)                          COCO pretrained weights (auto-download)
  ↓                                       ↓
Train Model (EfficientNet backbone)     effdet library: tf_efficientdet_d0
  ↓                                       ↓
Trained Weights                         pretrained=True  (model_setup.py)
  ↓                                       ↓
Quantization (FP32 → INT8)              _export_onnx() → ONNXRuntime INT8
  ↓                                       ↓
Model Conversion (ONNX → FPGA)          .onnx file (Stage 3: Vivado bitstream)
  ↓ DEPLOYMENT                            ↓ PIPELINE
Input Image + Task Query                --image + --task arguments
  ↓                                       ↓
Preprocessing (Resize+Normalize)        stage1_preprocess()  — pipeline.py
  ↓                                       ↓
FPGA: CNN Backbone + Detection Head     stage2_fpga_detection()  — pipeline.py
  ↓                                       ↓
Detected Objects + Feature Vectors      list of {class, conf, bbox, feat_vec}
  ↓                                       ↓
Task Input → Task→Vector                _identify_task() + W_t lookup
  ↓                                       ↓
Relevance Scoring: score = x · W        np.dot(x, W_t)  — pipeline.py
  ↓                                       ↓
Relative Thresholding                   stage4_relative_threshold() — 0.80×top
  ↓                                       ↓
Final Task-Aware Output                 primary_object + selected candidates
```

---

## Setup

```bash
# Step 1 — Install dependencies
pip install -r requirements.txt

# Step 2 — Quick demo (downloads 5 sample images, runs all 14 tasks)
python main.py --demo

# Step 3 — Your own images: put them in test_images/ then:
python main.py --images_dir ./test_images
```

---

## Run Options

```bash
# All 14 tasks × all images in folder (for contest evaluation)
python main.py --images_dir ./test_images

# All 14 tasks on one image
python main.py --image ./test_images/kitchen.jpg

# One specific task on one image
python main.py --image ./test_images/kitchen.jpg --task "cook"

# Use more accurate (but slower) EfficientDet variant
python main.py --images_dir ./test_images --variant tf_efficientdet_d1
```

---

## The 14 Evaluation Tasks

| ID | Task Query       | Primary COCO Objects              |
|----|-----------------|-----------------------------------|
| 1  | serve wine      | wine glass, bottle, cup           |
| 2  | eat food        | fork, spoon, bowl, food items     |
| 3  | drink water     | cup, bottle                       |
| 4  | sit down        | chair, couch, bench               |
| 5  | travel          | airplane, car, train, suitcase    |
| 6  | play sport      | sports ball, tennis racket        |
| 7  | cook            | oven, microwave, knife            |
| 8  | write           | book, keyboard, laptop            |
| 9  | read            | book, laptop, tv                  |
| 10 | carry bag       | backpack, handbag, suitcase       |
| 11 | extinguish fire | fire hydrant                      |
| 12 | cut             | knife, scissors                   |
| 13 | clean           | toothbrush, sink                  |

---

## Output Files

```
outputs/
├── <image>__<task>.jpg         ← annotated image (green=selected, red=rejected)
├── evaluation_results.json     ← full structured results for all images × tasks
└── summary_table.txt           ← readable table for your Stage 2A report
```

---

## File Structure

```
specops_pipeline/
├── main.py           ← Entry point / evaluation runner
├── pipeline.py       ← 4-stage pipeline (§7 preprocess → §8 detect → §9 score → §11 threshold)
├── model_setup.py    ← EfficientDet loader, ONNX export, inferencer class
├── task_vectors.py   ← 14 task weight vectors W_1...W_14 (DDR memory simulation)
├── requirements.txt  ← Dependencies
├── test_images/      ← Put your test images here
└── outputs/          ← Results saved here automatically
```

---

## Scoring Logic (Proposal §9, §11)

```
score = x · W_t        (dot product)

  x   = object feature vector (80-dim, one-hot class × confidence)
  W_t = task weight vector for task t (80-dim, normalized)

Relative thresholding (§11):
  top_score = max(all scores)
  threshold = 0.80 × top_score
  selected  = { obj | score(obj) ≥ threshold }
```

---

## Stage 2A Deliverables Checklist

- [x] Functionally correct pipeline — all 14 tasks, CPU inference
- [x] EfficientNet backbone (EfficientDet) — matches Stage 1 proposal
- [x] ONNX export — proposal §5 (PyTorch → ONNX)
- [x] Multi-image evaluation
- [x] Annotated output images for results page
- [ ] 2-page report (use summary_table.txt + output images)
- [ ] Demo video (screen record `python main.py --demo`)
- Deadline: May 5, 2026

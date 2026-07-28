# SpecOps 199 — DVCon India 2026 Stage 2A
## Task-Aware Object Detection Pipeline — EfficientDet-D0

**Team:** K Jyothsna Padma | B Mythri Reddy | MSN Subhiksha  
**Institute:** NIT Warangal

---

## Architecture (as proposed)

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

**Why EfficientDet?**  
EfficientDet (Tan et al., 2020) uses compound scaling across depth, width,
and resolution, achieving a better accuracy/efficiency trade-off than YOLO
for FPGA deployment after INT8 quantization. The BiFPN neck fuses
multi-scale features bidirectionally, improving small-object detection.

---

## STEP 1 — Install Python

Python 3.9 or higher required.

```bash
python --version
```

---

## STEP 2 — Create Virtual Environment

**Linux / Mac:**
```bash
python3 -m venv specops_env
source specops_env/bin/activate
```

**Windows:**
```cmd
python -m venv specops_env
specops_env\Scripts\activate
```

---

## STEP 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

Installs:
- `effdet` — EfficientDet-D0 object detector
- `timm` — EfficientNet-B0 backbone + weight registry
- `torch` / `torchvision` — deep learning runtime (CPU)
- `opencv-python` — image reading and drawing
- `numpy` — numerical operations
- `Pillow` — image utilities

On first run, EfficientDet-D0 pretrained COCO weights (~16 MB) download
automatically from GitHub and are cached in `~/.cache/effdet/`.

---

## STEP 4 — Run on a Single Image

```bash
python run.py --image your_image.jpg --task "serve wine"
```

This will:
1. Preprocess `your_image.jpg` (resize → 512×512, normalize)
2. Detect all objects using EfficientDet-D0
3. Score each object against the "serve wine" task weight vector W₁₀
4. Print the best matching object
5. Save an annotated output image to `outputs/result.jpg`

**More examples:**
```bash
python run.py --image sofa.jpg         --task "sit comfortably"
python run.py --image kitchen.jpg      --task "serve wine"
python run.py --image desk.jpg         --task "open parcel"
python run.py --image street.jpg       --task "extinguish fire"
python run.py --image living_room.jpg  --task "place flowers"
python run.py --image bedroom.jpg      --task "step on something"

# Use task ID instead of name
python run.py --image sofa.jpg         --task 2

# Save to specific output path
python run.py --image sofa.jpg         --task "sit comfortably" --output result.jpg

# Show all detected objects
python run.py --image sofa.jpg         --task "sit comfortably" --verbose

# List all 14 tasks
python run.py --list_tasks
```

---

## STEP 5 — Run All 14 Tasks at Once

Put test images in `test_images/` named `task1.jpg` … `task14.jpg`, then:

```bash
python batch_run.py --images_dir ./test_images --output_dir ./outputs
```

Results saved in `./outputs/` with annotated images + `summary.txt`.

---

## All 14 Supported Tasks

| ID | Task Name                 | Best Object In Image              |
|----|---------------------------|-----------------------------------|
| 1  | step on something         | skateboard, surfboard, snowboard  |
| 2  | sit comfortably           | couch, chair, bench               |
| 3  | place flowers             | vase, bowl, cup                   |
| 4  | get potatoes out of fire  | fork, knife, spoon                |
| 5  | water plant               | bottle, cup, sink                 |
| 6  | get lemon out of tea      | spoon, fork, knife                |
| 7  | dig hole                  | spoon, fork, knife                |
| 8  | open bottle of beer       | knife, scissors                   |
| 9  | open parcel               | scissors, knife                   |
| 10 | serve wine                | wine glass, cup                   |
| 11 | pour sugar                | bowl, bottle, cup                 |
| 12 | smear butter              | knife, spoon                      |
| 13 | extinguish fire           | fire hydrant, bottle, sink        |
| 14 | pound carpet              | baseball bat, tennis racket       |

---

## File Structure

```
specops_final/
├── run.py           ← Main script (single image + task)
├── batch_run.py     ← Run all 14 tasks at once
├── detector.py      ← EfficientDet-D0 object detection  ← CHANGED
├── scorer.py        ← Dot product relevance scoring
├── task_config.py   ← 14 task weight vectors W1...W14
├── visualizer.py    ← Bounding box drawing
├── requirements.txt ← pip dependencies                  ← CHANGED
└── README.md        ← This file
```

---

## Scoring Formula

For each detected object `i` in image, with task `t`:

```
x_i[c]    = confidence_i + area_bonus          (80-dim feature vector)
W_t[c]    = task relevance weight for class c  (from task_config.py)

dot_score = x_i · W_t
class_wt  = W_t[class_of_i]

relevance = 0.5 × dot_score + 0.5 × (class_wt × confidence_i)
```

Objects with `relevance ≥ 0.70 × max_relevance` are kept (top-5 max).

---

## Troubleshooting

**"effdet / timm not found"**
```bash
pip install effdet timm
```

**Weights not downloading automatically**
- Check internet connection
- Download manually from:  
  `https://github.com/rwightman/efficientdet-pytorch/releases/download/v0.1/tf_efficientdet_d0-d92fd44f.pth`
- Place at: `~/.cache/effdet/tf_efficientdet_d0.pth`

**"No relevant object found"**
- Use `--verbose` to see what was detected
- Try `--threshold 0.50` to lower the relative threshold
- Ensure the image contains clear, well-lit task-relevant objects

**Slow on CPU**
- EfficientDet-D0 on CPU: ~2–5 seconds per image (normal)
- For real deployment: quantize to INT8 and run on FPGA (Genesys 2)
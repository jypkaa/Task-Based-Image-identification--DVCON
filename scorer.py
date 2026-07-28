"""
scorer.py
=========
Task-aware relevance scoring using dot product.

Enhanced version:
- keeps BEST object
- also keeps OTHER valid candidates
- supports multiple useful objects per task
"""

import numpy as np

from task_config import (
    COCO_CLASSES,
    TASK_DEFINITIONS,
    get_task_weight_vector,
    match_task_by_name
)


def score_detections(detections: list, task_weight_vector: list) -> list:
    """
    Compute relevance score for each detection.
    score_i = x_i · W_t  (dot product)

    Returns detections sorted by relevance score (descending).
    """

    W_t = np.array(task_weight_vector, dtype=np.float32)

    scored = []

    for det in detections:

        x_i = det.get(
            "feature_vector",
            np.zeros(80, dtype=np.float32)
        )

        cls_id = det["class_id"]
        conf   = det["confidence"]

        # Dot product score
        dot_score = float(np.dot(x_i, W_t))

        # Direct class weight
        class_weight = float(W_t[cls_id])

        # Combined relevance score
        relevance = (
            0.5 * dot_score +
            0.5 * (class_weight * conf)
        )

        d = dict(det)

        d["relevance_score"] = relevance
        d["class_weight"]    = class_weight

        scored.append(d)

    # Sort highest relevance first
    scored.sort(
        key=lambda d: d["relevance_score"],
        reverse=True
    )

    return scored


def relative_threshold(
    scored: list,
    threshold: float = 0.70,
    top_k: int = 5
) -> list:
    """
    Keep detections whose score >= threshold × max_score.

    This allows:
    - best object
    - other valid objects

    Example:
        bottle
        cup
        sink

    instead of only one object.
    """

    if not scored:
        return []

    max_score = scored[0]["relevance_score"]

    # If everything scores 0
    if max_score <= 0:
        return scored[:1]

    cutoff = max_score * threshold

    candidates = []

    for d in scored:

        if d["relevance_score"] >= cutoff:
            candidates.append(d)

    return candidates[:top_k]


def run_task_aware_selection(
    detections: list,
    task_input: str,
    threshold: float = 0.70,
    top_k: int = 5
) -> tuple:
    """
    Full task-aware selection pipeline.

    Steps:
      1. Identify task
      2. Load W_t
      3. Score detections
      4. Keep multiple valid candidates

    Returns:
        (
            selected_detections,
            task_id,
            task_name
        )
    """

    # --------------------------------------------------
    # STEP 1 — Task identification
    # --------------------------------------------------

    task_id = match_task_by_name(task_input)

    task_name = TASK_DEFINITIONS[task_id]["name"]

    print("\n" + "=" * 60)
    print(f"[Scorer] Task [{task_id}]: {task_name}")
    print("=" * 60)

    # --------------------------------------------------
    # STEP 2 — Load task weight vector
    # --------------------------------------------------

    W_t = get_task_weight_vector(task_id)

    top_classes = [
        (COCO_CLASSES[i], W_t[i])
        for i in range(80)
        if W_t[i] > 0
    ]

    top_classes.sort(
        key=lambda x: -x[1]
    )

    print("\n[Scorer] Relevant task classes:")

    for cls_name, weight in top_classes[:10]:
        print(f"  - {cls_name:20s} weight={weight:.2f}")

    # --------------------------------------------------
    # STEP 3 — Handle empty detections
    # --------------------------------------------------

    if not detections:

        print("\n[Scorer] No detections found.")

        return [], task_id, task_name

    # --------------------------------------------------
    # STEP 4 — Score detections
    # --------------------------------------------------

    scored = score_detections(
        detections,
        W_t
    )

    print(f"\n[Scorer] Scoring {len(scored)} detections:\n")

    for d in scored:

        print(
            f"  {d['class_name']:20s} "
            f"conf={d['confidence']:.3f}   "
            f"W_t={d['class_weight']:.2f}   "
            f"score={d['relevance_score']:.4f}"
        )

    # --------------------------------------------------
    # STEP 5 — Relative thresholding
    # --------------------------------------------------

    selected = relative_threshold(
        scored,
        threshold,
        top_k
    )

    # --------------------------------------------------
    # STEP 6 — Print selected objects
    # --------------------------------------------------

    print("\n" + "-" * 60)
    print("[Scorer] FINAL SELECTED OBJECTS")
    print("-" * 60)

    for idx, d in enumerate(selected):

        if idx == 0:
            tag = "BEST"
        else:
            tag = "ALSO VALID"

        print(
            f"{tag:12s} -> "
            f"{d['class_name']:20s} "
            f"(score={d['relevance_score']:.4f})"
        )

    print("-" * 60 + "\n")

    return selected, task_id, task_name
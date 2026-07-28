#!/usr/bin/env python3

"""
SpecOps 199 — DVCon India 2026 Design Contest: Stage 2A
Task-Aware Object Detection Pipeline
"""

import argparse
import sys
import os
import time
import cv2
import numpy as np

from pathlib import Path


from task_config import (
    TASK_DEFINITIONS,
    match_task_by_name
)

from detector import (
    load_model,
    detect_from_file
)

from scorer import (
    run_task_aware_selection
)

from visualizer import (
    draw_results
)


BANNER = """
╔══════════════════════════════════════════════════════════════╗
║  SpecOps 199 — DVCon India 2026 Design Contest: Stage 2A    ║
║  Task-Aware Object Detection  [EfficientDet-D0 + BiFPN]     ║
║  K Jyothsna Padma | B Mythri Reddy | MSN Subhiksha          ║
║  NIT Warangal                                                ║
╚══════════════════════════════════════════════════════════════╝
"""


# ============================================================
# TASK PARSER
# ============================================================

def parse_task(task_str: str):

    task_str = task_str.strip()

    # numeric task
    if task_str.isdigit():

        tid = int(task_str)

        if 1 <= tid <= 14:
            return tid, TASK_DEFINITIONS[tid]["name"]

        print(f"[Pipeline] Task ID must be 1-14. Got {tid}.")
        sys.exit(1)

    # task name
    tid = match_task_by_name(task_str)

    return tid, TASK_DEFINITIONS[tid]["name"]


# ============================================================
# PRINT TASK LIST
# ============================================================

def print_task_list():

    print("\nAll 14 supported tasks:\n")

    for tid, td in TASK_DEFINITIONS.items():

        print(f"  {tid:2d}.  {td['name']}")

    print()


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_pipeline(
    image_path: str,
    task_input: str,
    output_path: str,
    threshold: float = 0.70,
    top_k: int = 5,
    verbose: bool = False
):

    print(BANNER)

    total_start = time.time()

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    print("[Pipeline] Loading YOLOv8 model...")

    model = load_model()

    # --------------------------------------------------------
    # PARSE TASK
    # --------------------------------------------------------

    task_id, task_name = parse_task(task_input)

    print(f"\n[Pipeline] Image : {image_path}")
    print(f"[Pipeline] Task  : [{task_id}] {task_name}")

    # --------------------------------------------------------
    # STAGE 1 — DETECTION
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("[Pipeline] STAGE 1 — OBJECT DETECTION")
    print("=" * 60)

    det_start = time.time()

    detections = detect_from_file(
        image_path,
        model
    )

    det_time = time.time() - det_start

    print(
        f"\n[Pipeline] {len(detections)} "
        f"objects detected in {det_time:.3f}s"
    )

    # --------------------------------------------------------
    # VERBOSE OUTPUT
    # --------------------------------------------------------

    if verbose and detections:

        print("\n[Pipeline] All detected objects:\n")

        for d in detections:

            print(
                f"  - {d['class_name']:20s} "
                f"conf={d['confidence']:.3f}"
            )

    # --------------------------------------------------------
    # STAGE 2 — TASK-AWARE SCORING
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("[Pipeline] STAGE 2 — TASK-AWARE SCORING")
    print("=" * 60)

    score_start = time.time()

    selected, task_id, task_name = run_task_aware_selection(
        detections=detections,
        task_input=task_input,
        threshold=threshold,
        top_k=top_k
    )

    score_time = time.time() - score_start

    # --------------------------------------------------------
    # FINAL RESULTS
    # --------------------------------------------------------

    total_time = time.time() - total_start

    print("\n" + "═" * 60)

    print(f"  TASK  : [{task_id}] {task_name.upper()}")

    print(f"  IMAGE : {os.path.basename(image_path)}")

    print("─" * 60)

    # --------------------------------------------------------
    # NO VALID OBJECTS
    # --------------------------------------------------------

    if (
        not selected or
        selected[0]["relevance_score"] <= 0
    ):

        print("\n  RESULT: No relevant object found.")

        print(
            "  Tip   : Use image containing "
            "task-related COCO objects."
        )

    # --------------------------------------------------------
    # SHOW RESULTS
    # --------------------------------------------------------

    else:

        print(
            f"\n  {len(selected)} valid object(s) found:\n"
        )

        for rank, d in enumerate(selected, start=1):

            # best object
            if rank == 1:
                tag = "★ BEST"
            else:
                tag = "○ ALSO VALID"

            print(
                f"  {tag}"
            )

            print(
                f"     Object     : "
                f"{d['class_name'].upper()}"
            )

            print(
                f"     Relevance  : "
                f"{d['relevance_score']:.4f}"
            )

            print(
                f"     Confidence : "
                f"{d['confidence']:.3f}"
            )

            print(
                f"     W_t[c]     : "
                f"{d['class_weight']:.3f}"
            )

            print(
                f"     BBox       : "
                f"{[round(v) for v in d['bbox']]}"
            )

            print()

    # --------------------------------------------------------
    # TIMING
    # --------------------------------------------------------

    print("─" * 60)

    print(
        f"  Time : "
        f"detection={det_time:.3f}s   "
        f"scoring={score_time:.4f}s   "
        f"total={total_time:.3f}s"
    )

    print("═" * 60)
       # --------------------------------------------------------
    # SAVE VISUALIZED OUTPUT
    # --------------------------------------------------------

    print("\n[Pipeline] Creating visualization...")

    try:

        img = cv2.imread(image_path)

        if img is None:

            print("[Pipeline] ERROR: Could not read image.")

            return selected

        vis = draw_results(
            img=img,
            detections=detections,
            selected=selected,
            task_name=task_name,
            task_id=task_id
        )

        # create outputs folder
        output_dir = "outputs"

        os.makedirs(output_dir, exist_ok=True)

        # fixed output filename
        output_file = os.path.join(
            output_dir,
            "result.jpg"
        )

                # make sure image is uint8
        if vis.dtype != "uint8":
            vis = vis.astype("uint8")

        # force valid range
        vis = np.clip(vis, 0, 255)

        # save image
        success = cv2.imwrite(
            output_file,
            vis
        )

        if success:

            abs_path = os.path.abspath(output_file)

            print("\n[Pipeline] SUCCESS!")
            print(f"[Pipeline] Output saved at:")
            print(abs_path)

        else:

            print("\n[Pipeline] ERROR: cv2.imwrite failed.")

    except Exception as e:

        print("\n[Pipeline] VISUALIZATION ERROR:")
        print(e)

    return selected

# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "SpecOps 199 Task-Aware Object Detection"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--image",
        "-i",
        default=None,
        help="Path to input image"
    )

    parser.add_argument(
        "--task",
        "-t",
        default=None,
        help='Task name or ID'
    )

    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output image path"
    )

    parser.add_argument(
        "--threshold",
        default=0.70,
        type=float,
        help="Relative threshold"
    )

    parser.add_argument(
        "--top_k",
        default=5,
        type=int,
        help="Max candidates"
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show all detections"
    )

    parser.add_argument(
        "--list_tasks",
        action="store_true",
        help="List all tasks"
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # SHOW TASK LIST
    # --------------------------------------------------------

    if args.list_tasks:

        print_task_list()

        sys.exit(0)

    # --------------------------------------------------------
    # VALIDATE INPUTS
    # --------------------------------------------------------

    if not args.image or not args.task:

        parser.print_help()

        sys.exit(1)

    if not os.path.exists(args.image):

        print(f"[Pipeline] Image not found: {args.image}")

        sys.exit(1)

    # --------------------------------------------------------
    # DEFAULT OUTPUT NAME
    # --------------------------------------------------------

    if args.output is None:

     stem = Path(args.image).stem

    safe_task = args.task.replace(" ", "_")

    output_folder = "outputs"

    os.makedirs(output_folder, exist_ok=True)

    args.output = os.path.join(
        output_folder,
        f"output_{stem}_{safe_task}.jpg"
    )

    # --------------------------------------------------------
    # RUN PIPELINE
    # --------------------------------------------------------

    run_pipeline(
        image_path=args.image,
        task_input=args.task,
        output_path=args.output,
        threshold=args.threshold,
        top_k=args.top_k,
        verbose=args.verbose
    )


if __name__ == "__main__":

    main()
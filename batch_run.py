#!/usr/bin/env python3
"""
batch_run.py
============
Run all 14 tasks at once. Useful for generating results screenshots for the report.

Usage:
    python batch_run.py --images_dir ./test_images/

Expects files named:  task1.jpg, task2.jpg ... task14.jpg
OR:                   step_on_something.jpg, sit_comfortably.jpg, etc.
OR any images in the folder — it will try all 14 tasks.

Example folder structure:
    test_images/
        task1.jpg     <- image with skateboard, chair etc.
        task2.jpg     <- image with sofa, chair
        task3.jpg     <- image with vase, flowers
        ...
        task14.jpg    <- image with baseball bat, tennis racket

Output:
    outputs/
        result_task01_step_on_something.jpg
        result_task02_sit_comfortably.jpg
        ...
        summary.txt
"""

import os
import sys
import glob
import time
import argparse
from pathlib import Path

from task_config import TASK_DEFINITIONS
from detector   import load_model
from scorer     import run_task_aware_selection
from visualizer import draw_results
import cv2


def find_image_for_task(images_dir: str, task_id: int, task_name: str) -> str | None:
    """
    Try to find the best image for a task.
    Looks for: task{id}.jpg, task{id:02d}.jpg, or any jpg in folder.
    """
    patterns = [
        f"task{task_id}.jpg", f"task{task_id}.png",
        f"task{task_id:02d}.jpg", f"task{task_id:02d}.png",
        f"{task_name.replace(' ', '_')}.jpg",
        f"{task_name.replace(' ', '_')}.png",
    ]
    for pat in patterns:
        p = os.path.join(images_dir, pat)
        if os.path.exists(p):
            return p
    return None


def run_batch(images_dir: str, output_dir: str,
              threshold: float = 0.70, top_k: int = 3):

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 65)
    print("  SpecOps 199 — DVCon India 2026 Stage 2A — Batch Runner")
    print("  Detector: EfficientDet-D0 + BiFPN (EfficientNet-B0 backbone)")
    print("=" * 65)
    print()

    # Load model once
    model = load_model()

    # Get all images in directory
    all_images = sorted(
        glob.glob(os.path.join(images_dir, "*.jpg")) +
        glob.glob(os.path.join(images_dir, "*.png")) +
        glob.glob(os.path.join(images_dir, "*.jpeg"))
    )

    if not all_images:
        print(f"No images found in {images_dir}")
        print("Put your test images there and try again.")
        sys.exit(1)

    print(f"Found {len(all_images)} image(s) in {images_dir}")
    print()

    results_summary = []
    t_total = time.time()

    # If images match tasks by name, run paired; otherwise run each image × all tasks
    paired = {}
    for tid in range(1, 15):
        tname = TASK_DEFINITIONS[tid]["name"]
        img   = find_image_for_task(images_dir, tid, tname)
        if img:
            paired[tid] = img

    if paired:
        print(f"Matched {len(paired)} task-image pairs by filename.\n")
        tasks_to_run = [(tid, TASK_DEFINITIONS[tid]["name"], paired[tid])
                        for tid in sorted(paired.keys())]
    else:
        # One image per task — cycle through images
        print("No task-named files found. Running all tasks on available images.\n")
        tasks_to_run = []
        for i, tid in enumerate(range(1, 15)):
            img = all_images[i % len(all_images)]
            tasks_to_run.append((tid, TASK_DEFINITIONS[tid]["name"], img))

    for task_id, task_name, image_path in tasks_to_run:
        print(f"[{task_id:2d}/14] Task: '{task_name}'  Image: {Path(image_path).name}")

        import cv2 as _cv
        img_bgr = _cv.imread(image_path)
        if img_bgr is None:
            print(f"       SKIP: cannot read image.")
            continue

        # Detect
        from detector import detect
        detections = detect(model, img_bgr)

        # Score
        selected, _, _ = run_task_aware_selection(
            detections, task_name,
            threshold=threshold,
            top_k=top_k
        )

        # Result
        if selected and selected[0]["relevance_score"] > 0:
            top  = selected[0]
            ans  = top["class_name"].upper()
            rel  = top["relevance_score"]
            conf = top["confidence"]
            print(f"       ✓ Answer: {ans}  (relevance={rel:.3f}, conf={conf:.3f})")
            status = "FOUND"
        else:
            ans, rel, conf = "NONE", 0.0, 0.0
            print(f"       ✗ No relevant object found")
            status = "NONE"

        # Save output image
        out_name = f"result_task{task_id:02d}_{task_name.replace(' ', '_')}.jpg"
        out_path = os.path.join(output_dir, out_name)
        vis = draw_results(img_bgr, detections, selected, task_name, task_id)
        _cv.imwrite(out_path, vis)
        print(f"       Saved: {out_path}\n")

        results_summary.append({
            "task_id": task_id,
            "task_name": task_name,
            "image": Path(image_path).name,
            "answer": ans,
            "relevance": rel,
            "confidence": conf,
            "status": status,
            "output": out_path,
        })

    total_time = time.time() - t_total

    # Save summary
    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("SpecOps 199 — DVCon India 2026 Stage 2A — Results Summary\n")
        f.write("=" * 65 + "\n\n")
        for r in results_summary:
            f.write(f"Task {r['task_id']:2d}: {r['task_name']:30s} → {r['answer']}\n")
        found = sum(1 for r in results_summary if r["status"] == "FOUND")
        f.write(f"\nDetected: {found}/{len(results_summary)}\n")
        f.write(f"Total time: {total_time:.1f}s\n")

    print("=" * 65)
    found = sum(1 for r in results_summary if r["status"] == "FOUND")
    print(f"  SUMMARY: {found}/{len(results_summary)} tasks answered")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Results saved to: {output_dir}/")
    print(f"  Summary: {summary_path}")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(
        description="Batch-run all 14 tasks — SpecOps 199 DVCon Stage 2A"
    )
    parser.add_argument("--images_dir", "-d", default="./test_images",
                        help="Folder with input images (default: ./test_images)")
    parser.add_argument("--output_dir", "-o", default="./outputs",
                        help="Folder to save results (default: ./outputs)")
    parser.add_argument("--threshold",  default=0.70, type=float,
                        help="Relative threshold (default 0.70)")
    parser.add_argument("--top_k",     default=3, type=int,
                        help="Max candidates (default 3)")
    args = parser.parse_args()
    run_batch(args.images_dir, args.output_dir, args.threshold, args.top_k)


if __name__ == "__main__":
    main()

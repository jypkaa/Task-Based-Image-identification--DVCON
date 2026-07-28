"""
Main Evaluation Runner — SpecOps 199 | DVCon India 2026 Stage 2A
=================================================================
Runs ALL 14 contest tasks on MULTIPLE images.

Usage:
  # Full evaluation — all 14 tasks × all images in folder:
  python main.py --images_dir ./test_images

  # Single image, all 14 tasks:
  python main.py --image ./test_images/kitchen.jpg

  # Single image, single task:
  python main.py --image ./test_images/kitchen.jpg --task "cook"

  # Quick demo (downloads 5 sample COCO images automatically):
  python main.py --demo

Output:
  outputs/<image>__<task>.jpg   — annotated result per image×task
  outputs/evaluation_results.json  — full JSON results
  outputs/summary_table.txt        — readable summary table
"""

import os
import json
import argparse
import urllib.request
from pathlib import Path
from model_setup import load_model, print_model_info
from pipeline import run_pipeline

# ── 13 contest evaluation queries (Tasks 1–13 from proposal) ──
# Task 0 (general) excluded from contest evaluation
EVAL_QUERIES = {
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

# ── Sample images covering multiple tasks ──
DEMO_IMAGES = {
    "kitchen.jpg":    "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6d/Good_Food_Display_-_NCI_Visuals_Online.jpg/640px-Good_Food_Display_-_NCI_Visuals_Online.jpg",
    "dining.jpg":     "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Spaghetti_bolognese_%28hozinja%29.jpg/640px-Spaghetti_bolognese_%28hozinja%29.jpg",
    "street.jpg":     "https://ultralytics.com/images/bus.jpg",
    "sports.jpg":     "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1d/Football_in_Bloomington%2C_Indiana%2C_1996.jpg/640px-Football_in_Bloomington%2C_Indiana%2C_1996.jpg",
    "office.jpg":     "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/640px-PNG_transparency_demonstration_1.png",
}


# ══════════════════════════════════════════════════════════════
# CORE EVALUATION FUNCTIONS
# ══════════════════════════════════════════════════════════════

def evaluate_image_all_tasks(image_path: str, model,
                              save_output: bool = True) -> list:
    """
    Run ALL 14 tasks on a single image.
    Returns list of result dicts (one per task).
    """
    results = []
    img_name = os.path.basename(image_path)

    print(f"\n{'═'*60}")
    print(f"  IMAGE: {img_name}")
    print(f"  Running all 14 evaluation queries...")
    print(f"{'═'*60}")

    for task_id, task_query in EVAL_QUERIES.items():
        result = run_pipeline(
            image_path=image_path,
            task_query=task_query,
            model=model,
            conf_threshold=0.25,
            threshold_ratio=0.80,
            save_output=save_output
        )
        result["task_id"] = task_id
        results.append(result)

    return results


def evaluate_all_images(images_dir: str, model) -> dict:
    """
    Run all 14 tasks on every image in images_dir.
    Saves results to outputs/evaluation_results.json and summary table.
    """
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths = sorted([
        str(p) for p in Path(images_dir).iterdir()
        if p.suffix.lower() in exts
    ])

    if not image_paths:
        print(f"  No images found in '{images_dir}'")
        print(f"  Put your test images there and re-run.")
        return {}

    print(f"\n  Found {len(image_paths)} image(s) × 14 tasks"
          f" = {len(image_paths)*14} pipeline runs\n")

    all_results = {}
    for img_path in image_paths:
        img_results = evaluate_image_all_tasks(img_path, model, save_output=True)
        all_results[os.path.basename(img_path)] = img_results

    # ── Save JSON ──────────────────────────────────────────
    os.makedirs("outputs", exist_ok=True)
    json_path = "outputs/evaluation_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved → {json_path}")

    # ── Save summary table ─────────────────────────────────
    table_path = _save_summary_table(all_results)
    print(f"  Summary saved → {table_path}")

    # ── Print summary ──────────────────────────────────────
    _print_summary(all_results)

    return all_results


# ══════════════════════════════════════════════════════════════
# REPORTING
# ══════════════════════════════════════════════════════════════

def _print_summary(all_results: dict):
    """Prints a compact summary table to console."""
    print("\n" + "="*75)
    print("  EVALUATION SUMMARY — All Images × All 14 Tasks")
    print("="*75)
    header = f"{'Image':<22} {'Task':<18} {'Primary Object':<20} {'Relevance':<10} Status"
    print(header)
    print("-"*75)

    for img_name, results in all_results.items():
        for r in results:
            task    = r.get("task", "")
            primary = r.get("primary_object") or "—"
            rel     = ""
            if r.get("selected_objects"):
                rel = f"{r['selected_objects'][0]['relevance']:.4f}"
            status  = "✓" if r.get("status") == "success" else "✗"
            print(f"  {img_name:<20} {task:<18} {primary:<20} {rel:<10} {status}")
    print("="*75)


def _save_summary_table(all_results: dict) -> str:
    """Saves summary table as plain text file (useful for the 2-page report)."""
    os.makedirs("outputs", exist_ok=True)
    path = "outputs/summary_table.txt"
    lines = []
    lines.append("SpecOps 199 — Stage 2A Evaluation Summary")
    lines.append("DVCon India 2026 | NIT Warangal")
    lines.append("="*75)
    lines.append(f"{'Image':<22} {'Task':<18} {'Primary Object':<20} {'Relevance':<10} Status")
    lines.append("-"*75)

    for img_name, results in all_results.items():
        for r in results:
            task    = r.get("task", "")
            primary = r.get("primary_object") or "none"
            rel     = ""
            if r.get("selected_objects"):
                rel = f"{r['selected_objects'][0]['relevance']:.4f}"
            status  = "SUCCESS" if r.get("status") == "success" else "NO MATCH"
            lines.append(f"{img_name:<22} {task:<18} {primary:<20} {rel:<10} {status}")

    lines.append("="*75)
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path


# ══════════════════════════════════════════════════════════════
# DEMO MODE
# Downloads sample images covering all major task categories
# ══════════════════════════════════════════════════════════════

def run_demo(model):
    """
    Downloads 5 sample images and runs all 14 tasks on each.
    Good for testing the pipeline and generating results for the report.
    """
    os.makedirs("test_images", exist_ok=True)
    print("\n  Downloading sample images for demo...")

    downloaded = []
    for fname, url in DEMO_IMAGES.items():
        dest = f"test_images/{fname}"
        if not os.path.exists(dest):
            try:
                print(f"  → {fname} ...")
                urllib.request.urlretrieve(url, dest)
            except Exception as e:
                print(f"    Failed: {e}")
                continue
        downloaded.append(dest)

    if not downloaded:
        print("  No images downloaded. Check internet connection.")
        return

    print(f"\n  Running evaluation on {len(downloaded)} image(s)...")
    all_results = {}
    for img_path in downloaded:
        img_results = evaluate_image_all_tasks(img_path, model, save_output=True)
        all_results[os.path.basename(img_path)] = img_results

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/demo_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    _save_summary_table(all_results)
    _print_summary(all_results)

    print(f"\n  Demo complete!")
    print(f"  • Annotated images → outputs/")
    print(f"  • JSON results     → outputs/demo_results.json")
    print(f"  • Summary table    → outputs/summary_table.txt")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="SpecOps 199 — Task-Oriented Object Identification | Stage 2A"
    )
    parser.add_argument("--images_dir", type=str,
                        help="Folder of test images — runs all 14 tasks on each")
    parser.add_argument("--image", type=str,
                        help="Single image path")
    parser.add_argument("--task", type=str,
                        help="Single task query (use with --image)")
    parser.add_argument("--demo", action="store_true",
                        help="Download sample images and run full demo")
    parser.add_argument("--variant", type=str, default="tf_efficientdet_d0",
                        help="EfficientDet variant (default: tf_efficientdet_d0)")
    parser.add_argument("--no_onnx", action="store_true",
                        help="Use PyTorch backend instead of ONNX")
    args = parser.parse_args()

    # ── Load EfficientDet (EfficientNet backbone) ──
    model = load_model(
        variant=args.variant,
        use_onnx=not args.no_onnx
    )
    print_model_info(model)

    # ── Run selected mode ──
    if args.demo:
        run_demo(model)

    elif args.images_dir:
        # Full evaluation: all images × all 14 tasks
        evaluate_all_images(args.images_dir, model)

    elif args.image and args.task:
        # Single image, single task
        result = run_pipeline(args.image, args.task, model)
        print(json.dumps(result, indent=2))

    elif args.image:
        # Single image, all 14 tasks
        results = evaluate_image_all_tasks(args.image, model)
        os.makedirs("outputs", exist_ok=True)
        out_json = f"outputs/{Path(args.image).stem}_all_tasks.json"
        with open(out_json, "w") as f:
            json.dump(results, f, indent=2)
        _print_summary({Path(args.image).name: results})
        print(f"\n  Results saved → {out_json}")

    else:
        print("\n  No arguments provided. Running demo mode...")
        run_demo(model)


if __name__ == "__main__":
    main()

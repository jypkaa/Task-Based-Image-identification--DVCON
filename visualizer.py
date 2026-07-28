"""
visualizer.py
=============

Enhanced visualization for task-aware object detection.

Features:
- GOLD  box  = best object
- GREEN box  = other valid objects
- GRAY  box  = detected but irrelevant

Also shows:
- relevance score
- confidence
- task banner
- legend
"""

import cv2
import numpy as np


# ============================================================
# COLORS
# ============================================================

GOLD = (0, 200, 255)

GREEN = (0, 220, 100)

GRAY = (130, 130, 130)

DARK = (20, 20, 20)

WHITE = (240, 240, 240)


# ============================================================
# DRAW RESULTS
# ============================================================

def draw_results(
    img: np.ndarray,
    detections: list,
    selected: list,
    task_name: str,
    task_id: int
) -> np.ndarray:

    """
    Draw final annotated output image.
    """

    out = img.copy()

    h, w = out.shape[:2]

    # --------------------------------------------------------
    # HANDLE EMPTY SELECTIONS
    # --------------------------------------------------------

    if selected:

        selected_names = [
            d["class_name"]
            for d in selected
        ]

        top_name = selected[0]["class_name"]

    else:

        selected_names = []

        top_name = None

    # --------------------------------------------------------
    # DRAW ALL DETECTIONS
    # --------------------------------------------------------

    for det in detections:

        name = det["class_name"]

        conf = det["confidence"]

        rel = det.get(
            "relevance_score",
            0.0
        )

        x1, y1, x2, y2 = [
            int(v)
            for v in det["bbox"]
        ]

        # safety clipping
        x1 = max(0, x1)
        y1 = max(0, y1)

        x2 = min(w - 1, x2)
        y2 = min(h - 1, y2)

        # ----------------------------------------------------
        # COLOR SELECTION
        # ----------------------------------------------------

        if name == top_name:

            color = GOLD

            thickness = 4

            tag = "BEST"

        elif name in selected_names:

            color = GREEN

            thickness = 3

            tag = "VALID"

        else:

            color = GRAY

            thickness = 1

            tag = ""

        # ----------------------------------------------------
        # DRAW BOUNDING BOX
        # ----------------------------------------------------

        cv2.rectangle(
            out,
            (x1, y1),
            (x2, y2),
            color,
            thickness
        )

        # ----------------------------------------------------
        # LABEL TEXT
        # ----------------------------------------------------

        if tag:

            label = (
                f"{tag} | "
                f"{name} | "
                f"rel={rel:.4f} | "
                f"conf={conf:.2f}"
            )

        else:

            label = (
                f"{name} "
                f"{conf:.2f}"
            )

        font = cv2.FONT_HERSHEY_SIMPLEX

        fs = 0.55

        (tw, th), baseline = cv2.getTextSize(
            label,
            font,
            fs,
            1
        )

        ty = max(y1 - 8, th + 6)

        # label background
        cv2.rectangle(
            out,
            (x1, ty - th - 6),
            (x1 + tw + 8, ty + baseline),
            DARK,
            -1
        )

        # label text
        cv2.putText(
            out,
            label,
            (x1 + 4, ty),
            font,
            fs,
            color,
            1,
            cv2.LINE_AA
        )

    # ========================================================
    # TOP BANNER
    # ========================================================

    banner_h = 60

    banner = np.zeros(
        (banner_h, w, 3),
        dtype=np.uint8
    )

    banner[:] = (28, 28, 28)

    # task text
    cv2.putText(
        banner,
        f"Task [{task_id}] : {task_name.upper()}",
        (12, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (220, 220, 255),
        1,
        cv2.LINE_AA
    )

    # best object
    if top_name:

        best_text = f"Best Object : {top_name}"

        best_color = GOLD

    else:

        best_text = "No relevant object found"

        best_color = (180, 180, 180)

    cv2.putText(
        banner,
        best_text,
        (12, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.80,
        best_color,
        2,
        cv2.LINE_AA
    )

    # combine banner + image
    out = np.vstack([banner, out])

    # ========================================================
    # LEGEND
    # ========================================================

    lx = w - 260

    ly = 70

    legend_items = [

        (GOLD,  "Best object"),

        (GREEN, "Other valid objects"),

        (GRAY,  "Not relevant")
    ]

    for color, text in legend_items:

        cv2.rectangle(
            out,
            (lx, ly),
            (lx + 16, ly + 14),
            color,
            -1
        )

        cv2.putText(
            out,
            text,
            (lx + 24, ly + 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            WHITE,
            1,
            cv2.LINE_AA
        )

        ly += 22

    
    return out
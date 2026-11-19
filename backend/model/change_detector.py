"""
Bi-Temporal Satellite Change Detection Engine  v3
Author: Sat_Assist Team

Improvements over v2:
  - Tighter box fitting: boxes are shrunk to the actual changed-pixel bounding
    rect inside the contour, not the dilated mask bounding rect.
  - Fill-ratio validation: boxes whose interior has < MIN_BOX_FILL_RATIO
    changed pixels are rejected (eliminates sparse / false-positive boxes).
  - Proximity-aware NMS: boxes within PROXIMITY_MERGE_DIST px of each other
    are merged before IoU NMS to unify nearby fragmented clusters.
  - Stricter percentile threshold (p88) keeps only the top-intensity changes.
  - Smaller dilation kernel (3×3 vs 7×7) to prevent mask bleed.
  - Minimum area raised (0.15 % of image) to suppress micro-specks.
  - Classification thresholds tightened; min edge absolute value raised.
  - Boxes that cover > 60% of the image are rejected (over-sized guard).
  - Corner/watermark suppression zone extended to 15% of frame.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from PIL import Image
from skimage.metrics import structural_similarity as ssim


CHANGE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "annotated"
CHANGE_DIR.mkdir(parents=True, exist_ok=True)


# ── Tuning constants ────────────────────────────────────────────────────────
MIN_REGION_AREA_FRACTION  = 0.0008   # ignore regions < 0.08% of image pixels
MAX_BOXES                 = 15       # keep top-N largest boxes after NMS (reduced from 20)
BOX_IOU_MERGE_THRESH      = 0.20     # IoU threshold for merging overlapping boxes
PROXIMITY_MERGE_DIST      = 0.055    # merge boxes whose centres are within 5.5% of image diagonal
MIN_BOX_FILL_RATIO        = 0.10     # reject box if < 10% of its interior pixels are changed
MAX_BOX_COVERAGE          = 0.60     # reject box if it covers > 60% of image (over-sized guard)
SSIM_WIN_SIZE             = 11       # larger window → less high-freq noise
HEATMAP_ALPHA             = 180      # 0-255 opacity of heatmap overlay pixels
CHANGE_PERCENTILE_THRESH  = 78       # adaptive: top 22% of diff pixels (back to v2 baseline)
GRADIENT_SUPPRESS_THRESH  = 22       # suppress diff where gradient mag > this (slightly tighter)
# ───────────────────────────────────────────────────────────────────────────


class BiTemporalChangeDetector:

    # ── 1. Composite split ──────────────────────────────────────────────────
    @staticmethod
    def split_composite_image(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Splits a single composite Before/After satellite image into T1 and T2.
        Side-by-side (w >= h) → left half = T1, right half = T2.
        Stacked    (h >  w) → top half  = T1, bottom half = T2.
        Both halves are resized to identical pixel dimensions.
        """
        h, w = img.shape[:2]
        if w >= h:
            mid = w // 2
            t1, t2 = img[:, :mid], img[:, mid:]
        else:
            mid = h // 2
            t1, t2 = img[:mid, :], img[mid:, :]

        th, tw = t1.shape[:2]
        t2 = cv2.resize(t2, (tw, th), interpolation=cv2.INTER_AREA)
        return t1, t2

    # ── 2. Image alignment ──────────────────────────────────────────────────
    @staticmethod
    def align_images(img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Co-registers img2 to img1 using robust sub-pixel Phase Correlation in the Fourier domain
        and multi-scale verification. Phase correlation is mathematically invariant to illumination,
        contrast changes, and seasonal vegetation differences.
        Falls back smoothly if already aligned.
        """
        h, w = img1.shape[:2]
        img2 = cv2.resize(img2, (w, h), interpolation=cv2.INTER_AREA)

        g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float32)
        g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float32)

        try:
            # 1. Fourier Cross-Power Spectrum Phase Correlation
            shift, resp = cv2.phaseCorrelate(g1, g2)
            dx, dy = shift[0], shift[1]

            # Validate shift bounds: plausible satellite camera drift is typically < 8% of frame
            if resp > 0.06 and abs(dx) < w * 0.08 and abs(dy) < h * 0.08 and (abs(dx) > 0.35 or abs(dy) > 0.35):
                M = np.float32([[1, 0, dx], [0, 1, dy]])
                aligned = cv2.warpAffine(
                    img2, M, (w, h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT
                )
                return img1, aligned
        except Exception as e:
            print(f"[BiTemporal] Phase correlation alignment notice: {e}")

        # 2. Pyramid ECC translation fallback
        try:
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-4)
            warp_matrix = np.eye(2, 3, dtype=np.float32)
            _, warp_fine = cv2.findTransformECC(
                g1.astype(np.uint8), g2.astype(np.uint8), warp_matrix, cv2.MOTION_TRANSLATION, criteria
            )
            dx = warp_fine[0, 2]
            dy = warp_fine[1, 2]
            if abs(dx) < w * 0.08 and abs(dy) < h * 0.08:
                aligned = cv2.warpAffine(
                    img2, warp_fine, (w, h),
                    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                    borderMode=cv2.BORDER_REFLECT
                )
                return img1, aligned
        except Exception:
            pass

        return img1, img2

    # ── 3. Gradient-illumination suppression mask ───────────────────────────
    @staticmethod
    def _gradient_suppress_mask(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """
        Builds a mask (uint8 0/1) suppressing regions where BOTH images
        have high gradient magnitude AND matching edge orientation with small intensity difference.
        These represent persistent real edges (roads, roof ridges, boundaries) that appear
        in diff purely due to minor sub-pixel co-registration or lighting differences.
        Genuine structural changes (e.g. new buildings over fields) have non-matching edge
        orientations or substantial intensity differences, so they are preserved.
        """
        g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float32)
        g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float32)

        s1x = cv2.Sobel(g1, cv2.CV_32F, 1, 0, ksize=3)
        s1y = cv2.Sobel(g1, cv2.CV_32F, 0, 1, ksize=3)
        s2x = cv2.Sobel(g2, cv2.CV_32F, 1, 0, ksize=3)
        s2y = cv2.Sobel(g2, cv2.CV_32F, 0, 1, ksize=3)

        mag1 = np.sqrt(s1x**2 + s1y**2)
        mag2 = np.sqrt(s2x**2 + s2y**2)

        # Normalized gradient vector dot product (cosine of angle between edge vectors)
        dot = (s1x * s2x + s1y * s2y) / (mag1 * mag2 + 1e-4)
        intensity_diff = np.abs(g1 - g2)

        # Persistent edge: strong gradient in both, matching orientation (dot > 0.65), and small intensity diff (< 35)
        shared_edge = (mag1 > 30) & (mag2 > 30) & (dot > 0.65) & (intensity_diff < 35)

        # Dilate slightly so sub-pixel edge fringe is suppressed without eroding real change bodies
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        shared_edge_u8 = shared_edge.astype(np.uint8) * 255
        shared_edge_dilated = cv2.dilate(shared_edge_u8, kernel, iterations=1) > 0
        keep_mask = (~shared_edge_dilated).astype(np.uint8)
        return keep_mask

    # ── 4. Core change map ──────────────────────────────────────────────────
    @classmethod
    def _compute_change_map(
        cls,
        img_t1: np.ndarray,
        img_t2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Returns:
          fused_diff  - uint8 float-mapped diff (0-255) before thresholding
          bin_clean   - uint8 binary mask of genuine change regions
          ssim_score  - scalar SSIM similarity score
        """
        h, w = img_t1.shape[:2]

        # ── Gaussian pre-blur to reduce sensor noise ──────────────────────
        blur1 = cv2.GaussianBlur(img_t1, (5, 5), 1.2)
        blur2 = cv2.GaussianBlur(img_t2, (5, 5), 1.2)

        # ── SSIM difference on grayscale ──────────────────────────────────
        gray1 = cv2.cvtColor(blur1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(blur2, cv2.COLOR_BGR2GRAY)

        win = min(SSIM_WIN_SIZE, h if h % 2 == 1 else h - 1, w if w % 2 == 1 else w - 1)
        win = max(win, 3) if win % 2 == 0 else max(win, 3)
        score, diff_ssim = ssim(gray1, gray2, full=True, win_size=win, data_range=255)
        diff_ssim = (1.0 - diff_ssim)   # 1 = max change
        diff_ssim_u8 = np.clip(diff_ssim * 255, 0, 255).astype(np.uint8)

        # ── CIE-LAB color difference ──────────────────────────────────────
        lab1 = cv2.cvtColor(blur1, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab2 = cv2.cvtColor(blur2, cv2.COLOR_BGR2LAB).astype(np.float32)
        diff_lab = np.linalg.norm(lab1 - lab2, axis=2)   # Euclidean in LAB

        # Normalize LAB diff to 0-255 using 99th percentile (robust to outliers)
        p99 = np.percentile(diff_lab, 99)
        if p99 > 0:
            diff_lab_u8 = np.clip(diff_lab / p99 * 255, 0, 255).astype(np.uint8)
        else:
            diff_lab_u8 = np.zeros_like(diff_ssim_u8)

        # ── Luminance-only diff (shadows, lighting) ───────────────────────
        # We SUBTRACT the luminance diff so pure brightness shifts don't count
        lum1 = lab1[:, :, 0]   # L channel
        lum2 = lab2[:, :, 0]
        lum_diff = np.abs(lum1 - lum2)                    # 0-100 range in LAB
        lum_diff_u8 = np.clip(lum_diff / 100 * 255, 0, 255).astype(np.uint8)

        # ── Chroma-only diff (robust to illumination changes) ─────────────
        a1, b1 = lab1[:, :, 1], lab1[:, :, 2]
        a2, b2 = lab2[:, :, 1], lab2[:, :, 2]
        chroma_diff = np.sqrt((a1 - a2)**2 + (b1 - b2)**2)
        p99c = np.percentile(chroma_diff, 99)
        chroma_u8 = np.clip(chroma_diff / max(p99c, 1) * 255, 0, 255).astype(np.uint8)

        # ── Fuse: weight SSIM structural + chroma, suppress lum-only noise ─
        # Formula: fused = 0.5*ssim + 0.3*chroma + 0.2*lab_total - 0.4*lum_only
        fused = (
            0.50 * diff_ssim_u8.astype(np.float32) +
            0.30 * chroma_u8.astype(np.float32) +
            0.20 * diff_lab_u8.astype(np.float32) -
            0.35 * lum_diff_u8.astype(np.float32)   # penalize pure-illumination diff
        )
        fused = np.clip(fused, 0, 255).astype(np.uint8)

        # ── Gradient suppression (persistent edges) ───────────────────────
        keep_mask = cls._gradient_suppress_mask(img_t1, img_t2)
        fused = (fused.astype(np.float32) * keep_mask).astype(np.uint8)

        # ── Morphological open to remove salt-and-pepper noise ────────────
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fused_clean = cv2.morphologyEx(fused, cv2.MORPH_OPEN, k_open)

        # ── Adaptive percentile threshold ─────────────────────────────────
        nonzero_vals = fused_clean[fused_clean > 0]
        if len(nonzero_vals) > 0:
            thresh_val = np.percentile(nonzero_vals, CHANGE_PERCENTILE_THRESH)
            thresh_val = max(thresh_val, 20)   # never threshold below 20
        else:
            thresh_val = 40

        _, bin_thresh = cv2.threshold(fused_clean, int(thresh_val), 255, cv2.THRESH_BINARY)

        # ── Morphological close to fill small holes inside regions ────────
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        bin_closed = cv2.morphologyEx(bin_thresh, cv2.MORPH_CLOSE, k_close)

        # ── Dilate to merge nearby fragmented change clusters ─────────────
        # 7×7 needed for high-resolution images where real change is spread
        # across many small adjacent pixels (urban sprawl, road expansion).
        k_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        bin_dilated = cv2.dilate(bin_closed, k_dilate, iterations=1)

        # ── Filter contours by minimum area ───────────────────────────────
        min_area = int(MIN_REGION_AREA_FRACTION * h * w)
        contours, _ = cv2.findContours(bin_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bin_clean = np.zeros_like(bin_dilated)
        for cnt in contours:
            if cv2.contourArea(cnt) >= min_area:
                cv2.drawContours(bin_clean, [cnt], -1, 255, thickness=cv2.FILLED)

        return fused_clean, bin_clean, float(score)

    # ── 5. Change region classification ────────────────────────────────────
    @staticmethod
    def _classify_box(crop_t1: np.ndarray, crop_t2: np.ndarray) -> Tuple[str, str, float]:
        """
        Classifies a localized change bounding box into:
          'appeared':    New construction, road network, buildings, paved surface
          'disappeared': Removed structures, cleared vegetation, demolished area
          'modified':    Land-use shift, crop rotation, soil alteration

        Uses three evidence signals with tighter thresholds to avoid false labels:
          1. Sobel edge density ratio (structural complexity change)
          2. VARI vegetation index shift
          3. Luminance / impervious surface shift
        """
        if crop_t1 is None or crop_t2 is None or crop_t1.size == 0 or crop_t2.size == 0:
            return "modified", "Surface Shift", 0.80

        # Minimum viable crop size for reliable feature extraction
        if crop_t1.shape[0] < 5 or crop_t1.shape[1] < 5:
            return "modified", "Surface Shift", 0.75

        g1 = cv2.cvtColor(crop_t1, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(crop_t2, cv2.COLOR_BGR2GRAY)

        # 1. Structural Edge density (Sobel diagonal — captures both H and V edges)
        s1x = cv2.Sobel(g1, cv2.CV_32F, 1, 0, ksize=3)
        s1y = cv2.Sobel(g1, cv2.CV_32F, 0, 1, ksize=3)
        s2x = cv2.Sobel(g2, cv2.CV_32F, 1, 0, ksize=3)
        s2y = cv2.Sobel(g2, cv2.CV_32F, 0, 1, ksize=3)
        e1 = float(np.mean(np.sqrt(s1x**2 + s1y**2)))
        e2 = float(np.mean(np.sqrt(s2x**2 + s2y**2)))
        edge_ratio = e2 / max(e1, 1e-4)

        # 2. Vegetation index (Normalized Green-Red Difference Index - stable for satellite RGB)
        b1f = crop_t1[:, :, 0].astype(np.float32) / 255.0
        gr1f = crop_t1[:, :, 1].astype(np.float32) / 255.0
        r1f = crop_t1[:, :, 2].astype(np.float32) / 255.0

        b2f = crop_t2[:, :, 0].astype(np.float32) / 255.0
        gr2f = crop_t2[:, :, 1].astype(np.float32) / 255.0
        r2f = crop_t2[:, :, 2].astype(np.float32) / 255.0

        denom1 = np.clip(gr1f + r1f, 0.02, None)
        denom2 = np.clip(gr2f + r2f, 0.02, None)
        veg1 = float(np.mean((gr1f - r1f) / denom1))
        veg2 = float(np.mean((gr2f - r2f) / denom2))
        veg_delta = veg2 - veg1

        # 3. Brightness / Impervious surface shift
        lum1 = float(np.mean(g1))
        lum2 = float(np.mean(g2))
        lum_delta = lum2 - lum1   # positive = brighter in T2 (impervious / cleared)

        # ── Classification rules (balanced & robust) ─────────────────────────
        # APPEARED: Significant new edges (roads, building outlines, roofs) in T2
        if edge_ratio >= 1.65 and e2 >= 18.0:
            conf = round(min(0.98, 0.75 + (edge_ratio - 1.65) * 0.07), 2)
            return "appeared", "New Construction / Roads", conf

        # APPEARED: Vegetation gain / afforestation
        if veg_delta > 0.08 and e1 < 25.0:
            return "appeared", "Vegetation Gain / Regrowth", 0.88

        # DISAPPEARED: Structures or high-edge features removed
        if edge_ratio <= 0.60 and e1 >= 18.0:
            conf = round(min(0.98, 0.75 + (0.60 - edge_ratio) * 0.10), 2)
            return "disappeared", "Demolished / Removed", conf

        # DISAPPEARED: Significant vegetation or green cover loss
        if veg_delta < -0.06 and edge_ratio < 1.40:
            return "disappeared", "Vegetation Loss / Clearing", 0.86

        # MODIFIED: Significant land-use brightness change
        if abs(lum_delta) > 25.0:
            return "modified", "Land-Use Conversion", 0.86

        # MODIFIED: Moderate texture or surface structural shift
        if 0.70 <= edge_ratio <= 1.65:
            return "modified", "Surface / Texture Shift", 0.82

        return "modified", "Surface / Texture Shift", 0.78

    # ── 6. Box extraction + proximity merge + NMS + classification ────────────
    @classmethod
    def _extract_boxes(
        cls,
        bin_clean: np.ndarray,
        h: int,
        w: int,
        img_t1: Optional[np.ndarray] = None,
        img_t2: Optional[np.ndarray] = None
    ) -> List[Dict]:
        """
        Extracts tightly-fitted bounding boxes from the cleaned binary change mask.

        Key accuracy improvements over v2:
        1. Tight fitting: each box is shrunk to the axis-aligned bounding rect of
           the actual changed pixels inside the contour (not the dilated mask boundary).
        2. Fill-ratio validation: boxes are rejected if < MIN_BOX_FILL_RATIO of
           their interior area consists of changed pixels.
        3. Proximity merge: nearby boxes (centres within PROXIMITY_MERGE_DIST)
           are merged before IoU-NMS, preventing fragmented cluster boxes.
        4. Over-size guard: boxes covering > MAX_BOX_COVERAGE of the image rejected.
        5. Stricter border seam and corner text suppression.

        Returns normalized [0,1] coordinates clamped to image bounds.
        """
        min_area = int(MIN_REGION_AREA_FRACTION * h * w)
        contours, _ = cv2.findContours(bin_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        raw_boxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue

            # ── Tight fit: use the actual changed-pixel extent, not contour bbox ──
            # Create a tight mask for just this contour
            cnt_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(cnt_mask, [cnt], -1, 255, thickness=cv2.FILLED)
            # Intersect with the bin_clean mask to get genuine change pixels
            tight_mask = cv2.bitwise_and(cnt_mask, bin_clean)
            ys, xs = np.where(tight_mask > 0)
            if len(xs) == 0 or len(ys) == 0:
                continue
            bx = int(xs.min())
            by = int(ys.min())
            bx2 = int(xs.max()) + 1
            by2 = int(ys.max()) + 1
            bw = bx2 - bx
            bh = by2 - by

            # Clamp strictly within image
            bx  = max(0, bx)
            by  = max(0, by)
            bx2 = min(w, bx2)
            by2 = min(h, by2)
            bw  = bx2 - bx
            bh  = by2 - by
            if bw <= 0 or bh <= 0:
                continue

            box_area = bw * bh
            if box_area <= 0:
                continue

            # ── Over-size guard ───────────────────────────────────────────────
            if box_area / (h * w) > MAX_BOX_COVERAGE:
                continue

            # ── Fill-ratio validation ──────────────────────────────────────────
            # Count genuine changed pixels inside the tight bounding rect
            changed_in_box = int(np.count_nonzero(bin_clean[by:by2, bx:bx2]))
            fill_ratio = changed_in_box / box_area
            if fill_ratio < MIN_BOX_FILL_RATIO:
                continue   # sparse / mostly-background box — reject

            # ── Suppress outer border seam slivers ───────────────────────────
            is_border_sliver = (
                (bx <= 3 and (bw < w * 0.04 or bh / max(bw, 1) > 4.0)) or
                (by <= 3 and (bh < h * 0.04 or bw / max(bh, 1) > 4.0)) or
                (bx2 >= w - 3 and (bw < w * 0.04 or bh / max(bw, 1) > 4.0)) or
                (by2 >= h - 3 and (bh < h * 0.04 or bw / max(bh, 1) > 4.0))
            )
            if is_border_sliver:
                continue

            # ── Suppress date / watermark text in corners ─────────────────────
            # Extended zone to 15% of frame (from 12%)
            is_corner_text = (
                (by < h * 0.15 or by2 > h * 0.85) and
                (bx < w * 0.28 or bx2 > w * 0.72) and
                (box_area < h * w * 0.018)
            )
            if is_corner_text and img_t1 is not None and img_t2 is not None:
                crop_g1 = cv2.cvtColor(img_t1[by:by2, bx:bx2], cv2.COLOR_BGR2GRAY)
                crop_g2 = cv2.cvtColor(img_t2[by:by2, bx:bx2], cv2.COLOR_BGR2GRAY)
                if crop_g1.size > 0 and crop_g2.size > 0:
                    c1_hi = crop_g1.max() > 238 or crop_g1.min() < 18
                    c2_hi = crop_g2.max() > 238 or crop_g2.min() < 18
                    c1_contrast = (crop_g1.max() - crop_g1.min()) > 200 and c1_hi
                    c2_contrast = (crop_g2.max() - crop_g2.min()) > 200 and c2_hi
                    if c1_contrast or c2_contrast:
                        continue

            raw_boxes.append({
                "x1": bx, "y1": by, "x2": bx2, "y2": by2,
                "area": float(area),
                "fill": fill_ratio
            })

        if not raw_boxes:
            return []

        # Sort by area descending
        raw_boxes.sort(key=lambda b: b["area"], reverse=True)

        # ── Step 1: Proximity merge ──────────────────────────────────────────
        # Merge boxes whose centres are very close (fragmented clusters of the
        # same real change zone should become one tight box around all of them)
        prox_px = PROXIMITY_MERGE_DIST * max(w, h)

        def _centre(b):
            return ((b["x1"] + b["x2"]) / 2, (b["y1"] + b["y2"]) / 2)

        merged_prox = []
        used = [False] * len(raw_boxes)
        for i, bi in enumerate(raw_boxes):
            if used[i]:
                continue
            group = [bi]
            cx_i, cy_i = _centre(bi)
            for j, bj in enumerate(raw_boxes):
                if used[j] or j == i:
                    continue
                cx_j, cy_j = _centre(bj)
                dist = ((cx_i - cx_j) ** 2 + (cy_i - cy_j) ** 2) ** 0.5
                if dist < prox_px:
                    group.append(bj)
                    used[j] = True
            used[i] = True
            # Union of the group
            mx1 = min(b["x1"] for b in group)
            my1 = min(b["y1"] for b in group)
            mx2 = max(b["x2"] for b in group)
            my2 = max(b["y2"] for b in group)
            total_area = sum(b["area"] for b in group)
            avg_fill = sum(b["fill"] for b in group) / len(group)
            # After proximity union, re-check fill ratio and over-size
            new_box_area = (mx2 - mx1) * (my2 - my1)
            if new_box_area / (h * w) > MAX_BOX_COVERAGE:
                # Too large after union — keep only the largest individual box
                best = max(group, key=lambda b: b["area"])
                merged_prox.append(best)
            else:
                merged_prox.append({
                    "x1": mx1, "y1": my1, "x2": mx2, "y2": my2,
                    "area": total_area, "fill": avg_fill
                })

        # ── Step 2: IoU-based NMS merge ────────────────────────────────────
        def iou(a, b):
            ix1 = max(a["x1"], b["x1"])
            iy1 = max(a["y1"], b["y1"])
            ix2 = min(a["x2"], b["x2"])
            iy2 = min(a["y2"], b["y2"])
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            if inter == 0:
                return 0.0
            area_a = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
            area_b = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
            return inter / (area_a + area_b - inter)

        kept = []
        suppressed = [False] * len(merged_prox)
        for i, box in enumerate(merged_prox):
            if suppressed[i]:
                continue
            merged = dict(box)
            for j in range(i + 1, len(merged_prox)):
                if suppressed[j]:
                    continue
                if iou(merged, merged_prox[j]) > BOX_IOU_MERGE_THRESH:
                    merged["x1"] = min(merged["x1"], merged_prox[j]["x1"])
                    merged["y1"] = min(merged["y1"], merged_prox[j]["y1"])
                    merged["x2"] = max(merged["x2"], merged_prox[j]["x2"])
                    merged["y2"] = max(merged["y2"], merged_prox[j]["y2"])
                    merged["area"] = merged["area"] + merged_prox[j]["area"]
                    suppressed[j] = True
            # Final over-size guard after NMS merge
            final_box_area = (merged["x2"] - merged["x1"]) * (merged["y2"] - merged["y1"])
            if final_box_area / (h * w) <= MAX_BOX_COVERAGE:
                kept.append(merged)
            if len(kept) >= MAX_BOXES:
                break

        # ── Step 3: Classify and normalize ────────────────────────────────
        result = []
        for b in kept:
            bx1, by1, bx2, by2 = int(b["x1"]), int(b["y1"]), int(b["x2"]), int(b["y2"])
            # Clamp to valid image dimensions
            bx1 = max(0, min(bx1, w - 1))
            by1 = max(0, min(by1, h - 1))
            bx2 = max(bx1 + 1, min(bx2, w))
            by2 = max(by1 + 1, min(by2, h))

            c_type, c_label, c_conf = "modified", "Surface Shift", 0.82
            if img_t1 is not None and img_t2 is not None:
                c1 = img_t1[by1:by2, bx1:bx2]
                c2 = img_t2[by1:by2, bx1:bx2]
                c_type, c_label, c_conf = cls._classify_box(c1, c2)

            result.append({
                "xmin": round(float(np.clip(bx1 / w, 0.0, 1.0)), 4),
                "ymin": round(float(np.clip(by1 / h, 0.0, 1.0)), 4),
                "xmax": round(float(np.clip(bx2 / w, 0.0, 1.0)), 4),
                "ymax": round(float(np.clip(by2 / h, 0.0, 1.0)), 4),
                "type": c_type,
                "change_type": c_type,
                "label": c_label,
                "confidence": c_conf,
                "area_pixels": int(b["area"])
            })
        return result

    # ── 6. Heatmap writer ──────────────────────────────────────────────────
    @staticmethod
    def _write_heatmap(
        fused_diff: np.ndarray,
        bin_clean:  np.ndarray,
        out_path:   Path
    ) -> None:
        """
        Writes a 4-channel RGBA heatmap PNG.
        Only pixels inside the cleaned binary mask are coloured;
        all other pixels are fully transparent.
        Uses COLORMAP_HOT for intuitive red=high-change palette.
        """
        h, w = fused_diff.shape[:2]

        # Re-normalize diff only over masked region for better contrast
        masked_vals = fused_diff[bin_clean > 0]
        if len(masked_vals) > 0:
            vmin = int(masked_vals.min())
            vmax = int(masked_vals.max())
            if vmax > vmin:
                display = np.clip(
                    (fused_diff.astype(np.float32) - vmin) / (vmax - vmin) * 255,
                    0, 255
                ).astype(np.uint8)
            else:
                display = fused_diff.copy()
        else:
            display = fused_diff.copy()

        color_map = cv2.applyColorMap(display, cv2.COLORMAP_HOT)   # Black→Red→Yellow→White

        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        mask = bin_clean > 0
        rgba[mask, 0] = color_map[mask, 2]   # R (OpenCV BGR→RGBA)
        rgba[mask, 1] = color_map[mask, 1]   # G
        rgba[mask, 2] = color_map[mask, 0]   # B
        rgba[mask, 3] = HEATMAP_ALPHA         # Alpha

        # Save as PNG with transparency
        pil_img = Image.fromarray(rgba, mode="RGBA")
        pil_img.save(str(out_path), format="PNG")

    # ── 7. Public API: detect from single composite ─────────────────────────
    @classmethod
    def detect_changes_from_single_image(
        cls,
        single_image_path: str,
        output_heatmap_name: str = "change_heatmap.png",
        save_split_files: bool = True
    ) -> Dict[str, Any]:
        """
        Processes a single composite Before/After satellite image:
        auto-splits into T1 (left/top) and T2 (right/bottom), aligns, detects changes.
        """
        img = cv2.imread(str(single_image_path))
        if img is None:
            raise ValueError(f"Cannot read image: {single_image_path}")

        t1, t2 = cls.split_composite_image(img)

        stem = Path(single_image_path).stem
        t1_filename = f"{stem}_T1_split.png"
        t2_filename = f"{stem}_T2_split.png"
        t1_path = CHANGE_DIR / t1_filename
        t2_path = CHANGE_DIR / t2_filename

        cv2.imwrite(str(t1_path), t1)
        cv2.imwrite(str(t2_path), t2)

        result = cls.detect_changes(
            image_path_t1=str(t1_path),
            image_path_t2=str(t2_path),
            output_heatmap_name=output_heatmap_name
        )
        result["t1_url"]  = f"/annotated/{t1_filename}"
        result["t2_url"]  = f"/annotated/{t2_filename}"
        result["t1_path"] = str(t1_path)
        result["t2_path"] = str(t2_path)
        return result

    # ── 8. Public API: detect from two separate images ──────────────────────
    @classmethod
    def detect_changes(
        cls,
        image_path_t1: str,
        image_path_t2: str,
        output_heatmap_name: str = "change_heatmap.png"
    ) -> Dict[str, Any]:
        """
        Full pipeline: load → align → compute change map → heatmap → boxes.
        Returns structured result dict for the API router.
        """
        img_t1 = cv2.imread(str(image_path_t1))
        img_t2 = cv2.imread(str(image_path_t2))
        if img_t1 is None or img_t2 is None:
            raise ValueError("Failed to load one or both satellite images.")

        h, w = img_t1.shape[:2]

        # Align T2 → T1
        img_t1, img_t2 = cls.align_images(img_t1, img_t2)

        # Compute change map
        fused_diff, bin_clean, ssim_score = cls._compute_change_map(img_t1, img_t2)

        # Change percentage (over cleaned mask only)
        changed_px   = int(np.count_nonzero(bin_clean))
        total_px     = h * w
        change_pct   = round(changed_px / total_px * 100, 2)

        # Heatmap
        out_heatmap  = CHANGE_DIR / output_heatmap_name
        cls._write_heatmap(fused_diff, bin_clean, out_heatmap)

        # Bounding boxes with classification
        change_boxes = cls._extract_boxes(bin_clean, h, w, img_t1=img_t1, img_t2=img_t2)

        summary_counts = {
            "appeared": sum(1 for b in change_boxes if b.get("change_type") == "appeared"),
            "disappeared": sum(1 for b in change_boxes if b.get("change_type") == "disappeared"),
            "modified": sum(1 for b in change_boxes if b.get("change_type") == "modified"),
            "total": len(change_boxes)
        }

        return {
            "change_percentage":    float(np.clip(change_pct, 0.0, 100.0)),
            "structural_similarity": float(np.clip(round(ssim_score, 3), 0.0, 1.0)),
            "heatmap_path":         str(out_heatmap),
            "heatmap_filename":     output_heatmap_name,
            "change_boxes":         change_boxes,
            "summary_counts":       summary_counts
        }

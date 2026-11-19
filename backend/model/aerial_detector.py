"""
Satellite & Aerial Feature Detector
Author: Sat_Assist Team

Specialized remote sensing object detector powered by satellite-pretrained YOLOv8
(NWPU-VHR10 benchmark: aircraft, ships, storage tanks, harbors, bridges, vehicles).
Provides multi-scale sliced aerial object detection for high-recall dense target detection.
"""

import os
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import cv2

# Local model checkpoint path
_DEFAULT_WEIGHTS = Path(__file__).resolve().parent.parent.parent / "models" / "yolo" / "nwpu_vhr10_yolov8n.pt"

_MODEL_INSTANCE = None


def _get_model():
    """Lazy loader for NWPU-VHR10 YOLO satellite detection model."""
    global _MODEL_INSTANCE
    if _MODEL_INSTANCE is None:
        try:
            from ultralytics import YOLO
            if _DEFAULT_WEIGHTS.exists():
                print(f"[AerialFeatureDetector] Loading local satellite YOLO: {_DEFAULT_WEIGHTS}")
                _MODEL_INSTANCE = YOLO(str(_DEFAULT_WEIGHTS))
            else:
                from huggingface_hub import hf_hub_download
                print(f"[AerialFeatureDetector] Downloading NWPU-VHR10 satellite YOLO...")
                pt_path = hf_hub_download(
                    repo_id="bluelabel/satellite-equipment-detection-yolov8n-vhr10",
                    filename="best.pt"
                )
                _MODEL_INSTANCE = YOLO(pt_path)
        except Exception as e:
            print(f"[AerialFeatureDetector] Failed to load YOLO satellite model: {e}")
            _MODEL_INSTANCE = None
    return _MODEL_INSTANCE


def _nms_fast(boxes: List[List[float]], confs: List[float], labels: List[str], iou_thresh: float = 0.35):
    """Vectorized Non-Maximum Suppression across detection slices."""
    if len(boxes) == 0:
        return []
    b_arr = np.array(boxes, dtype=np.float32)
    c_arr = np.array(confs, dtype=np.float32)
    x1 = b_arr[:, 0]
    y1 = b_arr[:, 1]
    x2 = b_arr[:, 2]
    y2 = b_arr[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = c_arr.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]
    return keep


class AerialFeatureDetector:
    @classmethod
    def detect_water_bodies(cls, image_path: str, query: str = "river") -> List[Dict[str, Any]]:
        """
        Remote sensing water body & river detection using spectral absorption indices
        (water strongly absorbs Red wavelengths while reflecting Cyan/Green/Blue) + morphological contour extraction.
        Returns tightly-bound reach sections along the river course or bounding envelopes.
        """
        if not os.path.exists(str(image_path)):
            return []

        img = cv2.imread(str(image_path))
        if img is None:
            return []

        h, w = img.shape[:2]
        b, g, r = cv2.split(img.astype(np.float32))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        H, S, V = cv2.split(hsv)

        # Multi-spectral RGB water indices:
        # 1. Turquoise / Azure / Greenish river water (e.g. Rhine, Danube, Ganges, etc.):
        #    B > R + 25, G > R + 25, (B/R > 1.35 and G/R > 1.35), R < 95
        # 2. Cyan / Blue open water: H in [70, 135], S > 30, B > R + 15
        # 3. Deep blue / dark water: R < 45, G < 60, B < 85, B >= R - 2
        w1 = (b - r > 30) & (g - r > 30) & (b > 55) & (g > 55) & (r < 85)
        w2 = (H >= 70) & (H <= 135) & (S >= 30) & (b > r + 15) & (g > r + 10) & (r < 100)
        w3 = (r < 40) & (g < 55) & (b < 80) & (b >= r - 3)

        water_mask = ((w1 | w2 | w3).astype(np.uint8)) * 255

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        clean_mask = cv2.morphologyEx(water_mask, cv2.MORPH_CLOSE, kernel)
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, kernel)

        cnts, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = (h * w) * 0.03  # at least 3% of scene
        large_cnts = [c for c in cnts if cv2.contourArea(c) > min_area]

        if not large_cnts:
            return []

        q_lower = (query or "").lower()
        if "lake" in q_lower:
            base_label = "Lake"
        elif "canal" in q_lower:
            base_label = "Canal"
        elif any(k in q_lower for k in ["ocean", "sea", "bay"]):
            base_label = "Coast / Sea"
        else:
            base_label = "River"

        boxes = []
        for c in sorted(large_cnts, key=cv2.contourArea, reverse=True)[:2]:
            x, y, bw, bh = cv2.boundingRect(c)
            # If the river spans significantly across the image, segment into reaches along the flow
            if bh > h * 0.40 or bw > w * 0.40:
                if bh >= bw:
                    # Vertical river flow: slice into 3 reaches
                    reaches = [
                        (0.0, 0.38, 'Upper Reach'),
                        (0.33, 0.72, 'Middle Reach'),
                        (0.67, 1.0, 'Lower Reach')
                    ]
                    for y0_f, y1_f, sec_name in reaches:
                        y0 = int(y0_f * h)
                        y1 = int(y1_f * h)
                        sub_slice = clean_mask[y0:y1, :]
                        s_cnts, _ = cv2.findContours(sub_slice, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if s_cnts:
                            sc = max(s_cnts, key=cv2.contourArea)
                            if cv2.contourArea(sc) > 1200:
                                sx, sy, sw, sh = cv2.boundingRect(sc)
                                boxes.append({
                                    'ymin': round((y0 + sy) / h, 4),
                                    'xmin': round(sx / w, 4),
                                    'ymax': round((y0 + sy + sh) / h, 4),
                                    'xmax': round((sx + sw) / w, 4),
                                    'label': f'{base_label} ({sec_name})',
                                    'confidence': 0.98
                                })
                else:
                    # Horizontal river flow: slice into 3 reaches
                    reaches = [
                        (0.0, 0.38, 'West Reach'),
                        (0.33, 0.72, 'Central Reach'),
                        (0.67, 1.0, 'East Reach')
                    ]
                    for x0_f, x1_f, sec_name in reaches:
                        x0 = int(x0_f * w)
                        x1 = int(x1_f * w)
                        sub_slice = clean_mask[:, x0:x1]
                        s_cnts, _ = cv2.findContours(sub_slice, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if s_cnts:
                            sc = max(s_cnts, key=cv2.contourArea)
                            if cv2.contourArea(sc) > 1200:
                                sx, sy, sw, sh = cv2.boundingRect(sc)
                                boxes.append({
                                    'ymin': round(sy / h, 4),
                                    'xmin': round((x0 + sx) / w, 4),
                                    'ymax': round((sy + sh) / h, 4),
                                    'xmax': round((x0 + sx + sw) / w, 4),
                                    'label': f'{base_label} ({sec_name})',
                                    'confidence': 0.98
                                })

            # If reaches were not created or didn't trigger, provide full envelope
            if not boxes:
                boxes.append({
                    'ymin': round(y / h, 4),
                    'xmin': round(x / w, 4),
                    'ymax': round((y + bh) / h, 4),
                    'xmax': round((x + bw) / w, 4),
                    'label': base_label,
                    'confidence': 0.98
                })

        return boxes

    @classmethod
    def detect_objects(cls, image_path: str, query: str = "") -> List[Dict[str, Any]]:
        """
        Detects remote-sensing objects (vehicles, ships, aircraft, storage tanks, bridges, rivers, etc.)
        using satellite-trained NWPU-VHR10 YOLOv8 with full-image + multi-scale sliced inference,
        and remote sensing spectral water analysis.
        Returns pixel-accurate normalized bounding boxes [ymin, xmin, ymax, xmax] in [0, 1].
        """
        q = (query or "").lower().strip()

        # Check for water bodies / rivers first
        water_keywords = ["river", "stream", "lake", "canal", "water", "waterbody", "water body", "ocean", "sea", "pond", "waterway", "bay", "creek"]
        if any(k in q for k in water_keywords):
            water_boxes = cls.detect_water_bodies(image_path, query=q)
            if water_boxes:
                print(f"[AerialFeatureDetector] Detected {len(water_boxes)} water/river regions for query: '{query}'")
                return water_boxes

        model = _get_model()
        if model is None or not os.path.exists(str(image_path)):
            return []

        img = cv2.imread(str(image_path))
        if img is None:
            return []
        orig_h, orig_w = img.shape[:2]

        # Class matching based on user query
        # NWPU Classes:
        # 0: 'airplane', 1: 'ship', 2: 'storage_tank', 3: 'baseball_diamond',
        # 4: 'tennis_court', 5: 'basketball_court', 6: 'ground_track_field',
        # 7: 'harbor', 8: 'bridge', 9: 'vehicle'
        target_labels = set()
        if any(k in q for k in ["vehicle", "car", "cars", "truck", "trucks", "automobile", "traffic", "van", "vans", "bus", "buses"]):
            target_labels.add("vehicle")
        if any(k in q for k in ["airplane", "airplanes", "plane", "planes", "aircraft", "jet", "jets", "flight", "runway"]):
            target_labels.add("airplane")
        if any(k in q for k in ["ship", "ships", "vessel", "vessels", "boat", "boats", "freighter"]):
            target_labels.add("ship")
        if any(k in q for k in ["tank", "tanks", "storage"]):
            target_labels.add("storage_tank")
        if any(k in q for k in ["harbor", "port", "dock"]):
            target_labels.add("harbor")
        if any(k in q for k in ["bridge"]):
            target_labels.add("bridge")

        # ONLY default to all classes if the user's prompt is a truly generic / broad query
        is_generic_query = any(k in q for k in ["all", "everything", "objects", "features", "anything", "detect all", "locate all"]) or not q
        if not target_labels:
            if is_generic_query:
                target_labels = {"vehicle", "airplane", "ship", "storage_tank", "harbor", "bridge"}
            else:
                # User asked for something specific that is not a YOLO object (e.g. specific landmark)
                # Never return unrelated ships or bridges!
                return []

        raw_boxes = []
        raw_confs = []
        raw_labels = []

        is_vehicle_query = "vehicle" in target_labels
        base_conf = 0.035 if is_vehicle_query else 0.08

        # --- Stage 1: Full-image high-resolution inference ---
        try:
            res_full = model(
                image_path,
                conf=base_conf,
                iou=0.45,
                imgsz=1280,
                augment=True,
                verbose=False
            )
            for b in res_full[0].boxes:
                cls_name = res_full[0].names.get(int(b.cls[0].item()), "").lower()
                if cls_name in target_labels:
                    raw_boxes.append(b.xyxy[0].tolist())
                    raw_confs.append(float(b.conf[0].item()))
                    raw_labels.append(cls_name)
        except Exception as e:
            print(f"[AerialFeatureDetector] Full inference error: {e}")

        # --- Stage 2: 2x2 Sliced High-Density Inference (SAHI-style) for small dense objects ---
        # Essential for catching 100% of tiny parked vehicles, queued cars, small boats, etc.
        if is_vehicle_query or "ship" in target_labels:
            try:
                tile_size_w = int(orig_w * 0.58)
                tile_size_h = int(orig_h * 0.58)
                step_x = int(orig_w * 0.42)
                step_y = int(orig_h * 0.42)

                y_offsets = [0, step_y] if orig_h > 350 else [0]
                x_offsets = [0, step_x] if orig_w > 350 else [0]

                for y0 in y_offsets:
                    for x0 in x_offsets:
                        y1 = min(orig_h, y0 + tile_size_h)
                        x1 = min(orig_w, x0 + tile_size_w)
                        crop = img[y0:y1, x0:x1]
                        if crop.size == 0:
                            continue
                        r_crop = model(
                            crop,
                            conf=base_conf,
                            imgsz=800,
                            verbose=False
                        )
                        for b in r_crop[0].boxes:
                            cls_name = r_crop[0].names.get(int(b.cls[0].item()), "").lower()
                            if cls_name in target_labels:
                                cx1, cy1, cx2, cy2 = b.xyxy[0].tolist()
                                # Shift tile coordinates back to global image space
                                raw_boxes.append([cx1 + x0, cy1 + y0, cx2 + x0, cy2 + y0])
                                raw_confs.append(float(b.conf[0].item()))
                                raw_labels.append(cls_name)
            except Exception as e:
                print(f"[AerialFeatureDetector] Sliced inference error: {e}")

        if not raw_boxes:
            return []

        # --- Stage 3: Global Non-Maximum Suppression to merge duplicates cleanly ---
        keep_indices = _nms_fast(raw_boxes, raw_confs, raw_labels, iou_thresh=0.35)

        boxes = []
        for idx in keep_indices:
            x1, y1, x2, y2 = raw_boxes[idx]
            conf = raw_confs[idx]
            cls_name = raw_labels[idx]

            # Boundary clamps
            x1 = max(0.0, min(float(orig_w), x1))
            x2 = max(0.0, min(float(orig_w), x2))
            y1 = max(0.0, min(float(orig_h), y1))
            y2 = max(0.0, min(float(orig_h), y2))

            # Filter trivial noise
            bw = x2 - x1
            bh = y2 - y1
            if bw < 4 or bh < 4:
                continue

            ymin = round(y1 / float(orig_h), 4)
            xmin = round(x1 / float(orig_w), 4)
            ymax = round(y2 / float(orig_h), 4)
            xmax = round(x2 / float(orig_w), 4)

            # Map label nicely
            if cls_name == "airplane":
                pretty_label = "Airplane"
            elif cls_name == "ship":
                pretty_label = "Ship"
            elif cls_name == "vehicle":
                pretty_label = "Vehicle"
            elif cls_name == "storage_tank":
                pretty_label = "Storage Tank"
            else:
                pretty_label = cls_name.replace("_", " ").title()

            boxes.append({
                "ymin": ymin,
                "xmin": xmin,
                "ymax": ymax,
                "xmax": xmax,
                "label": pretty_label,
                "confidence": round(conf, 3)
            })

        print(f"[AerialFeatureDetector] Detected {len(boxes)} satellite objects for query: '{query}'")
        return boxes

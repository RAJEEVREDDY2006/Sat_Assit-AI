"""
Spatial Bounding Box & Visual Grounding Parser for Remote Sensing VLM
Author: Sat_Assist Team

Parses various bounding box formats from Vision-Language Model outputs:
1. VRSBench Coordinate Format: {<ymin><xmin><ymax><xmax>} (0-100 normalized)
2. Qwen2-VL format: <|box_start|>(y1,x1),(y2,x2)<|box_end|> or (ymin,xmin,ymax,xmax) (0-1000 scale)
3. Normalized bracket format: [ymin, xmin, ymax, xmax] (0.0-1.0 or 0-1000)
4. Standard JSON/text coordinates with object labels.
"""

import re
from typing import List, Dict, Any, Tuple

class BoundingBoxParser:
    @staticmethod
    def normalize_coord(val: float, scale_max: float = 1000.0) -> float:
        """Clamps and normalizes a coordinate value to [0.0, 1.0]."""
        if val > 1.0:
            val = val / float(scale_max)
        return max(0.0, min(1.0, float(val)))

    @classmethod
    def parse_vrsbench_boxes(cls, text: str, default_label: str = "Target Object") -> List[Dict[str, Any]]:
        """
        Extracts VRSBench style bounding boxes:
        Example: '{<39><17><78><57>}' -> ymin=0.39, xmin=0.17, ymax=0.78, xmax=0.57
        """
        boxes = []
        pattern = r"\{<(-?\d+)><(-?\d+)><(-?\d+)><(-?\d+)>\}"
        for match in re.finditer(pattern, text):
            try:
                y1, x1, y2, x2 = map(float, match.groups())
                ymin = cls.normalize_coord(min(y1, y2), 100.0)
                xmin = cls.normalize_coord(min(x1, x2), 100.0)
                ymax = cls.normalize_coord(max(y1, y2), 100.0)
                xmax = cls.normalize_coord(max(x1, x2), 100.0)
                boxes.append({
                    "ymin": ymin,
                    "xmin": xmin,
                    "ymax": ymax,
                    "xmax": xmax,
                    "label": default_label,
                    "raw": match.group(0)
                })
            except Exception:
                continue
        return boxes

    @classmethod
    def parse_qwen_boxes(cls, text: str, default_label: str = "Detected Feature") -> List[Dict[str, Any]]:
        """
        Extracts Qwen-VL style bounding box tokens:
        <|box_start|>(ymin,xmin),(ymax,xmax)<|box_end|>
        or [ymin, xmin, ymax, xmax] on 0-1000 scale.
        """
        boxes = []
        # Pattern 1: <|box_start|>(y1,x1),(y2,x2)<|box_end|>
        pattern1 = r"<\|box_start\|>\((\d+),(\d+)\),\((\d+),(\d+)\)<\|box_end\|>"
        for match in re.finditer(pattern1, text):
            try:
                y1, x1, y2, x2 = map(float, match.groups())
                boxes.append({
                    "ymin": cls.normalize_coord(min(y1, y2), 1000.0),
                    "xmin": cls.normalize_coord(min(x1, x2), 1000.0),
                    "ymax": cls.normalize_coord(max(y1, y2), 1000.0),
                    "xmax": cls.normalize_coord(max(x1, x2), 1000.0),
                    "label": default_label,
                    "raw": match.group(0)
                })
            except Exception:
                continue

        # Pattern 2: (y1,x1) to (y2,x2) or (y1,x1),(y2,x2)
        pattern2 = r"\((\d{1,4}),\s*(\d{1,4})\)\s*(?:to|,|-)\s*\((\d{1,4}),\s*(\d{1,4})\)"
        for match in re.finditer(pattern2, text):
            try:
                y1, x1, y2, x2 = map(float, match.groups())
                boxes.append({
                    "ymin": cls.normalize_coord(min(y1, y2), 1000.0),
                    "xmin": cls.normalize_coord(min(x1, x2), 1000.0),
                    "ymax": cls.normalize_coord(max(y1, y2), 1000.0),
                    "xmax": cls.normalize_coord(max(x1, x2), 1000.0),
                    "label": default_label,
                    "raw": match.group(0)
                })
            except Exception:
                continue

        # Pattern 3: [ymin, xmin, ymax, xmax] or (ymin, xmin, ymax, xmax)
        pattern3 = r"[\[\(](\d{1,4}),\s*(\d{1,4}),\s*(\d{1,4}),\s*(\d{1,4})[\]\)]"
        for match in re.finditer(pattern3, text):
            try:
                y1, x1, y2, x2 = map(float, match.groups())
                max_val = max(y1, x1, y2, x2)
                scale = 1000.0 if max_val > 100.0 else 100.0
                boxes.append({
                    "ymin": cls.normalize_coord(min(y1, y2), scale),
                    "xmin": cls.normalize_coord(min(x1, x2), scale),
                    "ymax": cls.normalize_coord(max(y1, y2), scale),
                    "xmax": cls.normalize_coord(max(x1, x2), scale),
                    "label": default_label,
                    "raw": match.group(0)
                })
            except Exception:
                continue

        return boxes

    @classmethod
    def clean_text_response(cls, text: str, default_label: str = "Target Object") -> str:
        """
        Cleans coordinate tokens from raw text so user sees clean conversational response.
        """
        # Remove VRSBench tokens
        clean = re.sub(r"\{<(-?\d+)><(-?\d+)><(-?\d+)><(-?\d+)>\}\s*", "", text)
        # Remove completed Qwen box tokens & tags
        clean = re.sub(r"<\|box_start\|>.*?<\|box_end\|>\s*", "", clean)
        clean = re.sub(r"<\|object_ref_start\|>.*?<\|object_ref_end\|>\s*", "", clean)
        clean = re.sub(r"<\|ref_start\|>.*?<\|ref_end\|>\s*", "", clean)
        clean = re.sub(r"<\|im_end\|>", "", clean)
        clean = re.sub(r"\((\d{1,4}),\s*(\d{1,4})\)\s*(?:to|,|-)\s*\((\d{1,4}),\s*(\d{1,4})\)", "", clean)
        # Remove any truncated/unclosed box fragments at end of stream
        clean = re.sub(r"<\|box_start\|>.*$", "", clean)
        clean = re.sub(r"<\|object_ref_start\|>.*$", "", clean)
        clean = re.sub(r"\(\d{1,4},\s*\d{1,4}\)?.*$", "", clean)
        # Remove empty bracket artifacts
        clean = re.sub(r"\[\s*\]", "", clean).strip()
        return clean if clean else f"Located and highlighted {default_label} with bounding boxes in the satellite viewport."

    @classmethod
    def extract_all(cls, text: str, default_label: str = "Detected Feature") -> Tuple[str, List[Dict[str, Any]]]:
        """
        Extracts all bounding boxes and returns (cleaned_text, list_of_boxes).
        Filters out garbage full-image boxes (area > 80% of image).
        """
        boxes = []
        boxes.extend(cls.parse_vrsbench_boxes(text, default_label=default_label))
        if not boxes:
            boxes.extend(cls.parse_qwen_boxes(text, default_label=default_label))

        # Deduplicate overlapping boxes
        unique_boxes = []
        for b in boxes:
            # Check valid coordinates
            w = b["xmax"] - b["xmin"]
            h = b["ymax"] - b["ymin"]
            if w <= 0.005 or h <= 0.005:
                continue

            is_dup = any(
                abs(b["ymin"] - u["ymin"]) < 0.03 and
                abs(b["xmin"] - u["xmin"]) < 0.03 and
                abs(b["ymax"] - u["ymax"]) < 0.03 and
                abs(b["xmax"] - u["xmax"]) < 0.03
                for u in unique_boxes
            )
            if not is_dup:
                unique_boxes.append(b)

        # Filter out boxes covering > 80% of the image (nearly full-scene)
        valid_boxes = []
        for b in unique_boxes:
            area = (b["ymax"] - b["ymin"]) * (b["xmax"] - b["xmin"])
            if area < 0.80:
                valid_boxes.append(b)

        clean_text = cls.clean_text_response(text, default_label=default_label)
        return clean_text, valid_boxes

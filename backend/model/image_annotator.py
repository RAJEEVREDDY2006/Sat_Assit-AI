"""
Image Annotator — Server-Side Bounding Box Renderer
Author: Sat_Assist Team

Draws bounding boxes with labels onto satellite images using PIL.
Returns a downloadable annotated PNG file.
"""

from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from typing import List, Dict, Any
import io

# Color palette for different object types (label keyword → RGB color)
LABEL_COLORS = {
    "vehicle":   (6,   182, 212),   # cyan
    "car":       (6,   182, 212),
    "truck":     (6,   182, 212),
    "aircraft":  (251, 191,  36),   # amber
    "airplane":  (251, 191,  36),
    "runway":    (251, 191,  36),
    "building":  (99,  102, 241),   # indigo
    "ship":      (52,  211, 153),   # emerald
    "vessel":    (52,  211, 153),
    "road":      (248, 113, 113),   # red
    "default":   (6,   182, 212),   # cyan fallback
}

ANNOTATED_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "annotated"
ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)


def _get_label_color(label: str) -> tuple:
    """Pick a color based on the label keyword."""
    label_lower = label.lower()
    for keyword, color in LABEL_COLORS.items():
        if keyword in label_lower:
            return color
    return LABEL_COLORS["default"]


def _try_load_font(size: int) -> ImageFont.FreeTypeFont:
    """Load a font, fall back to PIL default if no TTF available."""
    try:
        # Try common system fonts on Windows
        for font_path in [
            "C:/Windows/Fonts/arialbd.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/calibrib.ttf",
        ]:
            if Path(font_path).exists():
                return ImageFont.truetype(font_path, size)
    except Exception:
        pass
    return ImageFont.load_default()


class ImageAnnotator:

    @staticmethod
    def annotate(
        image_path: str,
        bounding_boxes: List[Dict[str, Any]],
        output_filename: str,
        line_thickness: int = 3,
        label_font_size: int = 14,
    ) -> str:
        """
        Draws bounding boxes + labels onto the image and saves the result as PNG.

        Args:
            image_path:      Path to the original satellite image.
            bounding_boxes:  List of dicts with keys: ymin, xmin, ymax, xmax (0-1 normalized), label.
            output_filename: Filename to save the annotated PNG (e.g. "sat_abc_annotated.png").
            line_thickness:  Box border thickness in pixels.
            label_font_size: Font size for the label chip text.

        Returns:
            Absolute path to the saved annotated PNG.
        """
        img = Image.open(image_path).convert("RGB")
        W, H = img.size
        draw = ImageDraw.Draw(img, "RGBA")
        font = _try_load_font(label_font_size)

        # If dense objects (e.g. > 8 boxes), adapt chip size and suppress repetitive chips so image stays clear
        is_dense = len(bounding_boxes) > 8
        chip_font_size = 9 if is_dense else label_font_size
        font = _try_load_font(chip_font_size)

        for i, box in enumerate(bounding_boxes):
            try:
                # Convert normalized [0,1] coords to pixel coordinates
                x1 = int(box["xmin"] * W)
                y1 = int(box["ymin"] * H)
                x2 = int(box["xmax"] * W)
                y2 = int(box["ymax"] * H)

                # Clamp to image bounds
                x1, x2 = max(0, x1), min(W, x2)
                y1, y2 = max(0, y1), min(H, y2)

                label = box.get("label", f"Object #{i + 1}")
                color = _get_label_color(label)
                color_alpha = (*color, 45)   # semi-transparent fill

                # Draw semi-transparent filled rectangle with crisp colored border
                draw.rectangle([x1, y1, x2, y2], fill=color_alpha, outline=color, width=line_thickness)

                # For dense detections (many small objects), only show chip if box is large enough or periodically
                # so chips don't overlap and occlude the vehicles underneath!
                box_w = x2 - x1
                box_h = y2 - y1
                draw_chip = (not is_dense) or (box_w >= 18 and (i % 3 == 0 or i < 4))

                if draw_chip:
                    try:
                        bbox_text = draw.textbbox((0, 0), label, font=font)
                        text_w = bbox_text[2] - bbox_text[0]
                        text_h = bbox_text[3] - bbox_text[1]
                    except AttributeError:
                        text_w, text_h = draw.textsize(label, font=font)

                    pad = 2 if is_dense else 4
                    chip_x1 = x1
                    chip_y1 = max(0, y1 - text_h - pad * 2)
                    chip_x2 = min(W, x1 + text_w + pad * 2)
                    chip_y2 = max(text_h + pad * 2, y1)

                    # Chip background (solid color)
                    draw.rectangle([chip_x1, chip_y1, chip_x2, chip_y2], fill=color)
                    # Chip text (black for contrast)
                    draw.text(
                        (chip_x1 + pad, chip_y1 + pad),
                        label,
                        fill=(0, 0, 0),
                        font=font
                    )

            except Exception as e:
                print(f"[Annotator] Skipped box {i}: {e}")
                continue

        # Watermark
        wm_text = "SatAssist AI"
        try:
            wm_font = _try_load_font(11)
            draw.text((8, H - 20), wm_text, fill=(255, 255, 255, 160), font=wm_font)
        except Exception:
            pass

        out_path = ANNOTATED_DIR / output_filename
        img.save(str(out_path), format="PNG", optimize=True)
        print(f"[Annotator] Saved annotated image: {out_path}")
        return str(out_path)

    @staticmethod
    def annotate_to_bytes(
        image_path: str,
        bounding_boxes: List[Dict[str, Any]],
    ) -> bytes:
        """Annotates the image and returns PNG bytes (for streaming without saving)."""
        img = Image.open(image_path).convert("RGB")
        W, H = img.size
        draw = ImageDraw.Draw(img, "RGBA")
        font = _try_load_font(14)

        for i, box in enumerate(bounding_boxes):
            try:
                x1 = int(box["xmin"] * W)
                y1 = int(box["ymin"] * H)
                x2 = int(box["xmax"] * W)
                y2 = int(box["ymax"] * H)
                x1, x2 = max(0, x1), min(W, x2)
                y1, y2 = max(0, y1), min(H, y2)

                label = box.get("label", f"Object #{i + 1}")
                color = _get_label_color(label)
                draw.rectangle([x1, y1, x2, y2], fill=(*color, 50), outline=color, width=3)

                try:
                    bbox_text = draw.textbbox((0, 0), label, font=font)
                    text_w = bbox_text[2] - bbox_text[0]
                    text_h = bbox_text[3] - bbox_text[1]
                except AttributeError:
                    text_w, text_h = draw.textsize(label, font=font)

                pad = 4
                chip_y1 = max(0, y1 - text_h - pad * 2)
                draw.rectangle([x1, chip_y1, x1 + text_w + pad * 2, max(text_h + pad * 2, y1)], fill=color)
                draw.text((x1 + pad, chip_y1 + pad), label, fill=(0, 0, 0), font=font)

            except Exception:
                continue

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()


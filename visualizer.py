#draw colored boxes & labels

from typing import List, Dict
from PIL import Image, ImageDraw, ImageFont
import hashlib
import os

class Visualizer:
    def __init__(self, visual_dir: str = "visuals"):
        self.visual_dir = visual_dir
        os.makedirs(self.visual_dir, exist_ok=True)
        # try to load a ttf font if available
        try:
            self.font = ImageFont.truetype("DejaVuSans.ttf", size=16)
        except Exception:
            self.font = ImageFont.load_default()

    def _color_for_group(self, group_label) -> tuple:
        """
        Deterministic color per group integer label.
        group_label may be negative (noise).
        """
        s = str(group_label).encode("utf-8")
        h = hashlib.sha1(s).hexdigest()
        r = int(h[0:2], 16)
        g = int(h[2:4], 16)
        b = int(h[4:6], 16)
        return (r, g, b)

    def draw_boxes_and_groups(self, pil_img: Image.Image, detections: List[Dict], groups, out_path: str):
        """
        pil_img: original image (PIL)
        detections: list of dicts with "bbox" e.g. [x1,y1,x2,y2], "score", "detection_id"
        groups: list/array of integer labels aligned with detections order
        out_path: file path to save visualization
        """
        img = pil_img.copy()
        draw = ImageDraw.Draw(img)

        for i, det in enumerate(detections):
            bbox = det["bbox"]
            label = det.get("detection_id", f"det_{i}")
            score = det.get("score", None)
            group_label = int(groups[i]) if len(groups)>i else -1
            color = self._color_for_group(group_label)
            x1,y1,x2,y2 = bbox
            draw.rectangle([x1,y1,x2,y2], outline=color, width=4)
            text_label = f"{label[:8]} {('%.2f'%score) if score is not None else ''} G{group_label}"
            # text background
            tw, th = draw.textsize(text_label, font=self.font)
            ty = max(0, y1 - th - 4)
            draw.rectangle([x1, ty, x1+tw+6, ty+th+4], fill=(0,0,0))
            draw.text((x1+3, ty+2), text_label, fill=color, font=self.font)

        img.save(out_path, quality=90)
        return out_path

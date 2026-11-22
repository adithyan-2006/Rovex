"""
Visualization and Annotation Engine for Road Damage and Severity Assessment.
Precision optical survey overlay with civil engineering telemetry and ASTM classification.
"""

from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image

from .severity import DefectAssessment, RoadAssessmentReport, SeverityGrade, RoadCondition

# Standard Civil Engineering Palette (BGR format)
# Low (Minor): Standard Green (#16A34A / #22C55E)
# Medium (Moderate): Technical Amber (#D97706)
# High (Severe): Brick Red (#EA580C)
# Critical: Deep Crimson (#DC2626)
SEVERITY_COLORS_BGR: Dict[SeverityGrade, Tuple[int, int, int]] = {
    SeverityGrade.LOW: (74, 197, 34),        # Standard Green (#22C55E in BGR)
    SeverityGrade.MEDIUM: (6, 119, 217),     # Technical Amber (#D97706 in BGR)
    SeverityGrade.HIGH: (12, 88, 234),       # Brick Red (#EA580C in BGR)
    SeverityGrade.CRITICAL: (38, 38, 220),   # Deep Crimson (#DC2626 in BGR)
}

SEVERITY_COLORS_RGB: Dict[SeverityGrade, Tuple[int, int, int]] = {
    SeverityGrade.LOW: (34, 197, 74),
    SeverityGrade.MEDIUM: (217, 119, 6),
    SeverityGrade.HIGH: (234, 88, 12),
    SeverityGrade.CRITICAL: (220, 38, 38),
}

CONDITION_ACCENTS_BGR: Dict[RoadCondition, Tuple[int, int, int]] = {
    RoadCondition.GOOD: (74, 197, 34),
    RoadCondition.FAIR: (6, 119, 217),
    RoadCondition.POOR: (12, 88, 234),
    RoadCondition.CRITICAL: (38, 38, 220),
}


class RoadDamageVisualizer:
    """Renders precision survey telemetry, reticles, and severity bounding boxes on pavement imagery."""

    def __init__(
        self,
        box_thickness: int = 2,
        font_scale: float = 0.50,
        show_hud: bool = True,
    ):
        self.box_thickness = box_thickness
        self.font_scale = font_scale
        self.show_hud = show_hud

    def draw_hud(
        self,
        img: np.ndarray,
        report: RoadAssessmentReport,
    ) -> np.ndarray:
        """Renders a precision civil engineering survey telemetry bar at the top of the frame."""
        h, w = img.shape[:2]
        hud_h = max(52, int(h * 0.085))
        hud_h = min(hud_h, 80)

        # Technical dark slate banner (semi-transparent)
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, hud_h), (17, 24, 34), -1)
        cv2.addWeighted(overlay, 0.88, img, 0.12, 0, img)

        # 1px bottom separator with standard green accent
        accent = CONDITION_ACCENTS_BGR.get(report.road_condition, (74, 197, 34))
        cv2.line(img, (0, hud_h - 1), (w, hud_h - 1), (45, 55, 70), 1)
        cv2.line(img, (0, hud_h), (min(w, int(w * (report.pavement_condition_index / 100.0))), hud_h), accent, 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        t_scale = max(0.40, min(0.60, w / 1800.0))
        y_center = int(hud_h * 0.58)

        # Left: System identification & condition
        tag_left = f"ROVEX SURVEY  |  PCI: {report.pavement_condition_index:.1f}  |  RATING: {report.road_condition.value.upper()}"
        cv2.putText(img, tag_left, (16, y_center), font, t_scale, (241, 245, 249), 1, cv2.LINE_AA)

        # Right: Telemetry breakdown
        tag_right = f"DEFECTS: {report.total_defects:02d} [P:{report.pothole_count} C:{report.crack_count}]  |  CRITICAL: {report.critical_defects_count}"
        (tw, _), _ = cv2.getTextSize(tag_right, font, t_scale * 0.90, 1)
        cv2.putText(img, tag_right, (w - tw - 16, y_center), font, t_scale * 0.90, (148, 163, 184), 1, cv2.LINE_AA)

        return img

    def draw_defect_box(
        self,
        img: np.ndarray,
        defect: DefectAssessment,
    ) -> np.ndarray:
        """Renders precision bounding box with corner crosshairs and technical spec tag."""
        h, w = img.shape[:2]
        x1, y1, x2, y2 = defect.box
        color = SEVERITY_COLORS_BGR.get(defect.severity_grade, (148, 163, 184))

        # Precision 1-2px bounding outline
        cv2.rectangle(img, (x1, y1), (x2, y2), color, self.box_thickness)

        # Precision corner reticles (technical surveyor aesthetic)
        reticle_len = min(int((x2 - x1) * 0.20), int((y2 - y1) * 0.20), 16)
        if reticle_len > 4:
            c_thick = self.box_thickness + 1
            cv2.line(img, (x1, y1), (x1 + reticle_len, y1), (255, 255, 255), c_thick)
            cv2.line(img, (x1, y1), (x1, y1 + reticle_len), (255, 255, 255), c_thick)
            cv2.line(img, (x2, y1), (x2 - reticle_len, y1), (255, 255, 255), c_thick)
            cv2.line(img, (x2, y1), (x2, y1 + reticle_len), (255, 255, 255), c_thick)
            cv2.line(img, (x1, y2), (x1 + reticle_len, y2), (255, 255, 255), c_thick)
            cv2.line(img, (x1, y2), (x1, y2 - reticle_len), (255, 255, 255), c_thick)
            cv2.line(img, (x2, y2), (x2 - reticle_len, y2), (255, 255, 255), c_thick)
            cv2.line(img, (x2, y2), (x2, y2 - reticle_len), (255, 255, 255), c_thick)

        # Technical Data Label: "[#01] POTHOLE | SEV: CRITICAL (84) | CONF: 0.91"
        type_code = defect.raw_class.upper()
        label_text = f"#{defect.defect_id:02d} {defect.display_name.upper()} [{type_code}]"
        meta_text = f"SEV: {defect.severity_grade.value.upper()} ({defect.severity_score:.0f}) | C: {defect.confidence:.2f}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.38, min(0.52, w / 1800.0))

        (tw1, th1), _ = cv2.getTextSize(label_text, font, font_scale, 1)
        (tw2, th2), _ = cv2.getTextSize(meta_text, font, font_scale * 0.85, 1)

        box_w = max(tw1, tw2) + 12
        box_h = th1 + th2 + 12

        # Placement logic
        tag_y2 = y1 - 2
        tag_y1 = tag_y2 - box_h
        if tag_y1 < (70 if self.show_hud else 6):
            tag_y1 = y1 + 2
            tag_y2 = tag_y1 + box_h

        tag_x1 = max(2, x1)
        tag_x2 = min(w - 2, tag_x1 + box_w)

        # Dark charcoal label chip with 1px border
        cv2.rectangle(img, (tag_x1, tag_y1), (tag_x2, tag_y2), (16, 22, 28), -1)
        cv2.rectangle(img, (tag_x1, tag_y1), (tag_x2, tag_y2), color, 1)

        # Primary classification text
        cv2.putText(
            img,
            label_text,
            (tag_x1 + 6, tag_y1 + th1 + 3),
            font,
            font_scale,
            (248, 250, 252),
            1,
            cv2.LINE_AA,
        )
        # Severity and confidence meta text
        cv2.putText(
            img,
            meta_text,
            (tag_x1 + 6, tag_y1 + th1 + th2 + 8),
            font,
            font_scale * 0.85,
            color,
            1,
            cv2.LINE_AA,
        )

        return img

    def annotate(
        self,
        image_input: Union[np.ndarray, Image.Image, str],
        report: RoadAssessmentReport,
        return_pil: bool = False,
    ) -> Union[np.ndarray, Image.Image]:
        """Renders precision inspection overlay onto the target pavement image."""
        if isinstance(image_input, str):
            img_bgr = cv2.imread(image_input)
            if img_bgr is None:
                raise FileNotFoundError(f"Could not load image: {image_input}")
        elif isinstance(image_input, Image.Image):
            rgb = np.array(image_input)
            img_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        elif isinstance(image_input, np.ndarray):
            img_bgr = image_input.copy()
            if len(img_bgr.shape) == 2:
                img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
        else:
            raise TypeError(f"Unsupported image type: {type(image_input)}")

        # Draw defect bounding boxes
        for defect in report.defects:
            img_bgr = self.draw_defect_box(img_bgr, defect)

        # Draw HUD bar
        if self.show_hud:
            img_bgr = self.draw_hud(img_bgr, report)

        if return_pil:
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb)

        return img_bgr

    def create_side_by_side(
        self,
        original_img: Union[np.ndarray, Image.Image],
        annotated_img: Union[np.ndarray, Image.Image],
    ) -> np.ndarray:
        """Combines original and annotated image with a precision divider line."""
        if isinstance(original_img, Image.Image):
            orig_bgr = cv2.cvtColor(np.array(original_img), cv2.COLOR_RGB2BGR)
        else:
            orig_bgr = original_img.copy()

        if isinstance(annotated_img, Image.Image):
            ann_bgr = cv2.cvtColor(np.array(annotated_img), cv2.COLOR_RGB2BGR)
        else:
            ann_bgr = annotated_img.copy()

        h1, w1 = orig_bgr.shape[:2]
        h2, w2 = ann_bgr.shape[:2]
        if h1 != h2:
            ann_bgr = cv2.resize(ann_bgr, (int(w2 * (h1 / h2)), h1))

        divider = np.zeros((h1, 2, 3), dtype=np.uint8)
        divider[:] = (51, 65, 85)

        return np.hstack([orig_bgr, divider, ann_bgr])

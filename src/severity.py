"""
Severity Assessment Engine for Road Damage Detection.
Implements civil-engineering-inspired calculations (ASTM D6433 PCI deduct logic,
geometric scaling, road perspective correction, and defect risk weighting).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


class SeverityGrade(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"


class RoadCondition(str, Enum):
    GOOD = "Good"
    FAIR = "Fair"
    POOR = "Poor"
    CRITICAL = "Critical"


# Base hazard weights by defect category (0.0 to 1.0)
DAMAGE_WEIGHTS: Dict[str, float] = {
    # RDD benchmark classes
    "D40": 1.00,  # Pothole
    "D20": 0.85,  # Alligator Crack (Fatigue cracking - base failure)
    "D10": 0.65,  # Transverse Crack
    "D00": 0.50,  # Longitudinal Crack
    "D43": 0.40,  # Crosswalk / line blur
    # Human-readable labels
    "pothole": 1.00,
    "alligator crack": 0.85,
    "alligator_crack": 0.85,
    "transverse crack": 0.65,
    "transverse_crack": 0.65,
    "longitudinal crack": 0.50,
    "longitudinal_crack": 0.50,
    "crack": 0.60,
    "cracks": 0.60,
    "manhole": 0.30,
    "other": 0.40,
}

# Standardized display names
DAMAGE_LABELS: Dict[str, str] = {
    "D00": "Longitudinal Crack",
    "D10": "Transverse Crack",
    "D20": "Alligator Crack",
    "D40": "Pothole",
    "D43": "Surface Marking Blur",
    "pothole": "Pothole",
    "alligator crack": "Alligator Crack",
    "alligator_crack": "Alligator Crack",
    "transverse crack": "Transverse Crack",
    "transverse_crack": "Transverse Crack",
    "longitudinal crack": "Longitudinal Crack",
    "longitudinal_crack": "Longitudinal Crack",
    "crack": "Road Crack",
    "cracks": "Road Crack",
}


@dataclass
class DefectAssessment:
    """Detailed severity assessment for an individual detected road defect."""
    defect_id: int
    raw_class: str
    display_name: str
    confidence: float
    box: List[int]  # [x1, y1, x2, y2]
    box_width: int
    box_height: int
    relative_area_pct: float  # Percentage of image area
    perspective_factor: float  # Scaling based on road perspective distance
    estimated_depth_contrast: float  # Shadow/darkness contrast for depth estimation
    severity_score: float  # 0 to 100
    severity_grade: SeverityGrade
    repair_recommendation: str
    hazard_weight: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "defect_id": self.defect_id,
            "raw_class": self.raw_class,
            "display_name": self.display_name,
            "confidence": round(self.confidence, 4),
            "box": self.box,
            "box_width": self.box_width,
            "box_height": self.box_height,
            "relative_area_pct": round(self.relative_area_pct, 3),
            "perspective_factor": round(self.perspective_factor, 2),
            "estimated_depth_contrast": round(self.estimated_depth_contrast, 2),
            "severity_score": round(self.severity_score, 1),
            "severity_grade": self.severity_grade.value,
            "repair_recommendation": self.repair_recommendation,
            "hazard_weight": self.hazard_weight,
        }


@dataclass
class RoadAssessmentReport:
    """Comprehensive pavement condition and severity assessment report for an image."""
    total_defects: int
    pothole_count: int
    crack_count: int
    pavement_condition_index: float  # 0 to 100 (PCI)
    road_condition: RoadCondition
    max_defect_severity: float
    average_defect_severity: float
    critical_defects_count: int
    high_defects_count: int
    medium_defects_count: int
    low_defects_count: int
    action_summary: str
    defects: List[DefectAssessment] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_defects": self.total_defects,
            "pothole_count": self.pothole_count,
            "crack_count": self.crack_count,
            "pavement_condition_index": round(self.pavement_condition_index, 1),
            "road_condition": self.road_condition.value,
            "max_defect_severity": round(self.max_defect_severity, 1),
            "average_defect_severity": round(self.average_defect_severity, 1),
            "defect_counts_by_grade": {
                "critical": self.critical_defects_count,
                "high": self.high_defects_count,
                "medium": self.medium_defects_count,
                "low": self.low_defects_count,
            },
            "action_summary": self.action_summary,
            "defects": [d.to_dict() for d in self.defects],
        }


class SeverityAnalyzer:
    """Calculates defect severity and overall Pavement Condition Index (PCI)."""

    def __init__(self, perspective_weight: float = 1.5):
        """
        Args:
            perspective_weight: Strength of perspective correction for distant road defects.
        """
        self.perspective_weight = perspective_weight

    def resolve_label(self, raw_class: str) -> Tuple[str, str, float]:
        """Maps model class string to display name, canonical type, and hazard weight."""
        key = str(raw_class).strip().lower()
        original_key = str(raw_class).strip()

        # Check raw code (e.g. D40, D00)
        if original_key in DAMAGE_LABELS:
            display_name = DAMAGE_LABELS[original_key]
            weight = DAMAGE_WEIGHTS.get(original_key, 0.5)
            canonical = "pothole" if "pothole" in display_name.lower() else "crack"
            return display_name, canonical, weight

        # Check lowercase
        if key in DAMAGE_LABELS:
            display_name = DAMAGE_LABELS[key]
            weight = DAMAGE_WEIGHTS.get(key, 0.5)
            canonical = "pothole" if "pothole" in display_name.lower() else "crack"
            return display_name, canonical, weight

        # Fuzzy check
        if "pothole" in key or "d40" in key:
            return "Pothole", "pothole", 1.00
        elif "alligator" in key or "d20" in key:
            return "Alligator Crack", "crack", 0.85
        elif "transverse" in key or "d10" in key:
            return "Transverse Crack", "crack", 0.65
        elif "longitudinal" in key or "d00" in key:
            return "Longitudinal Crack", "crack", 0.50
        elif "crack" in key:
            return "Road Crack", "crack", 0.60

        return original_key, "defect", 0.50

    def compute_perspective_factor(self, y_center: float, img_height: float) -> float:
        """
        In road perspective geometry, objects higher up the frame (closer to horizon)
        represent larger physical ground area per pixel than objects near the bottom bumper.
        """
        if img_height <= 0:
            return 1.0
        normalized_y = np.clip(y_center / img_height, 0.05, 1.0)
        # Closer to camera (bottom, y ~ 1.0): factor ~ 1.0
        # Farther from camera (mid-top, y ~ 0.3): factor up to ~ 2.5
        perspective_factor = 1.0 + self.perspective_weight * ((1.0 - normalized_y) ** 1.8)
        return float(perspective_factor)

    def estimate_depth_contrast(
        self,
        image: Optional[np.ndarray],
        box: List[int],
    ) -> float:
        """
        Estimates contrast/darkness ratio between defect interior and surrounding asphalt.
        Potholes typically have low-luminance shaded cores; surface marks do not.
        Returns a score in range [1.0, 2.0].
        """
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            return 1.0

        try:
            h, w = image.shape[:2]
            x1, y1, x2, y2 = [int(v) for v in box]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 <= x1 or y2 <= y1:
                return 1.0

            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = np.mean(image, axis=2).astype(np.float32)
            else:
                gray = image.astype(np.float32)

            defect_roi = gray[y1:y2, x1:x2]
            if defect_roi.size == 0:
                return 1.0

            mean_defect = float(np.mean(defect_roi))

            # Surround margin
            pad_x = max(5, int((x2 - x1) * 0.3))
            pad_y = max(5, int((y2 - y1) * 0.3))
            sx1, sy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
            sx2, sy2 = min(w, x2 + pad_x), min(h, y2 + pad_y)

            surround_roi = gray[sy1:sy2, sx1:sx2]
            mean_surround = float(np.mean(surround_roi))

            if mean_surround > 5.0 and mean_defect < mean_surround:
                # Depression darkness ratio
                darkness_ratio = (mean_surround - mean_defect) / mean_surround
                contrast_factor = 1.0 + min(1.0, darkness_ratio * 2.0)
                return float(contrast_factor)
            return 1.0
        except Exception:
            return 1.0

    def get_repair_recommendation(
        self,
        canonical_type: str,
        grade: SeverityGrade,
    ) -> str:
        """Determines actionable civil maintenance recommendation based on defect and severity."""
        if canonical_type == "pothole":
            if grade == SeverityGrade.CRITICAL:
                return "Emergency excavation & structural hot-mix patch; immediate traffic warning"
            elif grade == SeverityGrade.HIGH:
                return "Full-depth asphalt patching and edge compaction within 48h"
            elif grade == SeverityGrade.MEDIUM:
                return "Standard semi-permanent mastic or cold-mix pothole repair"
            else:
                return "Surface patch leveling and monitor in scheduled cycle"
        else:
            # Crack types
            if grade == SeverityGrade.CRITICAL:
                return "Severe sub-base fatigue; milling and asphalt overlay required"
            elif grade == SeverityGrade.HIGH:
                return "Rout and crack-seal with hot-pour polymer-modified bitumen"
            elif grade == SeverityGrade.MEDIUM:
                return "Standard rubberized asphalt crack sealant application"
            else:
                return "Minor hairline crack; preventive surface seal during routine pass"

    def assess_defect(
        self,
        defect_id: int,
        raw_class: str,
        confidence: float,
        box: List[int],
        img_shape: Tuple[int, int],
        image: Optional[np.ndarray] = None,
    ) -> DefectAssessment:
        """Performs comprehensive severity grading for a single bounding box."""
        h, w = img_shape[:2]
        x1, y1, x2, y2 = [int(v) for v in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        defect_pixel_area = bw * bh
        img_pixel_area = max(1, w * h)

        relative_area_pct = (defect_pixel_area / img_pixel_area) * 100.0

        y_center = (y1 + y2) / 2.0
        perspective_factor = self.compute_perspective_factor(y_center, h)
        depth_contrast = self.estimate_depth_contrast(image, box)
        display_name, canonical_type, hazard_weight = self.resolve_label(raw_class)

        # Adjusted effective area factoring road distance
        effective_area_pct = relative_area_pct * perspective_factor

        # Severity Score Computation (0 to 100)
        # Factor 1: Normalized Geometric Size (0 - 50 pts)
        # 0.1% area -> 5 pts, 2% area -> 25 pts, >6% area -> 50 pts
        size_score = np.clip((effective_area_pct / 6.0) * 50.0, 2.0, 50.0)

        # Factor 2: Hazard Weight (0 - 35 pts)
        hazard_score = hazard_weight * 35.0

        # Factor 3: Depth Contrast (0 - 15 pts)
        depth_score = (depth_contrast - 1.0) * 15.0

        raw_severity = size_score + hazard_score + depth_score
        severity_score = float(np.clip(raw_severity, 5.0, 100.0))

        # Categorize into Grade
        if severity_score >= 75.0 or (canonical_type == "pothole" and effective_area_pct > 3.5):
            grade = SeverityGrade.CRITICAL
        elif severity_score >= 55.0:
            grade = SeverityGrade.HIGH
        elif severity_score >= 32.0:
            grade = SeverityGrade.MEDIUM
        else:
            grade = SeverityGrade.LOW

        recommendation = self.get_repair_recommendation(canonical_type, grade)

        return DefectAssessment(
            defect_id=defect_id,
            raw_class=str(raw_class),
            display_name=display_name,
            confidence=float(confidence),
            box=[x1, y1, x2, y2],
            box_width=bw,
            box_height=bh,
            relative_area_pct=relative_area_pct,
            perspective_factor=perspective_factor,
            estimated_depth_contrast=depth_contrast,
            severity_score=severity_score,
            severity_grade=grade,
            repair_recommendation=recommendation,
            hazard_weight=hazard_weight,
        )

    def assess_image(
        self,
        detections: List[Dict[str, Any]],
        img_shape: Tuple[int, int],
        image: Optional[np.ndarray] = None,
    ) -> RoadAssessmentReport:
        """
        Assesses all detections in an image and computes aggregate Pavement Condition Index (PCI).
        """
        if not detections:
            return RoadAssessmentReport(
                total_defects=0,
                pothole_count=0,
                crack_count=0,
                pavement_condition_index=100.0,
                road_condition=RoadCondition.GOOD,
                max_defect_severity=0.0,
                average_defect_severity=0.0,
                critical_defects_count=0,
                high_defects_count=0,
                medium_defects_count=0,
                low_defects_count=0,
                action_summary="Pavement in good condition. No distress detected.",
                defects=[],
            )

        assessed_defects: List[DefectAssessment] = []
        potholes = 0
        cracks = 0
        crit_count = 0
        high_count = 0
        med_count = 0
        low_count = 0

        # Deduct values for ASTM D6433 style calculation
        total_deduct = 0.0

        for idx, det in enumerate(detections, start=1):
            assessment = self.assess_defect(
                defect_id=idx,
                raw_class=det.get("class", det.get("class_name", "defect")),
                confidence=det.get("confidence", 0.5),
                box=det.get("box", [0, 0, 10, 10]),
                img_shape=img_shape,
                image=image,
            )
            assessed_defects.append(assessment)

            if "pothole" in assessment.display_name.lower():
                potholes += 1
            else:
                cracks += 1

            if assessment.severity_grade == SeverityGrade.CRITICAL:
                crit_count += 1
                deduct_multiplier = 1.4
            elif assessment.severity_grade == SeverityGrade.HIGH:
                high_count += 1
                deduct_multiplier = 1.1
            elif assessment.severity_grade == SeverityGrade.MEDIUM:
                med_count += 1
                deduct_multiplier = 0.8
            else:
                low_count += 1
                deduct_multiplier = 0.5

            # Deduct contribution per defect
            deduct = (assessment.severity_score / 100.0) * 30.0 * deduct_multiplier
            total_deduct += deduct

        # Deduct curve with diminishing returns for overlapping multiple defects
        effective_deduct = total_deduct / (1.0 + 0.12 * max(0, len(detections) - 1))
        pci = float(np.clip(100.0 - effective_deduct, 0.0, 100.0))

        # Overall Road Condition Label
        if pci >= 85.0:
            condition = RoadCondition.GOOD
            summary = "Pavement in Good condition. Minor surface maintenance recommended."
        elif pci >= 60.0:
            condition = RoadCondition.FAIR
            summary = "Pavement in Fair condition. Moderate distress observed; schedule repairs."
        elif pci >= 35.0:
            condition = RoadCondition.POOR
            summary = "Pavement in Poor condition. Significant hazards; prioritized repair required."
        else:
            condition = RoadCondition.CRITICAL
            summary = "Pavement in Critical condition! Severe structural deterioration; urgent intervention needed."

        severities = [d.severity_score for d in assessed_defects]
        max_sev = float(max(severities)) if severities else 0.0
        avg_sev = float(np.mean(severities)) if severities else 0.0

        return RoadAssessmentReport(
            total_defects=len(assessed_defects),
            pothole_count=potholes,
            crack_count=cracks,
            pavement_condition_index=pci,
            road_condition=condition,
            max_defect_severity=max_sev,
            average_defect_severity=avg_sev,
            critical_defects_count=crit_count,
            high_defects_count=high_count,
            medium_defects_count=med_count,
            low_defects_count=low_count,
            action_summary=summary,
            defects=assessed_defects,
        )

"""
Unit tests for Severity Assessment and PCI calculations in Rovex.
"""

import unittest
import numpy as np

from src.severity import (
    SeverityAnalyzer,
    SeverityGrade,
    RoadCondition,
    DAMAGE_WEIGHTS,
)


class TestSeverityAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = SeverityAnalyzer(perspective_weight=1.5)

    def test_label_resolution(self):
        """Test RDD class code resolution to human names and hazard weights."""
        name, canon, weight = self.analyzer.resolve_label("D40")
        self.assertEqual(name, "Pothole")
        self.assertEqual(canon, "pothole")
        self.assertEqual(weight, 1.0)

        name, canon, weight = self.analyzer.resolve_label("D20")
        self.assertEqual(name, "Alligator Crack")
        self.assertEqual(canon, "crack")
        self.assertEqual(weight, 0.85)

        name, canon, weight = self.analyzer.resolve_label("D10")
        self.assertEqual(name, "Transverse Crack")
        self.assertEqual(weight, 0.65)

        name, canon, weight = self.analyzer.resolve_label("D00")
        self.assertEqual(name, "Longitudinal Crack")
        self.assertEqual(weight, 0.50)

    def test_perspective_factor(self):
        """Test that distant defects get higher perspective scaling than near defects."""
        img_h = 1000
        near_factor = self.analyzer.compute_perspective_factor(950, img_h)  # near bumper
        mid_factor = self.analyzer.compute_perspective_factor(500, img_h)   # middle road
        far_factor = self.analyzer.compute_perspective_factor(250, img_h)   # distant road

        self.assertGreater(far_factor, mid_factor)
        self.assertGreater(mid_factor, near_factor)
        self.assertGreaterEqual(near_factor, 1.0)

    def test_empty_detections_report(self):
        """Test that a clean road with zero detections has PCI 100 and Good status."""
        report = self.analyzer.assess_image([], (600, 800))
        self.assertEqual(report.total_defects, 0)
        self.assertEqual(report.pavement_condition_index, 100.0)
        self.assertEqual(report.road_condition, RoadCondition.GOOD)
        self.assertEqual(len(report.defects), 0)

    def test_severe_pothole_assessment(self):
        """Test that a large pothole is graded as Critical or High with appropriate deduct."""
        # 800x600 image, pothole of 200x150 pixels in foreground
        img_shape = (600, 800)
        detections = [
            {
                "box": [300, 350, 500, 500],
                "confidence": 0.92,
                "class": "D40",
            }
        ]
        report = self.analyzer.assess_image(detections, img_shape)
        self.assertEqual(report.total_defects, 1)
        self.assertEqual(report.pothole_count, 1)
        self.assertEqual(report.crack_count, 0)
        self.assertIn(report.defects[0].severity_grade, [SeverityGrade.HIGH, SeverityGrade.CRITICAL])
        self.assertLess(report.pavement_condition_index, 85.0)

    def test_multi_defect_pci_deduct(self):
        """Test multiple defect deductions produce a cumulative degraded PCI score."""
        img_shape = (600, 800)
        detections = [
            {"box": [200, 200, 250, 400], "confidence": 0.88, "class": "D00"},
            {"box": [400, 300, 500, 350], "confidence": 0.85, "class": "D10"},
            {"box": [100, 400, 300, 550], "confidence": 0.95, "class": "D40"},
            {"box": [500, 420, 700, 580], "confidence": 0.91, "class": "D20"},
        ]
        report = self.analyzer.assess_image(detections, img_shape)
        self.assertEqual(report.total_defects, 4)
        self.assertEqual(report.pothole_count, 1)
        self.assertEqual(report.crack_count, 3)
        self.assertLess(report.pavement_condition_index, 60.0)
        self.assertIn(report.road_condition, [RoadCondition.POOR, RoadCondition.CRITICAL])


if __name__ == "__main__":
    unittest.main()

"""Юнит-тесты состояния колец (шаг 2 → шаг 4) и prepare_rings_for_drawing."""
from __future__ import annotations

import unittest

import numpy as np

from app import (
    any_ring_configured,
    apply_default_ring_positions,
    compute_dynamic_rings,
    draw_text_on_bgr,
    ensure_ring_zones_for_video,
    mark_ring_configured,
    prepare_rings_for_drawing,
    rings_from_session_state,
    sync_ring_widgets_to_canonical,
    test_draw_hoop_lines_on_frame,
    transform_ring_to_frame,
)


def _translation_transforms(n: int, dx_per_frame: float) -> list:
    transforms = []
    for i in range(n):
        t = np.eye(3, dtype=np.float64)
        t[0, 2] = i * dx_per_frame
        transforms.append(t)
    return transforms


class RingSessionFlowTests(unittest.TestCase):
    def test_configured_rings_survive_step4_prepare(self) -> None:
        state = {
            "ring1_x": 120,
            "ring1_y": 250,
            "ring1_r": 55,
            "ring1_configured": True,
            "ring2_x": 1800,
            "ring2_y": 260,
            "ring2_r": 60,
            "ring2_configured": True,
            "rings_initialized_for": "/tmp/fake.mp4",
        }
        rings = rings_from_session_state(state)
        prepared, warnings = prepare_rings_for_drawing(rings, 1920, 1080)
        self.assertEqual(len(prepared), 2)
        self.assertEqual(warnings, [])
        self.assertAlmostEqual(prepared[0]["y"], 250.0)
        self.assertAlmostEqual(prepared[1]["x"], 1800.0)

    def test_unconfigured_rings_warn_not_draw(self) -> None:
        state = {
            "ring1_x": 100,
            "ring1_y": 200,
            "ring1_r": 40,
            "ring1_configured": False,
            "ring2_x": 900,
            "ring2_y": 200,
            "ring2_r": 40,
            "ring2_configured": False,
        }
        rings = rings_from_session_state(state)
        prepared, warnings = prepare_rings_for_drawing(rings, 1280, 720)
        self.assertEqual(prepared, [])
        self.assertTrue(any("не заданы" in w.lower() for w in warnings))

    def test_ensure_ring_zones_does_not_overwrite_configured(self) -> None:
        state = {
            "ring1_x": 333,
            "ring1_y": 444,
            "ring1_r": 50,
            "ring1_configured": True,
            "ring2_x": 0,
            "ring2_y": 0,
            "ring2_r": 40,
            "ring2_configured": False,
            "rings_initialized_for": None,
        }
        ensure_ring_zones_for_video("/other/video.mp4", state)
        self.assertEqual(state["ring1_x"], 333)
        self.assertEqual(state["ring1_y"], 444)
        self.assertTrue(state["ring1_configured"])

    def test_click_syncs_widget_and_canonical_keys(self) -> None:
        state = {
            "ring1_x": 0,
            "ring1_y": 0,
            "ring1_r": 40,
            "ring1_configured": False,
            "ring2_x": 0,
            "ring2_y": 0,
            "ring2_r": 40,
            "ring2_configured": False,
        }
        mark_ring_configured(state, 1, 512, 288)
        self.assertTrue(state["ring1_configured"])
        self.assertEqual(state["wi_ring1_x"], 512)
        self.assertEqual(state["wi_ring1_y"], 288)
        state["wi_ring1_r"] = 70
        sync_ring_widgets_to_canonical(state, 1)
        self.assertEqual(state["ring1_r"], 70)

    def test_defaults_do_not_mark_configured(self) -> None:
        state = {}
        apply_default_ring_positions(state, 1920, 1080)
        self.assertFalse(any_ring_configured(state))
        self.assertGreater(state["ring1_y"], 0)
        self.assertIn("wi_ring1_x", state)

    def test_anchor_frame_warp_to_target(self) -> None:
        transforms = _translation_transforms(20, 10.0)
        ring = {
            "x": 100.0,
            "y": 200.0,
            "half_width": 50.0,
            "anchor_frame": 5,
            "configured": True,
        }
        at_anchor = transform_ring_to_frame(ring, transforms, 5)
        self.assertAlmostEqual(at_anchor["x"], 100.0)
        self.assertAlmostEqual(at_anchor["y"], 200.0)
        at_target = transform_ring_to_frame(ring, transforms, 10)
        self.assertAlmostEqual(at_target["x"], 150.0)
        self.assertAlmostEqual(at_target["y"], 200.0)

    def test_compute_dynamic_rings_per_ring_anchor(self) -> None:
        transforms = _translation_transforms(30, 8.0)
        rings = [
            {"x": 80.0, "y": 120.0, "half_width": 40.0, "anchor_frame": 10, "configured": True},
            {"x": 400.0, "y": 130.0, "half_width": 45.0, "anchor_frame": 20, "configured": True},
        ]
        out = compute_dynamic_rings(rings, transforms, 25)
        self.assertAlmostEqual(out[0]["x"], 80.0 + (25 - 10) * 8.0)
        self.assertAlmostEqual(out[1]["x"], 400.0 + (25 - 20) * 8.0)

    def test_mark_ring_stores_anchor_frame(self) -> None:
        state = {"ring1_x": 0, "ring1_y": 0, "ring1_r": 40, "ring1_configured": False}
        mark_ring_configured(state, 1, 300, 180, frame_idx=412)
        self.assertEqual(state["ring1_frame"], 412)


class DrawingTests(unittest.TestCase):
    def test_hoop_line_pixels(self) -> None:
        self.assertTrue(test_draw_hoop_lines_on_frame())

    def test_cyrillic_text_changes_pixels(self) -> None:
        frame = np.zeros((120, 400, 3), dtype=np.uint8)
        draw_text_on_bgr(frame, "Кольцо 1", (10, 80), font_size=24, color_bgr=(255, 255, 0))
        self.assertTrue(bool(np.any(frame)))


if __name__ == "__main__":
    unittest.main()

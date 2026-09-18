"""Юнит-тесты состояния колец (шаг 2 → шаг 4) и prepare_rings_for_drawing."""
from __future__ import annotations

import unittest

import numpy as np

from app import (
    any_ring_configured,
    apply_default_ring_positions,
    draw_text_on_bgr,
    ensure_ring_zones_for_video,
    mark_ring_configured,
    prepare_rings_for_drawing,
    rings_from_session_state,
    sync_ring_widgets_to_canonical,
    test_draw_hoop_lines_on_frame,
)


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


class DrawingTests(unittest.TestCase):
    def test_hoop_line_pixels(self) -> None:
        self.assertTrue(test_draw_hoop_lines_on_frame())

    def test_cyrillic_text_changes_pixels(self) -> None:
        frame = np.zeros((120, 400, 3), dtype=np.uint8)
        draw_text_on_bgr(frame, "Кольцо 1", (10, 80), font_size=24, color_bgr=(255, 255, 0))
        self.assertTrue(bool(np.any(frame)))


if __name__ == "__main__":
    unittest.main()

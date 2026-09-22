"""Регрессии спринта A: F01/F03/F04/F05/F08."""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from app import (
    BALL_CSRT_SEED_TAIL_FRAMES,
    BallAnchorGuide,
    BallTrackState,
    BallTracker,
    build_box_score,
    ball_state_counts_for_events,
    consume_pending_step2_ui_state,
    consume_pending_step3_ui_state,
    effective_prev_event_ball_xy,
    event_ball_from_tracking,
    mark_ring_configured,
    prepare_rings_for_drawing,
    rings_from_session_state,
    segment_crosses_hoop_line_top_to_bottom,
    sync_ring_widgets_to_canonical,
)


class RingClickPendingTests(unittest.TestCase):
    def test_static_click_pending_survives_widget_sync(self) -> None:
        state = {
            "ring1_x": 64,
            "ring1_y": 168,
            "ring1_r": 40,
            "ring1_configured": False,
            "wi_ring1_x": 64,
            "wi_ring1_y": 168,
            "wi_ring1_r": 40,
        }
        mark_ring_configured(state, 1, 300, 120)
        consume_pending_step2_ui_state(state)
        sync_ring_widgets_to_canonical(state, 1)
        self.assertEqual(state["ring1_x"], 300)
        self.assertEqual(state["ring1_y"], 120)
        self.assertEqual(state["wi_ring1_x"], 300)
        self.assertEqual(state["wi_ring1_y"], 120)


class EventBallGapTests(unittest.TestCase):
    def test_gap_resets_prev_event_ball(self) -> None:
        prev = (100.0, 200.0)
        self.assertIsNone(effective_prev_event_ball_xy(prev, 0, 99, 9))
        self.assertEqual(effective_prev_event_ball_xy(prev, 0, 5, 9), prev)

    def test_ninety_nine_frame_gap_does_not_score_goal(self) -> None:
        ring = {"x": 320.0, "y": 120.0, "half_width": 80.0, "configured": True}
        prev = effective_prev_event_ball_xy((320.0, 80.0), 0, 99, 9)
        self.assertIsNone(prev)
        if prev is not None:
            crossed = segment_crosses_hoop_line_top_to_bottom(
                prev[0], prev[1], 320.0, 160.0, ring["y"], ring["x"], ring["half_width"]
            )
            self.assertFalse(crossed)

    def test_kalman_ball_does_not_clear_persons_path(self) -> None:
        kalman = BallTrackState(10.0, 20.0, 0.0, "kalman")
        self.assertIsNone(event_ball_from_tracking(SimpleNamespace(), kalman, False, (10.0, 20.0)))
        yolo = BallTrackState(10.0, 20.0, 0.8, "yolo")
        self.assertEqual(event_ball_from_tracking(SimpleNamespace(), yolo, False, (10.0, 20.0)), (10.0, 20.0))

    def test_user_interp_not_event_eligible(self) -> None:
        state = BallTrackState(100.0, 200.0, 0.92, "user_interp")
        self.assertFalse(ball_state_counts_for_events(state))


class BallAnchorTrackerTests(unittest.TestCase):
    def test_yolo_wins_between_user_anchors(self) -> None:
        anchors = [
            {"x": 0.0, "y": 0.0, "frame": 0},
            {"x": 1000.0, "y": 0.0, "frame": 1000},
        ]
        guide = BallAnchorGuide(anchors)
        tracker = BallTracker(model=mock.Mock(), device="cpu")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        hit = tracker.update(
            500,
            frame,
            (400.0, 400.0),
            0.99,
            [],
            ball_anchor_guide=guide,
        )
        self.assertIsNotNone(hit)
        self.assertEqual(hit.source, "yolo")
        self.assertAlmostEqual(hit.x, 400.0)
        self.assertAlmostEqual(hit.y, 400.0)

    def test_csrt_seed_only_near_anchor_frame(self) -> None:
        anchors = [{"x": 50.0, "y": 60.0, "frame": 10}]
        guide = BallAnchorGuide(anchors)
        self.assertIsNotNone(guide.get_csrt_seed_bbox(10))
        self.assertIsNotNone(guide.get_csrt_seed_bbox(10 + BALL_CSRT_SEED_TAIL_FRAMES))
        self.assertIsNone(guide.get_csrt_seed_bbox(10 + BALL_CSRT_SEED_TAIL_FRAMES + 5))


class ConfiguredRingGoalTests(unittest.TestCase):
    def test_unconfigured_ring_excluded_from_prepared(self) -> None:
        state = {
            "ring1_x": 120,
            "ring1_y": 250,
            "ring1_r": 55,
            "ring1_configured": True,
            "ring2_x": 1800,
            "ring2_y": 260,
            "ring2_r": 60,
            "ring2_configured": False,
            "ring1_anchors": [],
            "ring2_anchors": [],
        }
        rings = rings_from_session_state(state)
        prepared, _ = prepare_rings_for_drawing(rings, 1920, 1080)
        self.assertEqual(len(prepared), 1)
        self.assertAlmostEqual(prepared[0]["x"], 120.0)


class BoxScoreTests(unittest.TestCase):
    def test_no_shots_or_accuracy_columns(self) -> None:
        df = build_box_score({7: {"shots": 3, "makes": 2, "passes": 1}})
        self.assertEqual(list(df.columns), ["ID игрока", "Имя", "Номер", "Попадания", "Сделано передач"])
        self.assertEqual(int(df.iloc[0]["Попадания"]), 2)


class MergePendingTests(unittest.TestCase):
    def test_consume_pending_clears_merge_pick_before_widgets(self) -> None:
        state = {
            "merge_pick_3": True,
            "merge_pick_7": True,
            "_pending_merge_pick_clear": [3, 7],
        }
        consume_pending_step3_ui_state(state)
        self.assertFalse(state["merge_pick_3"])
        self.assertFalse(state["merge_pick_7"])
        self.assertNotIn("_pending_merge_pick_clear", state)


if __name__ == "__main__":
    unittest.main()

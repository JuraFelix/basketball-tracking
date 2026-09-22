"""Тесты parse_track_results (мяч без track id) и фильтра событий по source мяча."""
from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from app import (
    COCO_BALL_CLASS_ID,
    COCO_PERSON_CLASS_ID,
    BallTrackState,
    ball_state_counts_for_events,
    parse_track_results,
)


class _FakeTensor:
    def __init__(self, arr):
        self._arr = np.asarray(arr)

    def cpu(self):
        return self

    def numpy(self):
        return self._arr


class _FakeBoxes:
    def __init__(self, xyxy, cls, conf, ids=None):
        self.xyxy = _FakeTensor(xyxy)
        self.cls = _FakeTensor(cls)
        self.conf = _FakeTensor(conf)
        self.id = _FakeTensor(ids) if ids is not None else None

    def __len__(self):
        return len(self.xyxy._arr)


class ParseTrackResultsTests(unittest.TestCase):
    def test_ball_detected_without_track_id(self) -> None:
        """1.6: мяч парсится по классу/conf даже если ByteTrack не выдал boxes.id."""
        boxes = _FakeBoxes(
            xyxy=[[40.0, 50.0, 60.0, 70.0]],
            cls=[COCO_BALL_CLASS_ID],
            conf=[0.42],
            ids=None,
        )
        result = SimpleNamespace(boxes=boxes)
        persons, ball, conf, bbox = parse_track_results([result], ball_conf_threshold=0.2)
        self.assertEqual(persons, [])
        self.assertIsNotNone(ball)
        self.assertAlmostEqual(ball[0], 50.0)
        self.assertAlmostEqual(ball[1], 60.0)
        self.assertAlmostEqual(conf, 0.42)
        self.assertIsNotNone(bbox)

    def test_person_requires_track_id(self) -> None:
        boxes = _FakeBoxes(
            xyxy=[[10.0, 20.0, 30.0, 80.0]],
            cls=[COCO_PERSON_CLASS_ID],
            conf=[0.9],
            ids=None,
        )
        result = SimpleNamespace(boxes=boxes)
        persons, ball, _, _ = parse_track_results([result])
        self.assertEqual(persons, [])
        self.assertIsNone(ball)

    def test_person_with_id_and_ball_without_id(self) -> None:
        boxes = _FakeBoxes(
            xyxy=[
                [10.0, 20.0, 30.0, 80.0],
                [100.0, 110.0, 120.0, 130.0],
            ],
            cls=[COCO_PERSON_CLASS_ID, COCO_BALL_CLASS_ID],
            conf=[0.8, 0.35],
            ids=[7],
        )
        result = SimpleNamespace(boxes=boxes)
        persons, ball, conf, _ = parse_track_results([result], ball_conf_threshold=0.2)
        self.assertEqual(len(persons), 1)
        self.assertEqual(persons[0][0], 7)
        self.assertIsNotNone(ball)
        self.assertAlmostEqual(conf, 0.35)


class BallEventSourceTests(unittest.TestCase):
    def test_kalman_not_eligible_for_events(self) -> None:
        state = BallTrackState(100.0, 200.0, 0.0, "kalman")
        self.assertFalse(ball_state_counts_for_events(state))

    def test_color_not_eligible_for_events(self) -> None:
        state = BallTrackState(100.0, 200.0, 0.25, "color")
        self.assertFalse(ball_state_counts_for_events(state))

    def test_interp_not_eligible_for_events(self) -> None:
        state = BallTrackState(100.0, 200.0, 0.0, "interp")
        self.assertFalse(ball_state_counts_for_events(state))

    def test_yolo_eligible(self) -> None:
        state = BallTrackState(100.0, 200.0, 0.55, "yolo")
        self.assertTrue(ball_state_counts_for_events(state))

    def test_user_interp_not_eligible(self) -> None:
        state = BallTrackState(100.0, 200.0, 0.92, "user_interp")
        self.assertFalse(ball_state_counts_for_events(state))

    def test_user_anchor_eligible(self) -> None:
        state = BallTrackState(100.0, 200.0, 1.0, "user")
        self.assertTrue(ball_state_counts_for_events(state))

    def test_lost_ball_not_eligible(self) -> None:
        state = BallTrackState(100.0, 200.0, 0.8, "yolo")
        self.assertFalse(ball_state_counts_for_events(state, ball_lost=True))


if __name__ == "__main__":
    unittest.main()

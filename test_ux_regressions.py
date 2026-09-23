"""Регрессии UX: превью шага 2, подписи групп, диагностика мяча на префиксе."""
from __future__ import annotations

import unittest
from unittest import mock

import cv2
import numpy as np

from app import (
    compute_quick_scan_frame_count,
    diagnose_ball_visibility,
    draw_annotations,
    extract_frame_at_index,
    format_player_video_label,
)


class PlayerVideoLabelTests(unittest.TestCase):
    def test_group_label_name_and_number(self) -> None:
        label = format_player_video_label(
            3,
            player_names={3: "Иван"},
            player_numbers={3: "7"},
            manual_id_map={12: 3, 46: 3},
        )
        self.assertEqual(label, "Иван #7")
        self.assertNotIn("12", label)
        self.assertNotIn("46", label)
        self.assertNotIn("=б.", label)

    def test_group_label_name_only(self) -> None:
        label = format_player_video_label(
            3, player_names={3: "Пётр"}, manual_id_map={12: 3}
        )
        self.assertEqual(label, "Пётр")

    def test_group_label_number_only(self) -> None:
        label = format_player_video_label(
            3, player_numbers={15: "23"}, manual_id_map={15: 3}
        )
        self.assertEqual(label, "#23")

    def test_group_label_empty_uses_canonical_id_only(self) -> None:
        label = format_player_video_label(3, manual_id_map={12: 3, 46: 3})
        self.assertEqual(label, "ID 3")
        self.assertNotIn("12", label)
        self.assertNotIn("46", label)

    def test_draw_annotations_uses_player_label_not_former_ids(self) -> None:
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        annotated = draw_annotations(
            frame,
            persons=[(3, (10.0, 20.0, 40.0, 80.0))],
            ball=None,
            player_names={3: "Саша"},
            player_numbers={3: "11"},
            manual_id_map={12: 3, 46: 3},
            id_former_labels={3: [12, 46]},
        )
        self.assertIsNotNone(annotated)


class BallVisibilityPrefixTests(unittest.TestCase):
    def test_diagnose_ball_visibility_scans_prefix_not_full_video(self) -> None:
        total_frames = 5000
        fps = 30.0
        max_seconds = 10.0
        expected_frames = compute_quick_scan_frame_count(fps, max_seconds)
        read_positions: list[int] = []

        class FakeCap:
            def __init__(self) -> None:
                self._pos = 0

            def isOpened(self) -> bool:
                return True

            def get(self, prop: int) -> float:
                if prop == cv2.CAP_PROP_FRAME_COUNT:
                    return float(total_frames)
                if prop == cv2.CAP_PROP_FPS:
                    return fps
                return 0.0

            def set(self, prop: int, value: float) -> bool:
                if prop == cv2.CAP_PROP_POS_FRAMES:
                    self._pos = int(value)
                return True

            def read(self):
                idx = self._pos
                read_positions.append(idx)
                if idx >= expected_frames:
                    return False, None
                self._pos += 1
                return True, np.zeros((64, 64, 3), dtype=np.uint8)

            def release(self) -> None:
                pass

        fake_cap = FakeCap()
        fake_model = object()

        def _fake_detect(*_args, **_kwargs):
            return None, 0.0

        with mock.patch("basketball.core.cv2.VideoCapture", return_value=fake_cap), mock.patch(
            "basketball.core.detect_yolo_ball_on_frame", side_effect=_fake_detect
        ), mock.patch("basketball.core.BallTracker.update", return_value=None):
            diag = diagnose_ball_visibility(
                "fake.mp4",
                fake_model,
                "cpu",
                ball_conf_threshold=0.2,
                imgsz=640,
                max_seconds=max_seconds,
                fps_hint=fps,
            )

        self.assertIsNotNone(diag)
        self.assertFalse(diag.get("scans_full_video", True))
        self.assertEqual(diag["segment_frames"], expected_frames)
        self.assertLessEqual(len(read_positions), expected_frames)
        self.assertLess(max(read_positions), total_frames)


class PreviewFrameCacheTests(unittest.TestCase):
    def test_extract_frame_cache_avoids_repeat_decode(self) -> None:
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        decode_calls = 0

        class FakeCap:
            def isOpened(self) -> bool:
                return True

            def set(self, *_args, **_kwargs) -> bool:
                return True

            def read(self):
                nonlocal decode_calls
                decode_calls += 1
                return True, frame

            def release(self) -> None:
                pass

        with mock.patch("basketball.core.cv2.VideoCapture", return_value=FakeCap()):
            first = extract_frame_at_index("/tmp/test.mp4", 0)
            second = extract_frame_at_index("/tmp/test.mp4", 0)

        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertEqual(decode_calls, 1)


if __name__ == "__main__":
    unittest.main()

"""Регрессии спринта B: F06/F07/F11/F12/F14."""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from app import (
    BallTrackState,
    BallTracker,
    compute_quick_scan_frame_count,
    detect_ball_yolo_on_crop,
    effective_prev_event_ball_xy,
    extract_reid_embedding,
    get_reid_error_message,
    quick_player_scan,
    reset_tracker,
    segment_crosses_hoop_line_top_to_bottom,
    tracker_callback_count,
    transform_point_between_frames,
)


def _translation_y_transforms(n: int, dy_per_frame: float) -> list:
    transforms = []
    for i in range(n):
        t = np.eye(3, dtype=np.float64)
        t[1, 2] = i * dy_per_frame
        transforms.append(t)
    return transforms


class F14CsrtGateTests(unittest.TestCase):
    def test_commit_detection_passes_previous_bbox_to_csrt_gate(self) -> None:
        tracker = BallTracker(model=mock.Mock(), device="cpu", ball_conf=0.18)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        prev_bbox = (100.0, 100.0, 128.0, 128.0)
        tracker.last_bbox = prev_bbox
        tracker.last_committed_conf = 0.8
        tracker.csrt_tracker = object()

        far_bbox = (300.0, 300.0, 328.0, 328.0)
        with mock.patch.object(tracker, "_should_reinit_csrt", wraps=tracker._should_reinit_csrt) as gate:
            tracker._commit_detection(1, frame, 314.0, 314.0, 0.82, "yolo", bbox=far_bbox)
            _, kwargs = gate.call_args
            self.assertEqual(kwargs.get("prev_bbox"), prev_bbox)

    def test_csrt_gate_uses_prev_bbox_not_self_iou(self) -> None:
        tracker = BallTracker(model=mock.Mock(), device="cpu", ball_conf=0.18)
        prev_bbox = (100.0, 100.0, 128.0, 128.0)
        far_bbox = (300.0, 300.0, 328.0, 328.0)
        tracker.last_bbox = far_bbox
        tracker.last_committed_conf = 0.9
        tracker.csrt_tracker = object()
        self.assertFalse(tracker._should_reinit_csrt(far_bbox, 0.82, "yolo", prev_bbox=prev_bbox))

    def test_weak_distant_roi_does_not_shift_last_confident_pos(self) -> None:
        tracker = BallTracker(model=mock.Mock(), device="cpu", ball_conf=0.18)
        tracker.last_committed_conf = 0.75
        tracker.last_confident_pos = (320.0, 240.0)
        tracker.last_confident_frame = 10
        tracker.csrt_max_jump_px = 80.0
        tracker.csrt_tracker = object()

        self.assertFalse(tracker._roi_detection_commits_confident(500.0, 500.0, 0.2))
        self.assertEqual(tracker.last_confident_pos, (320.0, 240.0))


class F06ReidDeviceTests(unittest.TestCase):
    def setUp(self) -> None:
        import app as app_module

        app_module._REID_MODEL = None
        app_module._REID_DEVICE = None
        app_module._REID_LAST_ERROR = None

    def test_get_reid_model_moves_weights_to_device(self) -> None:
        import app as app_module

        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")

        fake_model = mock.Mock()
        fake_model.eval.return_value = fake_model
        fake_model.to.return_value = fake_model
        with mock.patch("torchvision.models.mobilenet_v3_small", return_value=fake_model):
            with mock.patch("torchvision.models.MobileNet_V3_Small_Weights", mock.Mock()):
                with mock.patch("torch.nn.Identity", return_value=mock.Mock()):
                    app_module._REID_MODEL = None
                    app_module._REID_DEVICE = None
                    model = app_module._get_reid_model("cpu")
                    self.assertIs(model, fake_model)
                    fake_model.to.assert_called_once_with("cpu")
                    self.assertEqual(app_module._REID_DEVICE, "cpu")

    def test_reid_failure_sets_visible_error(self) -> None:
        import app as app_module

        with mock.patch("app._get_reid_model", return_value=mock.Mock()):
            with mock.patch("app.cv2.cvtColor", side_effect=RuntimeError("boom")):
                emb = extract_reid_embedding(np.zeros((40, 40, 3), dtype=np.uint8), (0, 0, 20, 20), device="cpu")
        self.assertIsNone(emb)
        self.assertIn("boom", get_reid_error_message() or "")


class F07TrackerCallbackTests(unittest.TestCase):
    def test_reset_tracker_clears_callbacks(self) -> None:
        model = SimpleNamespace(
            predictor=SimpleNamespace(
                callbacks={"on_predict_start": [lambda: None, lambda: None], "on_predict_end": [lambda: None]}
            )
        )
        self.assertEqual(tracker_callback_count(model), 3)
        reset_tracker(model)
        self.assertIsNone(model.predictor)
        reset_tracker(model)
        self.assertEqual(tracker_callback_count(model), 0)

    def test_repeated_reset_does_not_accumulate_callbacks(self) -> None:
        model = SimpleNamespace(predictor=None)

        def _attach_predictor() -> None:
            model.predictor = SimpleNamespace(
                callbacks={"on_predict_start": [lambda: None], "on_predict_end": [lambda: None]}
            )

        _attach_predictor()
        reset_tracker(model)
        _attach_predictor()
        reset_tracker(model)
        self.assertEqual(tracker_callback_count(model), 0)

    def test_ball_crop_predict_disables_trackers(self) -> None:
        saved_trackers = {"bytetrack": object()}
        predictor = SimpleNamespace(trackers=saved_trackers)
        model = SimpleNamespace(
            predictor=predictor,
            predict=mock.Mock(return_value=[SimpleNamespace(boxes=None)]),
        )
        crop = np.zeros((32, 32, 3), dtype=np.uint8)
        detect_ball_yolo_on_crop(crop, (0, 0), model, "cpu", 0.2, 640)
        model.predict.assert_called_once()
        self.assertIs(predictor.trackers, saved_trackers)


class F11ScanFramePrefixTests(unittest.TestCase):
    def test_scan_frame_count_caps_continuous_prefix(self) -> None:
        self.assertEqual(compute_quick_scan_frame_count(30.0, 15.0), 300)
        self.assertEqual(compute_quick_scan_frame_count(30.0, 5.0), 150)

    @mock.patch("app.parse_track_results", return_value=([], None, 0.0, None))
    @mock.patch("app.cv2.VideoCapture")
    def test_quick_scan_tracks_continuous_prefix_without_stride(
        self, mock_cap_cls: mock.Mock, _parse_mock: mock.Mock
    ) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap = mock_cap_cls.return_value
        mock_cap.isOpened.return_value = True
        mock_cap.read.side_effect = [(True, frame)] * 450 + [(False, None)]

        model = mock.Mock()
        model.track.return_value = [SimpleNamespace(boxes=None)]

        quick_player_scan("fake.mp4", model, "cpu", max_seconds=15.0, fps_hint=30.0)
        self.assertEqual(model.track.call_count, 300)


class F12GoalCoordinateTests(unittest.TestCase):
    def test_comoving_panorama_is_not_scored_as_goal(self) -> None:
        transforms = _translation_y_transforms(2, 20.0)
        bx_prev, by_prev = 320.0, 60.0
        bx_curr, by_curr = 320.0, 80.0
        ring_y = 70.0

        tx, ty = transform_point_between_frames(bx_prev, by_prev, 0, 1, transforms)
        self.assertAlmostEqual(ty, 80.0)

        goal_prev = effective_prev_event_ball_xy(
            (bx_prev, by_prev), 0, 1, gap_max_frames=5, camera_transforms=transforms
        )
        self.assertIsNotNone(goal_prev)
        self.assertFalse(
            segment_crosses_hoop_line_top_to_bottom(
                goal_prev[0],
                goal_prev[1],
                bx_curr,
                by_curr,
                ring_y,
                320.0,
                80.0,
            )
        )

    def test_unwarped_prev_ball_false_positive_documented(self) -> None:
        self.assertTrue(
            segment_crosses_hoop_line_top_to_bottom(320.0, 60.0, 320.0, 80.0, 70.0, 320.0, 80.0)
        )


if __name__ == "__main__":
    unittest.main()

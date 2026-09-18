"""Тесты ручной склейки ID, проверки интерполяции мяча и training seeds."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from app import (
    apply_manual_id_merge,
    append_ball_training_seed,
    build_id_former_labels,
    get_pending_ball_interp_checks,
    make_id_resolver,
    suggest_ball_interp_check_frames,
)


class ManualIdMergeTests(unittest.TestCase):
    def test_resolver_follows_chain(self) -> None:
        resolve = make_id_resolver({12: 3, 15: 3, 3: 7})
        self.assertEqual(resolve(12), 7)
        self.assertEqual(resolve(15), 7)
        self.assertEqual(resolve(7), 7)

    def test_apply_manual_id_merge_updates_state(self) -> None:
        state = {
            "manual_id_map": {},
            "manual_id_merge_log": [],
            "player_crops": {3: np.zeros((10, 10, 3), dtype=np.uint8), 12: np.ones((10, 10, 3), dtype=np.uint8)},
            "player_names": {12: "Иван"},
            "player_numbers": {},
        }
        apply_manual_id_merge(state, 3, [12, 15])
        self.assertEqual(state["manual_id_map"][12], 3)
        self.assertEqual(state["manual_id_map"][15], 3)
        self.assertEqual(state["player_names"][3], "Иван")

    def test_build_id_former_labels(self) -> None:
        labels = build_id_former_labels({12: 3, 15: 3})
        self.assertEqual(labels[3], [12, 15])


class BallInterpCheckTests(unittest.TestCase):
    def test_suggest_frames_between_anchors(self) -> None:
        anchors = [{"frame": 0, "x": 10.0, "y": 20.0}, {"frame": 30, "x": 100.0, "y": 120.0}]
        frames = suggest_ball_interp_check_frames(anchors, max_per_gap=3)
        self.assertTrue(all(0 < f < 30 for f in frames))
        self.assertGreaterEqual(len(frames), 1)
        self.assertLessEqual(len(frames), 3)

    def test_pending_skips_anchors_and_skipped(self) -> None:
        anchors = [{"frame": 0, "x": 1.0, "y": 2.0}, {"frame": 20, "x": 3.0, "y": 4.0}]
        pending = get_pending_ball_interp_checks(anchors, skipped_frames=[])
        first = pending[0]
        skipped = get_pending_ball_interp_checks(anchors, skipped_frames=[first])
        self.assertNotIn(first, skipped)


class TrainingSeedTests(unittest.TestCase):
    def test_append_ball_training_seed_writes_jsonl_and_crop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            with mock.patch("app.TRAINING_SEEDS_DIR", base / "training_seeds"), mock.patch(
                "app.BALL_TRAINING_CROPS_DIR", base / "training_seeds" / "crops"
            ), mock.patch("app.BALL_LABELS_JSONL", base / "training_seeds" / "ball_labels.jsonl"), mock.patch(
                "app.BASE_DIR", base
            ):
                frame = np.zeros((100, 120, 3), dtype=np.uint8)
                append_ball_training_seed("/tmp/test.mp4", "test", 5, 60.0, 40.0, frame, source="user_click")
                jsonl_path = base / "training_seeds" / "ball_labels.jsonl"
                self.assertTrue(jsonl_path.exists())
                record = json.loads(jsonl_path.read_text(encoding="utf-8").strip())
                self.assertEqual(record["frame_idx"], 5)
                self.assertEqual(record["source"], "user_click")
                crops = list((base / "training_seeds" / "crops").glob("*.jpg"))
                self.assertEqual(len(crops), 1)


if __name__ == "__main__":
    unittest.main()

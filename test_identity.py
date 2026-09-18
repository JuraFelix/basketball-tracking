"""Тесты стабилизации ID (anti-swap) поверх ByteTrack."""
from __future__ import annotations

import unittest

import numpy as np

from app import (
    APPEARANCE_SWAP_GATE,
    AppearanceMerger,
    _DetectionFeatures,
    _TrackProfile,
    bbox_center,
)


def _make_hist(peak_h: int) -> np.ndarray:
    hist = np.zeros(16 * 16, dtype=np.float32)
    hist[peak_h] = 1.0
    return hist


class IdentityStabilizerTests(unittest.TestCase):
    def test_jersey_conflict_blocks_match(self) -> None:
        merger = AppearanceMerger(similarity_threshold=0.5, jersey_ocr_enabled=True)
        profile = _TrackProfile(canonical_id=1, jersey_number="7", last_seen_frame=0)
        merger.profiles[1] = profile
        det = _DetectionFeatures(
            raw_id=9,
            box=(10.0, 10.0, 50.0, 90.0),
            center=bbox_center((10.0, 10.0, 50.0, 90.0)),
            hist=None,
            emb=None,
            jersey="11",
            area=1600.0,
        )
        self.assertEqual(merger._match_score(det, profile, frame_idx=1), -1.0)

    def test_swap_correction_swaps_two_player_assignment(self) -> None:
        merger = AppearanceMerger(similarity_threshold=0.5)
        h_a, h_b = _make_hist(10), _make_hist(200)
        merger.profiles[1] = _TrackProfile(
            canonical_id=1,
            histogram=h_a,
            last_center=(100.0, 100.0),
            last_box=(80.0, 60.0, 120.0, 140.0),
            last_seen_frame=0,
            velocity=(0.0, 0.0),
        )
        merger.profiles[2] = _TrackProfile(
            canonical_id=2,
            histogram=h_b,
            last_center=(300.0, 100.0),
            last_box=(280.0, 60.0, 320.0, 140.0),
            last_seen_frame=0,
            velocity=(0.0, 0.0),
        )
        merger.prev_active_ids = [1, 2]
        dets = [
            _DetectionFeatures(10, (280.0, 60.0, 320.0, 140.0), (300.0, 100.0), h_b, None, None, 1600.0),
            _DetectionFeatures(11, (80.0, 60.0, 120.0, 140.0), (100.0, 100.0), h_a, None, None, 1600.0),
        ]
        assignments = merger._greedy_assign(dets, [1, 2], frame_idx=1, min_score=0.5)
        fixed = merger._correct_pair_swap(dets, assignments, frame_idx=1)
        self.assertEqual(fixed.get(0), 2)
        self.assertEqual(fixed.get(1), 1)

    def test_never_merge_two_active_in_same_frame(self) -> None:
        merger = AppearanceMerger(similarity_threshold=0.5)
        h = _make_hist(50)
        merger.profiles[1] = _TrackProfile(canonical_id=1, histogram=h, last_center=(50, 50), last_seen_frame=0)
        merger.profiles[2] = _TrackProfile(canonical_id=2, histogram=h, last_center=(200, 50), last_seen_frame=0)
        merger.prev_active_ids = [1, 2]
        frame = np.zeros((200, 400, 3), dtype=np.uint8)
        persons = [(1, (40, 20, 80, 120)), (2, (180, 20, 220, 120))]
        remapped = merger.remap(1, frame, persons)
        ids = [pid for pid, _ in remapped]
        self.assertEqual(len(ids), len(set(ids)))


if __name__ == "__main__":
    unittest.main()

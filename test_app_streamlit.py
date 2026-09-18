"""AppTest сценарии шага 2: клики по кольцам и «Поправить кликом» без StreamlitAPIException."""
from __future__ import annotations

import tempfile
import unittest
from typing import Any, Dict, List

import numpy as np
from streamlit.testing.v1 import AppTest

from app import (
    CAMERA_MODE_PANNING,
    add_ring_anchor,
    build_player_id_groups,
    consume_pending_step2_ui_state,
    create_synthetic_pan_video,
    merge_player_ids_selection,
    sync_ring_widgets_to_canonical,
    unmerge_player_group,
)


def _translation_transforms(n: int, dx: float = 5.0) -> List[np.ndarray]:
    transforms: List[np.ndarray] = []
    for i in range(n):
        t = np.eye(3, dtype=np.float64)
        t[0, 2] = i * dx
        transforms.append(t)
    return transforms


def _step2_session(video_path: str, transforms: List[np.ndarray]) -> Dict[str, Any]:
    return {
        "step": 2,
        "video_path": video_path,
        "video_name": "test.mp4",
        "camera_mode": CAMERA_MODE_PANNING,
        "rings_initialized_for": video_path,
        "camera_transforms_video": video_path,
        "camera_transforms": transforms,
        "auto_threshold_computed_for": video_path,
        "click_target_ring": "Кольцо 1",
        "click_target_ring_radio": "Кольцо 1",
        "ring1_configured": False,
        "ring2_configured": False,
        "ring1_anchors": [],
        "ring2_anchors": [],
        "ball_anchors": [],
        "ball_interp_skipped": [],
        "preview_frame_idx": 0,
    }


class Step2AppTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmpdir = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        cls.video_path = cls._tmpdir.name
        create_synthetic_pan_video(cls.video_path, n_frames=30, dx_per_frame=5.0)
        cls.transforms = _translation_transforms(30)

    def _open_step2(self) -> AppTest:
        at = AppTest.from_file("app.py", default_timeout=30)
        for key, value in _step2_session(self.video_path, self.transforms).items():
            at.session_state[key] = value
        return at

    def _assert_no_streamlit_exception(self, at: AppTest) -> None:
        self.assertFalse(
            at.exception,
            msg="; ".join(str(exc.value) for exc in at.exception),
        )

    def test_ring1_click_rerun_no_exception(self) -> None:
        """Первый клик по кольцу 1: якорь + rerun после виджетов wi_ring*_r — без краша."""
        at = self._open_step2()
        at.run()
        self._assert_no_streamlit_exception(at)

        wi_before = (
            at.session_state.get("wi_ring1_x"),
            at.session_state.get("wi_ring1_y"),
            at.session_state.get("wi_ring1_r"),
        )
        sync_ring_widgets_to_canonical(at.session_state, 1)
        add_ring_anchor(
            at.session_state,
            1,
            320,
            120,
            frame_idx=int(at.session_state["preview_frame_idx"]),
            half_width=float(at.session_state.get("ring1_r", 40)),
        )
        at.session_state["_last_ring_click_time"] = 1.0
        at.run()
        self._assert_no_streamlit_exception(at)
        self.assertEqual(len(at.session_state["ring1_anchors"]), 1)
        self.assertTrue(at.session_state["ring1_configured"])
        wi_after = (
            at.session_state.get("wi_ring1_x"),
            at.session_state.get("wi_ring1_y"),
            at.session_state.get("wi_ring1_r"),
        )
        self.assertEqual(wi_after, wi_before)

    def test_ring2_click_rerun_no_exception(self) -> None:
        """Клик по кольцу 2 в panning-режиме — без записи в wi_ring2_* после виджетов."""
        at = self._open_step2()
        at.session_state["click_target_ring"] = "Кольцо 2"
        at.session_state["click_target_ring_radio"] = "Кольцо 2"
        at.session_state["ring1_configured"] = True
        at.session_state["ring1_anchors"] = [
            {"x": 300.0, "y": 100.0, "half_width": 40.0, "frame": 0}
        ]
        at.run()
        self._assert_no_streamlit_exception(at)

        sync_ring_widgets_to_canonical(at.session_state, 2)
        add_ring_anchor(
            at.session_state,
            2,
            500,
            130,
            frame_idx=3,
            half_width=float(at.session_state.get("ring2_r", 40)),
        )
        at.run()
        self._assert_no_streamlit_exception(at)
        self.assertEqual(len(at.session_state["ring2_anchors"]), 1)
        self.assertEqual(int(at.session_state["ring2_anchors"][0]["frame"]), 3)

    def test_ball_interp_fix_button_no_exception(self) -> None:
        """«Поправить кликом» не пишет в preview_frame_idx после слайдера."""
        at = self._open_step2()
        at.session_state["ring1_configured"] = True
        at.session_state["ring1_anchors"] = [
            {"x": 320.0, "y": 100.0, "half_width": 40.0, "frame": 0}
        ]
        at.session_state["ball_anchors"] = [
            {"x": 50.0, "y": 200.0, "frame": 0},
            {"x": 500.0, "y": 200.0, "frame": 20},
        ]
        at.run()
        self._assert_no_streamlit_exception(at)

        fix_btn = next(b for b in at.button if b.label and "Поправить кликом" in b.label)
        fix_btn.click().run()
        self._assert_no_streamlit_exception(at)
        self.assertIsNotNone(at.session_state.get("ball_interp_fix_frame"))
        self.assertEqual(at.session_state.get("click_target_ring_radio"), "Мяч")


class Step2UiHelperTests(unittest.TestCase):
    def test_consume_pending_step2_ui_state_before_widgets(self) -> None:
        state = {
            "preview_frame_idx": 0,
            "click_target_ring": "Кольцо 1",
            "_pending_preview_frame_idx": 12,
            "_pending_click_target_ring": "Мяч",
        }
        consume_pending_step2_ui_state(state)
        self.assertEqual(state["preview_frame_idx"], 12)
        self.assertEqual(state["click_target_ring"], "Мяч")
        self.assertEqual(state["click_target_ring_radio"], "Мяч")
        self.assertNotIn("_pending_preview_frame_idx", state)


class Step3MergeHelperTests(unittest.TestCase):
    def test_build_and_merge_player_groups(self) -> None:
        groups = build_player_id_groups([3, 12, 15], {12: 3, 15: 3})
        self.assertEqual(groups[3], [3, 12, 15])

        state = {"manual_id_map": {}, "manual_id_merge_log": [], "player_names": {}}
        merge_player_ids_selection(state, [7, 9, 11])
        self.assertEqual(state["manual_id_map"][9], 7)
        self.assertEqual(state["manual_id_map"][11], 7)

        unmerge_player_group(state, 7)
        self.assertEqual(state["manual_id_map"], {})


if __name__ == "__main__":
    unittest.main()

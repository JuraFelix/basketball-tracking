"""Регрессии спринта C: F15/F16/F17/F18."""
from __future__ import annotations

import os
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from app import (
    DiskFrameBuffer,
    PendingHighlight,
    _upload_fingerprint,
    bbox_iou,
    clear_project_cache,
    create_highlight_frame_cache,
    save_highlight_clip,
    store_uploaded_video,
)
from basketball.offline import apply_offline_env
from basketball import config as bt_config
from basketball import core as bt_core


class F15UploadFingerprintTests(unittest.TestCase):
    def test_same_name_different_bytes_changes_fingerprint(self) -> None:
        first = SimpleNamespace(name="game.mp4", getvalue=lambda: b"video-a")
        second = SimpleNamespace(name="game.mp4", getvalue=lambda: b"video-b")
        self.assertNotEqual(_upload_fingerprint(first), _upload_fingerprint(second))

    def test_store_uploaded_video_replaces_path_on_same_name(self) -> None:
        state: dict = {"video_upload_key": None, "video_path": None, "video_name": None, "step": 2}
        uploads: list[str] = []

        def _fake_reset() -> None:
            state["rings_initialized_for"] = None

        with mock.patch("basketball.core.st.session_state", state), mock.patch(
            "basketball.core.reset_for_new_video", _fake_reset
        ), mock.patch("basketball.core.tempfile.NamedTemporaryFile") as tmp_cls:
            for payload in (b"first-video", b"second-video"):
                handle = mock.Mock()
                handle.name = uploads[-1] if uploads else "/tmp/old.mp4"
                handle.__enter__ = mock.Mock(return_value=handle)
                handle.__exit__ = mock.Mock(return_value=False)

                def _write(data: bytes, _payload=payload) -> None:
                    path = f"/tmp/upload_{len(uploads)}.mp4"
                    uploads.append(path)
                    handle.name = path

                handle.write.side_effect = _write
                tmp_cls.return_value = handle

                uploaded = SimpleNamespace(name="game.mp4", getvalue=lambda p=payload: p)
                changed = store_uploaded_video(uploaded)
                self.assertTrue(changed)

        self.assertEqual(state["video_name"], "game.mp4")
        self.assertEqual(len(uploads), 2)
        self.assertNotEqual(uploads[0], uploads[1])
        self.assertEqual(state["step"], 1)

    def test_store_uploaded_video_skips_identical_reupload(self) -> None:
        state: dict = {"video_upload_key": None, "video_path": "/tmp/existing.mp4", "video_name": "game.mp4", "step": 3}
        uploaded = SimpleNamespace(name="game.mp4", getvalue=lambda: b"same-bytes")
        state["video_upload_key"] = _upload_fingerprint(uploaded)
        with mock.patch("basketball.core.st.session_state", state), mock.patch(
            "basketball.core.tempfile.NamedTemporaryFile"
        ) as tmp_cls:
            self.assertFalse(store_uploaded_video(uploaded))
            tmp_cls.assert_not_called()
        self.assertEqual(state["step"], 3)


class F16DiskBufferTests(unittest.TestCase):
    def test_disk_frame_buffer_limits_files_not_ram(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "frames"
            buf = DiskFrameBuffer(3, cache)
            paths = []
            for i in range(5):
                frame = np.full((8, 8, 3), i, dtype=np.uint8)
                paths.append(buf.append(frame))
            self.assertEqual(len(list(cache.glob("*.jpg"))), 3)
            self.assertEqual(len(buf.snapshot_paths()), 3)
            buf.cleanup()
            self.assertFalse(cache.exists())

    def test_pending_highlight_uses_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p1 = Path(tmp) / "a.jpg"
            p2 = Path(tmp) / "b.jpg"
            import cv2

            cv2.imwrite(str(p1), np.zeros((4, 4, 3), dtype=np.uint8))
            cv2.imwrite(str(p2), np.zeros((4, 4, 3), dtype=np.uint8))
            highlight = PendingHighlight("clip.mp4", [p1], frames_needed=1)
            highlight.future_paths.append(p2)
            out = save_highlight_clip(highlight, fps=25.0, width=4, height=4)
            self.assertTrue(out.exists())
            self.assertGreater(out.stat().st_size, 0)


class F17OfflineTests(unittest.TestCase):
    def test_offline_env_flags(self) -> None:
        keys = (
            "STREAMLIT_BROWSER_GATHER_USAGE_STATS",
            "DO_NOT_TRACK",
            "YOLO_VERBOSE",
            "ULTRALYTICS_AUTOINSTALL",
            "ULTRALYTICS_OFFLINE",
        )
        with mock.patch.dict(os.environ, {}, clear=True):
            apply_offline_env()
            self.assertEqual(os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"], "false")
            self.assertEqual(os.environ["ULTRALYTICS_AUTOINSTALL"], "0")
            self.assertEqual(os.environ["ULTRALYTICS_OFFLINE"], "1")
            for key in keys:
                self.assertIn(key, os.environ)

    def test_clear_project_cache_removes_temp_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            with mock.patch.object(bt_config, "PROJECT_CACHE_DIR", base / ".cache"), mock.patch.object(
                bt_config, "UPLOAD_CACHE_DIR", base / ".cache" / "uploads"
            ), mock.patch.object(bt_config, "HIGHLIGHT_FRAME_CACHE_DIR", base / ".cache" / "highlight_frames"), mock.patch.object(
                bt_config, "OUTPUT_DIR", base / "output_videos"
            ), mock.patch.object(bt_config, "HIGHLIGHTS_DIR", base / "highlights"), mock.patch.object(
                bt_core, "PROJECT_CACHE_DIR", base / ".cache"
            ), mock.patch.object(bt_core, "UPLOAD_CACHE_DIR", base / ".cache" / "uploads"), mock.patch.object(
                bt_core, "HIGHLIGHT_FRAME_CACHE_DIR", base / ".cache" / "highlight_frames"
            ), mock.patch.object(bt_core, "OUTPUT_DIR", base / "output_videos"), mock.patch.object(
                bt_core, "HIGHLIGHTS_DIR", base / "highlights"
            ), mock.patch("basketball.core.st.cache_resource.clear"):
                (base / ".cache" / "uploads").mkdir(parents=True)
                (base / ".cache" / "uploads" / "x.mp4").write_bytes(b"x")
                cleared = clear_project_cache()
                self.assertTrue(any(".cache" in entry for entry in cleared))
                self.assertFalse((base / ".cache" / "uploads" / "x.mp4").exists())


class F18ModuleSplitTests(unittest.TestCase):
    def test_single_bbox_iou_export(self) -> None:
        a = (0.0, 0.0, 10.0, 10.0)
        b = (5.0, 5.0, 15.0, 15.0)
        self.assertGreater(bbox_iou(a, b), 0.0)
        self.assertEqual(bt_core.bbox_iou.__module__, "basketball.core")

    def test_highlight_cache_dir_under_project_cache(self) -> None:
        with mock.patch.object(bt_config, "HIGHLIGHT_FRAME_CACHE_DIR", bt_config.PROJECT_CACHE_DIR / "highlight_frames"):
            path = create_highlight_frame_cache("run123")
            self.assertIn("highlight_frames", str(path))


if __name__ == "__main__":
    unittest.main()

"""
Basketball Tracking Analytics
==============================

Полностью автономное (офлайн) приложение на Streamlit для аналитики баскетбольных
тренировок по видео со статичной камеры.

Возможности:
    * Детекция и трекинг игроков и мяча моделью YOLO11x (Ultralytics) с трекером
      BoT-SORT + ReID (игроки не имеют номеров на майках, поэтому идентификация
      строится на визуальных признаках формы/обуви, а не на OCR номеров).
    * Математическая детекция бросков/попаданий по заданной пользователем
      окружности (зона кольца) с кулдауном событий.
    * Детекция передач (пасов) на основе смены владельца мяча в ограниченном
      временном окне.
    * Автоматическая нарезка автономных MP4-хайлайтов (5 сек до события +
      2 сек после) с помощью OpenCV.
    * Итоговая статистика по игрокам (Box Score) и встроенный просмотр хайлайтов.

Запуск:
    streamlit run app.py

Первый запуск скачивает веса модели YOLO11x (~110 МБ) и веса ReID-модуля
трекера из интернета. Все последующие запуски полностью офлайн.
"""

from __future__ import annotations

import math
import os
import tempfile
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import yaml

# ---------------------------------------------------------------------------
# Защищённые импорты "тяжёлых" библиотек.
#
# torch и ultralytics требуют интернет (для скачивания весов при первом
# запуске) и, в идеале, GPU. Импортируем их в try/except, чтобы приложение
# не падало при старте в среде без GPU/интернета (например, при разработке
# или CI) — вместо краха пользователь увидит понятное предупреждение в GUI
# и приложение продолжит работать в демонстрационном режиме без реального
# распознавания. На рабочей машине с RTX 4080 обе библиотеки установлены
# штатно, и вся боевая логика ниже используется без каких-либо "заглушек".
# ---------------------------------------------------------------------------
try:
    import torch

    TORCH_IMPORT_ERROR: Optional[str] = None
except Exception as exc:  # pragma: no cover - защита от отсутствия torch
    torch = None  # type: ignore[assignment]
    TORCH_IMPORT_ERROR = str(exc)

try:
    from ultralytics import YOLO

    ULTRALYTICS_IMPORT_ERROR: Optional[str] = None
except Exception as exc:  # pragma: no cover - защита от отсутствия ultralytics
    YOLO = None  # type: ignore[assignment]
    ULTRALYTICS_IMPORT_ERROR = str(exc)

try:
    import cv2

    CV2_IMPORT_ERROR: Optional[str] = None
except Exception as exc:  # pragma: no cover - защита от отсутствия opencv
    cv2 = None  # type: ignore[assignment]
    CV2_IMPORT_ERROR = str(exc)


# ---------------------------------------------------------------------------
# Константы, пути и настройки по умолчанию
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
HIGHLIGHTS_DIR = BASE_DIR / "highlights"
OUTPUT_DIR = BASE_DIR / "output_videos"
TRACKER_CONFIG_PATH = BASE_DIR / "custom_botsort.yaml"

# Самая точная модель семейства YOLO11. Официальное имя весов в Ultralytics —
# "yolo11x.pt"; ниже пробуем сначала имя из технического задания, а при
# неудаче — официальный алиас, чтобы первый запуск прошёл гладко в любом случае.
MODEL_WEIGHTS_PRIMARY = "yolov11x.pt"
MODEL_WEIGHTS_FALLBACK = "yolo11x.pt"

# Идентификаторы классов COCO, которые нас интересуют.
COCO_PERSON_CLASS_ID = 0
COCO_BALL_CLASS_ID = 32  # 'sports ball'

# Буфер хайлайтов: сколько секунд "прошлого" и "будущего" видео сохранять.
BUFFER_SECONDS = 5.0
FUTURE_SECONDS = 2.0

# Через сколько секунд после попадания мяча в зону кольца выносить вердикт
# "гол/промах" по траектории мяча.
SHOT_VERDICT_DELAY_SECONDS = 0.6

# Сколько последних точек траектории мяча хранить для математической оценки
# направления полёта (используется и для пасов, и для вердикта у кольца).
BALL_HISTORY_MAXLEN = 20

# Минимальное время "в воздухе"/переключения владения, ниже которого пас не
# засчитывается (защита от дребезга детекций одного кадра).
PASS_MIN_TIME_SECONDS = 0.0


PlayerStats = Dict[str, int]


def blank_stats() -> PlayerStats:
    """Пустая статистика для нового игрока."""
    return {"shots": 0, "makes": 0, "passes": 0}


# ---------------------------------------------------------------------------
# Инициализация окружения (папки, конфиг трекера)
# ---------------------------------------------------------------------------
def ensure_directories() -> None:
    """Создаёт папки highlights/ и output_videos/, если их ещё нет."""
    HIGHLIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def ensure_tracker_config(path: Path = TRACKER_CONFIG_PATH) -> Path:
    """Генерирует локальный конфиг трекера BoT-SORT при первом запуске.

    Игроки играют в разноцветной форме без номеров на майках, поэтому
    надёжный трекинг возможен только за счёт ReID (повторного распознавания
    по внешнему виду — цвету формы и обуви), а не по номеру. Поэтому ниже
    принудительно включён with_reid=True и подобраны пороги, устойчивые к
    быстрым перемещениям и частичным перекрытиям игроков.
    """
    if path.exists():
        return path

    tracker_cfg = {
        "tracker_type": "botsort",
        "track_high_thresh": 0.3,
        "track_low_thresh": 0.1,
        "new_track_thresh": 0.4,
        "track_buffer": 120,
        "match_thresh": 0.7,
        "fuse_score": True,
        "gmc_method": "sparseOptFlow",
        # --- ReID по внешнему виду (цвет формы/обуви) вместо номеров ---
        "with_reid": True,
        "proximity_thresh": 0.5,
        "appearance_thresh": 0.25,
    }
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            "# Автоматически сгенерированный конфиг трекера BoT-SORT.\n"
            "# with_reid=True включает повторную идентификацию игроков по\n"
            "# внешнему виду формы/обуви, т.к. номеров на майках нет.\n"
        )
        yaml.safe_dump(tracker_cfg, f, sort_keys=False, allow_unicode=True)
    return path


def resolve_device() -> Tuple[str, str, bool]:
    """Проверяет доступность CUDA и возвращает (device, сообщение, ok).

    Если CUDA доступна — принудительно используем 'cuda'. Если нет — выводим
    предупреждение (обработка при этом всё равно возможна на CPU, но будет
    заметно медленнее).
    """
    if torch is None:
        return (
            "cpu",
            f"⚠️ PyTorch не установлен или не импортируется ({TORCH_IMPORT_ERROR}). "
            "Установите зависимости из requirements.txt (см. README.md).",
            False,
        )
    if torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            gpu_name = "NVIDIA GPU"
        return (
            "cuda",
            f"✅ Обнаружена CUDA-видеокарта: {gpu_name}. Инференс принудительно "
            "выполняется на устройстве 'cuda'.",
            True,
        )
    return (
        "cpu",
        "⚠️ CUDA недоступна (нет GPU или NVIDIA-драйверов/CUDA-версии PyTorch). "
        "Обработка будет выполняться на CPU и может быть значительно медленнее. "
        "Для полноценной работы на RTX 4080 установите CUDA-версию PyTorch — см. README.md.",
        False,
    )


@st.cache_resource(show_spinner="Загрузка модели YOLO11x...")
def load_model(device: str):
    """Загружает модель YOLO11x. Возвращает (model, error_message).

    При первом запуске веса скачиваются из интернета и кэшируются локально
    Ultralytics (обычно в текущей папке или ~/.cache). Далее приложение
    работает полностью офлайн. Если веса недоступны (нет интернета при первом
    запуске) или отсутствует библиотека ultralytics — возвращаем None и текст
    ошибки, чтобы GUI показал понятное предупреждение и мог продолжить работу
    в демонстрационном режиме без реального распознавания.
    """
    if YOLO is None:
        return None, f"Библиотека ultralytics не установлена: {ULTRALYTICS_IMPORT_ERROR}"

    last_error: Optional[str] = None
    for weights_name in (MODEL_WEIGHTS_PRIMARY, MODEL_WEIGHTS_FALLBACK):
        try:
            model = YOLO(weights_name)
            model.to(device)
            return model, None
        except Exception as exc:  # нет интернета для скачивания весов, битый файл и т.п.
            last_error = f"{weights_name}: {exc}"
            continue
    return None, last_error or "Неизвестная ошибка загрузки модели"


# ---------------------------------------------------------------------------
# Геометрия и вспомогательные вычисления
# ---------------------------------------------------------------------------
def bbox_center(box: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def point_in_circle(px: float, py: float, cx: float, cy: float, r: float) -> bool:
    return (px - cx) ** 2 + (py - cy) ** 2 <= r * r


def distance_point_to_bbox(px: float, py: float, box: Tuple[float, float, float, float]) -> float:
    """Минимальное расстояние от точки до прямоугольника (0, если точка внутри)."""
    x1, y1, x2, y2 = box
    dx = max(x1 - px, 0.0, px - x2)
    dy = max(y1 - py, 0.0, py - y2)
    return math.hypot(dx, dy)


def nearest_player_to_point(
    persons: List[Tuple[int, Tuple[float, float, float, float]]], point: Tuple[float, float]
) -> Optional[int]:
    best_id, best_dist = None, None
    for pid, box in persons:
        cx, cy = bbox_center(box)
        d = math.hypot(cx - point[0], cy - point[1])
        if best_dist is None or d < best_dist:
            best_dist, best_id = d, pid
    return best_id


def ball_is_falling_through(
    ball_history: Deque[Tuple[float, float, float]], ring_center: Tuple[float, float], ring_radius: float
) -> bool:
    """Математическая оценка "гол или промах" по недавней траектории мяча.

    Через полсекунды-секунду после входа мяча в зону кольца смотрим на
    линейную регрессию вертикальной координаты мяча во времени: если мяч
    продолжает двигаться вниз (координата Y на кадре растёт вниз) и в итоге
    оказывается ниже кольца в пределах его горизонтальной проекции — это
    засчитывается как попадание (мяч прошёл через кольцо и сетку). Если мяч
    отскочил вверх/в сторону — это промах (отскок от кольца/щита).
    """
    if len(ball_history) < 3:
        return False
    ts = np.array([p[0] for p in ball_history], dtype=np.float64)
    xs = np.array([p[1] for p in ball_history], dtype=np.float64)
    ys = np.array([p[2] for p in ball_history], dtype=np.float64)

    # Линейная аппроксимация скорости по вертикали (наклон прямой y(t)).
    vy = float(np.polyfit(ts, ys, 1)[0]) if len(ts) >= 2 else 0.0

    last_x, last_y = float(xs[-1]), float(ys[-1])
    moving_down = vy > 0.0
    below_ring = last_y > ring_center[1] + ring_radius * 0.4
    near_ring_x = abs(last_x - ring_center[0]) < ring_radius * 1.6
    return moving_down and below_ring and near_ring_x


# ---------------------------------------------------------------------------
# Разбор результатов детекции/трекинга Ultralytics
# ---------------------------------------------------------------------------
def parse_results(results) -> Tuple["np.ndarray", List[Tuple[int, Tuple[float, float, float, float]]], Optional[Tuple[float, float]]]:
    """Извлекает из результата YOLO кадр с аннотациями, список игроков и мяч."""
    result = results[0]
    annotated = result.plot()  # BGR-кадр с нарисованными боксами и ID треков

    persons: List[Tuple[int, Tuple[float, float, float, float]]] = []
    ball: Optional[Tuple[float, float]] = None

    boxes = result.boxes
    if boxes is not None and boxes.id is not None and len(boxes) > 0:
        xyxy = boxes.xyxy.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        ids = boxes.id.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()

        best_ball_conf = -1.0
        for box, c, tid, conf in zip(xyxy, cls, ids, confs):
            if c == COCO_PERSON_CLASS_ID:
                persons.append((int(tid), (float(box[0]), float(box[1]), float(box[2]), float(box[3]))))
            elif c == COCO_BALL_CLASS_ID and conf > best_ball_conf:
                best_ball_conf = float(conf)
                ball = ((float(box[0]) + float(box[2])) / 2.0, (float(box[1]) + float(box[3])) / 2.0)

    return annotated, persons, ball


# ---------------------------------------------------------------------------
# Нарезка хайлайтов
# ---------------------------------------------------------------------------
class PendingHighlight:
    """Хайлайт, ожидающий "будущих" кадров перед сохранением на диск."""

    __slots__ = ("filename", "past_frames", "future_frames", "frames_needed")

    def __init__(self, filename: str, past_frames: List["np.ndarray"], frames_needed: int):
        self.filename = filename
        self.past_frames = past_frames
        self.future_frames: List["np.ndarray"] = []
        self.frames_needed = frames_needed

    def is_ready(self) -> bool:
        return len(self.future_frames) >= self.frames_needed


def save_highlight_clip(highlight: PendingHighlight, fps: float, width: int, height: int) -> Path:
    """Сохраняет буфер прошлого + будущего в автономный MP4-файл в highlights/."""
    frames = highlight.past_frames + highlight.future_frames
    out_path = HIGHLIGHTS_DIR / highlight.filename
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps if fps > 1 else 25.0, (width, height))
    for frame in frames:
        writer.write(frame)
    writer.release()
    return out_path


# ---------------------------------------------------------------------------
# Основной цикл обработки видео
# ---------------------------------------------------------------------------
def process_video(
    video_path: str,
    model,
    device: str,
    ring_center: Tuple[float, float],
    ring_radius: float,
    possession_threshold: float,
    pass_max_time: float,
    shot_cooldown: float,
    progress_bar,
    status_text,
) -> Tuple[Dict[int, PlayerStats], Path]:
    """Обрабатывает видео покадрово: детекция, трекинг, события, хайлайты."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Не удалось открыть видеофайл (повреждён или неподдерживаемый формат)")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if fps <= 1e-3:
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = 1

    buffer_len = max(int(fps * BUFFER_SECONDS), 1)
    future_frames_needed = max(int(fps * FUTURE_SECONDS), 1)
    shot_verdict_delay_frames = max(int(fps * SHOT_VERDICT_DELAY_SECONDS), 1)

    frame_buffer: Deque["np.ndarray"] = deque(maxlen=buffer_len)
    ball_history: Deque[Tuple[float, float, float]] = deque(maxlen=BALL_HISTORY_MAXLEN)
    pending_highlights: List[PendingHighlight] = []
    pending_shot_verdicts: List[Dict] = []

    stats: Dict[int, PlayerStats] = {}

    # Состояние владения мячом (для пасов и для "кто владел мячом перед броском").
    last_owner: Optional[int] = None
    last_owner_time: Optional[float] = None
    last_shot_time = -1e9  # для кулдауна событий у кольца

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"processed_{Path(video_path).stem}_{timestamp}.mp4"
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    frame_idx = 0
    tracker_path = str(TRACKER_CONFIG_PATH)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t = frame_idx / fps

        if model is not None:
            # persist=True сохраняет ID треков между кадрами одного видео.
            results = model.track(
                frame,
                persist=True,
                tracker=tracker_path,
                device=device,
                classes=[COCO_PERSON_CLASS_ID, COCO_BALL_CLASS_ID],
                verbose=False,
            )
            annotated, persons, ball = parse_results(results)
        else:
            # Демо-режим: модель не загружена (нет интернета/GPU) — не роняем
            # приложение, просто прокатываем видео без детекций.
            annotated = frame.copy()
            persons, ball = [], None
            cv2.putText(
                annotated,
                "DEMO MODE: model YOLO not loaded",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        # -------------------------------------------------------------
        # ВЛАДЕНИЕ МЯЧОМ И ДЕТЕКЦИЯ ПЕРЕДАЧ (ПАСОВ)
        #
        # Игрок считается владеющим мячом, если центр мяча находится не
        # дальше possession_threshold пикселей от его рамки (0, если центр
        # мяча внутри рамки). Если владелец сменился (мяч "долетел" от
        # игрока А к игроку Б) в течение не более pass_max_time секунд —
        # это успешная передача: игроку А засчитывается +1 пас.
        # -------------------------------------------------------------
        current_owner: Optional[int] = None
        if ball is not None and persons:
            best_dist = None
            for pid, box in persons:
                d = distance_point_to_bbox(ball[0], ball[1], box)
                if best_dist is None or d < best_dist:
                    best_dist, current_owner = d, pid
            if best_dist is not None and best_dist > possession_threshold:
                current_owner = None  # мяч ничейный/в полёте — слишком далеко от всех игроков

        if ball is not None:
            ball_history.append((t, ball[0], ball[1]))

        if current_owner is not None:
            if last_owner is not None and current_owner != last_owner and last_owner_time is not None:
                elapsed = t - last_owner_time
                if PASS_MIN_TIME_SECONDS <= elapsed <= pass_max_time:
                    # Мяч перелетел от last_owner к current_owner достаточно быстро —
                    # засчитываем передачу пасующему игроку (last_owner).
                    stats.setdefault(last_owner, blank_stats())["passes"] += 1
                    stats.setdefault(current_owner, blank_stats())  # чтобы получатель тоже был в таблице
                    pending_highlights.append(
                        PendingHighlight(
                            filename=f"pass_from_ID{last_owner}_to_ID{current_owner}_frame_{frame_idx}.mp4",
                            past_frames=list(frame_buffer),
                            frames_needed=future_frames_needed,
                        )
                    )
            last_owner = current_owner
            last_owner_time = t

        # -------------------------------------------------------------
        # ЗОНА КОЛЬЦА: ФИКСАЦИЯ БРОСКА И ВЕРДИКТ "ГОЛ/ПРОМАХ"
        #
        # Если центр мяча входит в окружность (ring_center, ring_radius) и
        # с прошлого события у кольца прошло не меньше shot_cooldown секунд —
        # регистрируем бросок. Бросок засчитывается игроку, который последним
        # владел мячом (last_owner); если такого нет — ближайшему к кольцу
        # игроку. Окончательный вердикт "попадание/промах" выносится чуть
        # позже (см. pending_shot_verdicts) по направлению полёта мяча.
        # -------------------------------------------------------------
        if ball is not None and point_in_circle(ball[0], ball[1], ring_center[0], ring_center[1], ring_radius):
            if (t - last_shot_time) >= shot_cooldown:
                last_shot_time = t
                credited_player = last_owner
                if credited_player is None:
                    credited_player = nearest_player_to_point(persons, ring_center)
                if credited_player is not None:
                    stats.setdefault(credited_player, blank_stats())["shots"] += 1
                    pending_shot_verdicts.append(
                        {
                            "player": credited_player,
                            "frames_left": shot_verdict_delay_frames,
                            "frame_idx": frame_idx,
                        }
                    )

        # Выносим вердикт по накопившимся броскам, у которых истекло время ожидания.
        still_pending_verdicts = []
        for verdict in pending_shot_verdicts:
            verdict["frames_left"] -= 1
            if verdict["frames_left"] <= 0:
                if ball_is_falling_through(ball_history, ring_center, ring_radius):
                    stats[verdict["player"]]["makes"] += 1
                    pending_highlights.append(
                        PendingHighlight(
                            filename=f"goal_ID{verdict['player']}_frame_{verdict['frame_idx']}.mp4",
                            past_frames=list(frame_buffer),
                            frames_needed=future_frames_needed,
                        )
                    )
            else:
                still_pending_verdicts.append(verdict)
        pending_shot_verdicts = still_pending_verdicts

        # -------------------------------------------------------------
        # Буфер прошлого (для хайлайтов) и добор "будущих" кадров для уже
        # зарегистрированных событий.
        # -------------------------------------------------------------
        frame_buffer.append(annotated.copy())

        still_pending_highlights = []
        for highlight in pending_highlights:
            highlight.future_frames.append(annotated.copy())
            if highlight.is_ready():
                save_highlight_clip(highlight, fps, width, height)
            else:
                still_pending_highlights.append(highlight)
        pending_highlights = still_pending_highlights

        writer.write(annotated)
        frame_idx += 1

        if frame_idx % 3 == 0 or frame_idx >= total_frames:
            progress = min(frame_idx / total_frames, 1.0)
            progress_bar.progress(progress)
            status_text.text(f"Обработка кадра {frame_idx}/{total_frames} ({progress * 100:.0f}%)")

    # Досохраняем хайлайты, у которых видео закончилось раньше, чем набралось
    # нужное число "будущих" кадров (событие произошло у самого конца ролика).
    for highlight in pending_highlights:
        if highlight.future_frames:
            save_highlight_clip(highlight, fps, width, height)

    cap.release()
    writer.release()
    return stats, output_path


def build_box_score(stats: Dict[int, PlayerStats]) -> pd.DataFrame:
    """Формирует итоговую таблицу статистики (Box Score) по игрокам."""
    columns = ["ID игрока", "Броски", "Попадания", "Точность (%)", "Сделано передач"]
    if not stats:
        return pd.DataFrame(columns=columns)

    rows = []
    for pid, s in sorted(stats.items()):
        shots, makes, passes = s["shots"], s["makes"], s["passes"]
        accuracy = round(100.0 * makes / shots, 1) if shots else 0.0
        rows.append(
            {
                "ID игрока": pid,
                "Броски": shots,
                "Попадания": makes,
                "Точность (%)": accuracy,
                "Сделано передач": passes,
            }
        )
    return pd.DataFrame(rows, columns=columns)


# ---------------------------------------------------------------------------
# Streamlit GUI
# ---------------------------------------------------------------------------
def run_analysis(
    uploaded_file,
    device: str,
    ring_x: float,
    ring_y: float,
    ring_radius: float,
    possession_threshold: float,
    pass_window: float,
    shot_cooldown: float,
) -> None:
    if cv2 is None:
        st.error(f"Библиотека opencv-python не установлена: {CV2_IMPORT_ERROR}. Установите зависимости из requirements.txt.")
        return

    ensure_directories()
    ensure_tracker_config()

    model, model_error = load_model(device)
    if model is None:
        st.warning(
            "⚠️ Модель YOLO11x не загружена ("
            f"{model_error}). Это ожидаемо в среде без интернета/GPU (например, при "
            "разработке). Обработка продолжится в демонстрационном режиме без "
            "реального распознавания. На рабочем компьютере с RTX 4080 и доступом "
            "в интернет (для первого скачивания весов) модель будет использоваться "
            "полноценно."
        )

    suffix = Path(uploaded_file.name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        video_path = tmp.name

    progress_bar = st.progress(0.0)
    status_text = st.empty()

    try:
        stats, output_path = process_video(
            video_path=video_path,
            model=model,
            device=device,
            ring_center=(float(ring_x), float(ring_y)),
            ring_radius=float(ring_radius),
            possession_threshold=float(possession_threshold),
            pass_max_time=float(pass_window),
            shot_cooldown=float(shot_cooldown),
            progress_bar=progress_bar,
            status_text=status_text,
        )
    except Exception as exc:
        st.error(f"Ошибка при обработке видео: {exc}")
        return
    finally:
        try:
            os.remove(video_path)
        except OSError:
            pass

    status_text.text("Обработка завершена ✅")
    st.session_state["box_score_df"] = build_box_score(stats)
    st.session_state["last_output_video"] = str(output_path)
    st.success(f"Готово! Полное аннотированное видео сохранено: {output_path}")


def render_results_section() -> None:
    st.header("📊 Итоговая статистика (Box Score)")
    df = st.session_state.get("box_score_df")
    if df is not None and not df.empty:
        st.dataframe(df, use_container_width=True, hide_index=True)
    elif df is not None:
        st.info("Видео обработано, но события (броски/передачи) не были зафиксированы.")
    else:
        st.info("Здесь появится итоговая таблица статистики после обработки видео.")

    last_output_video = st.session_state.get("last_output_video")
    if last_output_video and Path(last_output_video).exists():
        with st.expander("🎥 Полное видео с аннотациями"):
            st.video(last_output_video)

    st.header("🎬 Хайлайты (броски и передачи)")
    ensure_directories()
    highlight_files = sorted(HIGHLIGHTS_DIR.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not highlight_files:
        st.info("Нарезанные хайлайты появятся здесь после обнаружения бросков/передач в обработанном видео.")
        return

    for hf in highlight_files:
        st.subheader(hf.name)
        st.video(str(hf))


def main() -> None:
    st.set_page_config(page_title="Basketball Tracking Analytics", page_icon="🏀", layout="wide")
    ensure_directories()
    ensure_tracker_config()

    st.title("🏀 Basketball Tracking Analytics")
    st.caption("Офлайн-аналитика баскетбольных тренировок по видео со статичной камеры (YOLO11x + BoT-SORT/ReID)")

    device, device_message, device_ok = resolve_device()
    if device_ok:
        st.success(device_message)
    else:
        st.warning(device_message)

    with st.sidebar:
        st.header("⚙️ Настройки обработки")
        uploaded_file = st.file_uploader("Видеофайл тренировки", type=["mp4", "mov"])

        st.subheader("🎯 Зона кольца")
        ring_x = st.number_input("Центр кольца, X (px)", min_value=0, value=960, step=1)
        ring_y = st.number_input("Центр кольца, Y (px)", min_value=0, value=200, step=1)
        ring_radius = st.number_input("Радиус зоны кольца (px)", min_value=5, value=60, step=1)

        st.subheader("🤝 Владение мячом и передачи")
        possession_threshold = st.slider("Порог владения мячом (px)", min_value=30, max_value=120, value=65, step=5)
        pass_window = st.slider("Максимальное время передачи (сек)", min_value=0.5, max_value=2.0, value=1.5, step=0.1)

        st.subheader("⏱️ Кулдаун событий у кольца")
        shot_cooldown = st.slider("Кулдаун (сек)", min_value=1.0, max_value=6.0, value=3.0, step=0.5)

        run_button = st.button(
            "🚀 Запустить обработку",
            type="primary",
            use_container_width=True,
            disabled=uploaded_file is None,
        )

        with st.expander("ℹ️ Конфигурация трекера"):
            st.caption(f"Файл: `{TRACKER_CONFIG_PATH.name}` (генерируется автоматически)")
            if TRACKER_CONFIG_PATH.exists():
                st.code(TRACKER_CONFIG_PATH.read_text(encoding="utf-8"), language="yaml")

    if run_button and uploaded_file is not None:
        run_analysis(
            uploaded_file=uploaded_file,
            device=device,
            ring_x=ring_x,
            ring_y=ring_y,
            ring_radius=ring_radius,
            possession_threshold=possession_threshold,
            pass_window=pass_window,
            shot_cooldown=shot_cooldown,
        )

    render_results_section()


if __name__ == "__main__":
    main()

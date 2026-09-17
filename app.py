"""
Basketball Tracking Analytics
==============================

Полностью автономное (офлайн) приложение на Streamlit для аналитики баскетбольных
тренировок по видео со статичной камеры.

Пошаговый пользовательский путь:
    1. Загрузка видео.
    2. Превью кадра + настройка ДВУХ зон колец и порогов владения/передач.
    3. Быстрое предварительное сканирование трекера для сбора списка ID
       игроков и ручное сопоставление ID → Имя/Номер (номеров на форме нет,
       поэтому распознать их автоматически невозможно).
    4. Полный прогон трекинга + аналитики, итоговый Box Score и хайлайты.

Возможности:
    * Детекция и трекинг игроков и мяча моделью YOLO11x (Ultralytics) с трекером
      BoT-SORT + ReID (игроки не имеют номеров на майках, поэтому идентификация
      строится на визуальных признаках формы/обуви, а не на OCR номеров).
    * Математическая детекция бросков/попаданий по ДВУМ окружностям колец,
      заданным пользователем, с кулдауном событий.
    * Детекция передач (пасов) на основе смены владельца мяча в ограниченном
      временном окне, с "памятью" мяча при кратковременной потере детекции.
    * Отладочный таймлайн владения мячом — виден каждый переход владения и
      причина, по которой передача была засчитана или отклонена.
    * Автоматическая нарезка автономных MP4-хайлайтов (5 сек до события +
      2 сек после) с помощью OpenCV.
    * Итоговая статистика по игрокам (Box Score с именем/номером) и встроенный
      просмотр хайлайтов.

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
from typing import Any, Deque, Dict, List, Optional, Tuple

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

# --- Дефолты и диапазоны настраиваемых через GUI порогов ---
# ВАЖНО: на реальном видео исходный жёсткий порог владения (65 px) оказался
# слишком строгим — при чуть более высоком разрешении/дальнем плане камеры
# рука/мяч игрока легко выходит за пределы этого расстояния от рамки, и
# смена владельца никогда не засчитывалась как пас. Дефолт увеличен, а сам
# порог и окно передачи полностью управляются слайдерами в GUI (шаг 2).
POSSESSION_THRESHOLD_DEFAULT = 90
POSSESSION_THRESHOLD_MIN = 20
POSSESSION_THRESHOLD_MAX = 250

PASS_WINDOW_DEFAULT = 1.8
PASS_WINDOW_MIN = 0.3
PASS_WINDOW_MAX = 3.0

SHOT_COOLDOWN_DEFAULT = 3.0
SHOT_COOLDOWN_MIN = 1.0
SHOT_COOLDOWN_MAX = 6.0

# "Память" мяча: если детектор не нашёл мяч в текущем кадре (блики, смаз
# движения, быстрый полёт), сколько секунд продолжать считать его находящимся
# в последней известной точке. Без этого короткие пропуски детекции мяча
# ровно в момент приёма пасующим партнёром обрывали цепочку "владелец A → Б".
BALL_MEMORY_SECONDS_DEFAULT = 0.3
BALL_MEMORY_SECONDS_MIN = 0.0
BALL_MEMORY_SECONDS_MAX = 1.0

QUICK_SCAN_SECONDS_DEFAULT = 15


PlayerStats = Dict[str, int]
RingZone = Dict[str, float]


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


def reset_tracker(model) -> None:
    """Сбрасывает внутреннее состояние трекера BoT-SORT перед новой независимой
    сессией трекинга (быстрое сканирование ID и финальный полный прогон должны
    оба стартовать "с чистого листа" на кадре 0 — иначе ID из предыдущего
    прогона исказят нумерацию и сопоставление игрок → имя/номер разойдётся
    с финальной статистикой).
    """
    if model is None:
        return
    try:
        model.predictor = None  # заставит ultralytics создать трекер заново
    except Exception:
        pass


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
def parse_results(
    results,
) -> Tuple[Any, List[Tuple[int, Tuple[float, float, float, float]]], Optional[Tuple[float, float]]]:
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

    def __init__(self, filename: str, past_frames: List[Any], frames_needed: int):
        self.filename = filename
        self.past_frames = past_frames
        self.future_frames: List[Any] = []
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
# Видео-утилиты (метаданные, извлечение кадра, отрисовка зон)
# ---------------------------------------------------------------------------
def get_video_metadata(video_path: str) -> Dict[str, float]:
    """Возвращает fps/ширину/высоту/число кадров/длительность видео."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if fps <= 1e-3:
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    cap.release()
    duration = total_frames / fps if total_frames else 0.0
    return {"fps": fps, "width": width, "height": height, "total_frames": total_frames, "duration": duration}


def extract_frame_at_time(video_path: str, t_seconds: float):
    """Извлекает один кадр видео на заданной секунде (для превью настройки зон)."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_idx = max(int(t_seconds * fps), 0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def default_ring_zones(width: int, height: int) -> Tuple[RingZone, RingZone]:
    """Разумные дефолтные координаты для двух колец (по краям площадки)."""
    r = max(int(min(width, height) * 0.05), 20)
    ring1 = {"x": float(int(width * 0.10)), "y": float(int(height * 0.35)), "r": float(r)}
    ring2 = {"x": float(int(width * 0.90)), "y": float(int(height * 0.35)), "r": float(r)}
    return ring1, ring2


def draw_zones_preview(frame, rings: List[RingZone]):
    """Рисует окружности зон колец на копии кадра для наглядной проверки в GUI."""
    preview = frame.copy()
    colors = [(0, 140, 255), (255, 80, 0)]
    for idx, ring in enumerate(rings):
        color = colors[idx % len(colors)]
        center = (int(ring["x"]), int(ring["y"]))
        radius = max(int(ring["r"]), 1)
        cv2.circle(preview, center, radius, color, 3)
        cv2.drawMarker(preview, center, color, markerType=cv2.MARKER_CROSS, markerSize=14, thickness=2)
        label_pos = (max(center[0] - 45, 0), max(center[1] - radius - 12, 20))
        cv2.putText(preview, f"Кольцо {idx + 1}", label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    return preview


# ---------------------------------------------------------------------------
# Быстрое предварительное сканирование для сбора списка игроков (шаг 3)
# ---------------------------------------------------------------------------
def quick_player_scan(
    video_path: str, model, device: str, max_seconds: float, fps_hint: float
) -> Dict[int, Any]:
    """Короткий прогон трекера по первым max_seconds секундам видео.

    Цель — не полноценная аналитика, а быстрый сбор всех уникальных ID
    игроков и одного репрезентативного кропа (самой уверенной/крупной
    детекции) на каждого, чтобы пользователь мог вручную вписать имя/номер.
    Разрешение НЕ уменьшается и кадры не пропускаются, чтобы ID трекера
    совпадали с ID, которые получатся при финальном полном прогоне на шаге 4
    (тот же трекер, тот же конфиг, тот же сброс состояния — см. reset_tracker).
    """
    if model is None or cv2 is None:
        return {}

    reset_tracker(model)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {}

    fps = fps_hint or 25.0
    max_frames = max(int(fps * max_seconds), 1)

    best_crops: Dict[int, Tuple[float, Any]] = {}
    frame_idx = 0
    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(
            frame,
            persist=True,
            tracker=str(TRACKER_CONFIG_PATH),
            device=device,
            classes=[COCO_PERSON_CLASS_ID, COCO_BALL_CLASS_ID],
            verbose=False,
        )
        result = results[0]
        boxes = result.boxes
        if boxes is not None and boxes.id is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            cls = boxes.cls.cpu().numpy().astype(int)
            ids = boxes.id.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()
            h, w = frame.shape[:2]
            for box, c, tid, conf in zip(xyxy, cls, ids, confs):
                if c != COCO_PERSON_CLASS_ID:
                    continue
                x1, y1, x2, y2 = [int(v) for v in box]
                area = max(x2 - x1, 0) * max(y2 - y1, 0)
                score = float(conf) * area
                prev = best_crops.get(int(tid))
                if prev is None or score > prev[0]:
                    x1c, y1c = max(x1, 0), max(y1, 0)
                    x2c, y2c = min(x2, w), min(y2, h)
                    crop = frame[y1c:y2c, x1c:x2c].copy()
                    if crop.size > 0:
                        best_crops[int(tid)] = (score, crop)
        frame_idx += 1

    cap.release()
    reset_tracker(model)  # не оставляем состояние "подвешенным" перед шагом 4
    return {tid: crop for tid, (_, crop) in best_crops.items()}


# ---------------------------------------------------------------------------
# Основной цикл обработки видео (шаг 4)
# ---------------------------------------------------------------------------
def process_video(
    video_path: str,
    model,
    device: str,
    rings: List[RingZone],
    possession_threshold: float,
    pass_max_time: float,
    shot_cooldown: float,
    ball_memory_seconds: float,
    progress_bar,
    status_text,
) -> Tuple[Dict[int, PlayerStats], Path, Dict[str, List[Dict[str, Any]]]]:
    """Обрабатывает видео покадрово: детекция, трекинг, события, хайлайты.

    Возвращает (стата по игрокам, путь к аннотированному видео, отладочный лог
    владения мячом для GUI-таймлайна).
    """
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
    sample_every = max(int(fps // 2), 1)  # ~2 отсчёта в секунду для таймлайна отладки

    frame_buffer: Deque[Any] = deque(maxlen=buffer_len)
    ball_history: Deque[Tuple[float, float, float]] = deque(maxlen=BALL_HISTORY_MAXLEN)
    pending_highlights: List[PendingHighlight] = []
    pending_shot_verdicts: List[Dict[str, Any]] = []

    stats: Dict[int, PlayerStats] = {}
    debug_log: Dict[str, List[Dict[str, Any]]] = {"ownership_changes": [], "sampled_timeline": []}

    # Состояние владения мячом (для пасов и для "кто владел мячом перед броском").
    last_owner: Optional[int] = None
    last_owner_time: Optional[float] = None
    last_shot_time_per_ring = [-1e9 for _ in rings]

    # "Память" мяча: держим последнюю известную позицию, если детектор
    # временно "потерял" мяч (см. константу BALL_MEMORY_SECONDS_DEFAULT).
    last_known_ball: Optional[Tuple[float, float]] = None
    last_real_ball_time = -1e9

    reset_tracker(model)  # независимая от возможного шага 3 сессия трекинга

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
        # "ПАМЯТЬ" МЯЧА
        #
        # Если ровно в кадре, где мяч перелетает от игрока А к игроку Б,
        # детектор его не находит (смаз движения, блик, частичное перекрытие),
        # цепочка владения обрывалась и передача никогда не засчитывалась.
        # Поэтому в течение ball_memory_seconds после последней РЕАЛЬНОЙ
        # детекции мяча мы продолжаем считать его находящимся в последней
        # известной точке (effective_ball) — это не заменяет детекцию,
        # а лишь сглаживает короткие пропуски.
        # -------------------------------------------------------------
        if ball is not None:
            last_known_ball = ball
            last_real_ball_time = t

        effective_ball: Optional[Tuple[float, float]] = None
        if last_known_ball is not None and (t - last_real_ball_time) <= ball_memory_seconds:
            effective_ball = last_known_ball

        if effective_ball is not None:
            ball_history.append((t, effective_ball[0], effective_ball[1]))

        # -------------------------------------------------------------
        # ВЛАДЕНИЕ МЯЧОМ И ДЕТЕКЦИЯ ПЕРЕДАЧ (ПАСОВ)
        #
        # Игрок считается владеющим мячом, если центр мяча находится не
        # дальше possession_threshold пикселей от его рамки (0, если центр
        # мяча внутри рамки). Если владелец сменился (мяч "долетел" от
        # игрока А к игроку Б) в течение не более pass_max_time секунд —
        # это успешная передача: игроку А засчитывается +1 пас. Все смены
        # владения (в т.ч. отклонённые из-за долгого перелёта) логируются
        # в debug_log для отладочного таймлайна в GUI.
        # -------------------------------------------------------------
        current_owner: Optional[int] = None
        if effective_ball is not None and persons:
            best_dist = None
            for pid, box in persons:
                d = distance_point_to_bbox(effective_ball[0], effective_ball[1], box)
                if best_dist is None or d < best_dist:
                    best_dist, current_owner = d, pid
            if best_dist is not None and best_dist > possession_threshold:
                current_owner = None  # мяч ничейный/в полёте — слишком далеко от всех игроков

        if current_owner is not None and current_owner != last_owner:
            elapsed = (t - last_owner_time) if last_owner_time is not None else None
            if last_owner is None:
                reason = "первое владение в кадре"
            elif elapsed is not None and PASS_MIN_TIME_SECONDS <= elapsed <= pass_max_time:
                reason = "✅ передача засчитана"
                stats.setdefault(last_owner, blank_stats())["passes"] += 1
                stats.setdefault(current_owner, blank_stats())  # чтобы получатель тоже был в таблице
                pending_highlights.append(
                    PendingHighlight(
                        filename=f"pass_from_ID{last_owner}_to_ID{current_owner}_frame_{frame_idx}.mp4",
                        past_frames=list(frame_buffer),
                        frames_needed=future_frames_needed,
                    )
                )
            else:
                reason = f"❌ отклонено (Δt={elapsed:.2f}с > окно {pass_max_time:.2f}с)" if elapsed is not None else "❌ отклонено"

            debug_log["ownership_changes"].append(
                {
                    "Кадр": frame_idx,
                    "Время, с": round(t, 2),
                    "От игрока": last_owner if last_owner is not None else "—",
                    "К игроку": current_owner,
                    "Δt, с": round(elapsed, 2) if elapsed is not None else None,
                    "Результат": reason,
                }
            )

        if current_owner is not None:
            last_owner = current_owner
            last_owner_time = t

        if frame_idx % sample_every == 0:
            debug_log["sampled_timeline"].append(
                {
                    "Время, с": round(t, 2),
                    "Владелец (ID)": float(current_owner) if current_owner is not None else float("nan"),
                }
            )

        # -------------------------------------------------------------
        # ЗОНЫ КОЛЕЦ: ФИКСАЦИЯ БРОСКА И ВЕРДИКТ "ГОЛ/ПРОМАХ"
        #
        # Если центр мяча входит в одну из окружностей колец (ring_center,
        # ring_radius) и с прошлого события у ЭТОГО кольца прошло не меньше
        # shot_cooldown секунд — регистрируем бросок. Бросок засчитывается
        # игроку, который последним владел мячом (last_owner); если такого
        # нет — ближайшему к кольцу игроку. Окончательный вердикт
        # "попадание/промах" выносится чуть позже (см. pending_shot_verdicts)
        # по направлению полёта мяча.
        # -------------------------------------------------------------
        if effective_ball is not None:
            for ring_idx, ring in enumerate(rings):
                if not point_in_circle(effective_ball[0], effective_ball[1], ring["x"], ring["y"], ring["r"]):
                    continue
                if (t - last_shot_time_per_ring[ring_idx]) < shot_cooldown:
                    continue
                last_shot_time_per_ring[ring_idx] = t
                credited_player = last_owner
                if credited_player is None:
                    credited_player = nearest_player_to_point(persons, (ring["x"], ring["y"]))
                if credited_player is not None:
                    stats.setdefault(credited_player, blank_stats())["shots"] += 1
                    pending_shot_verdicts.append(
                        {
                            "player": credited_player,
                            "frames_left": shot_verdict_delay_frames,
                            "frame_idx": frame_idx,
                            "ring_idx": ring_idx,
                            "ring_center": (ring["x"], ring["y"]),
                            "ring_radius": ring["r"],
                        }
                    )

        # Выносим вердикт по накопившимся броскам, у которых истекло время ожидания.
        still_pending_verdicts = []
        for verdict in pending_shot_verdicts:
            verdict["frames_left"] -= 1
            if verdict["frames_left"] <= 0:
                if ball_is_falling_through(ball_history, verdict["ring_center"], verdict["ring_radius"]):
                    stats[verdict["player"]]["makes"] += 1
                    pending_highlights.append(
                        PendingHighlight(
                            filename=f"goal_ring{verdict['ring_idx'] + 1}_ID{verdict['player']}_frame_{verdict['frame_idx']}.mp4",
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
    return stats, output_path, debug_log


def build_box_score(
    stats: Dict[int, PlayerStats],
    player_names: Optional[Dict[int, str]] = None,
    player_numbers: Optional[Dict[int, str]] = None,
) -> pd.DataFrame:
    """Формирует итоговую таблицу статистики (Box Score) по игрокам."""
    player_names = player_names or {}
    player_numbers = player_numbers or {}
    columns = ["ID игрока", "Имя", "Номер", "Броски", "Попадания", "Точность (%)", "Сделано передач"]
    if not stats:
        return pd.DataFrame(columns=columns)

    rows = []
    for pid, s in sorted(stats.items()):
        shots, makes, passes = s["shots"], s["makes"], s["passes"]
        accuracy = round(100.0 * makes / shots, 1) if shots else 0.0
        name = (player_names.get(pid) or "").strip() or f"Игрок {pid}"
        number = (player_numbers.get(pid) or "").strip() or "—"
        rows.append(
            {
                "ID игрока": pid,
                "Имя": name,
                "Номер": number,
                "Броски": shots,
                "Попадания": makes,
                "Точность (%)": accuracy,
                "Сделано передач": passes,
            }
        )
    return pd.DataFrame(rows, columns=columns)


# ---------------------------------------------------------------------------
# Streamlit GUI — общая инициализация состояния
# ---------------------------------------------------------------------------
def init_session_state() -> None:
    defaults: Dict[str, Any] = {
        "step": 1,
        "video_path": None,
        "video_name": None,
        "rings_initialized_for": None,
        "ring1_x": 0.0,
        "ring1_y": 0.0,
        "ring1_r": 40.0,
        "ring2_x": 0.0,
        "ring2_y": 0.0,
        "ring2_r": 40.0,
        "possession_threshold": float(POSSESSION_THRESHOLD_DEFAULT),
        "pass_window": float(PASS_WINDOW_DEFAULT),
        "ball_memory": float(BALL_MEMORY_SECONDS_DEFAULT),
        "shot_cooldown": float(SHOT_COOLDOWN_DEFAULT),
        "preview_time": 0.0,
        "player_crops": {},
        "player_names": {},
        "player_numbers": {},
        "box_score_df": None,
        "debug_log": None,
        "last_output_video": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def reset_for_new_video() -> None:
    """Сбрасывает всё, что зависит от конкретного видео (при загрузке нового)."""
    for key in (
        "rings_initialized_for",
        "player_crops",
        "player_names",
        "player_numbers",
        "box_score_df",
        "debug_log",
        "last_output_video",
    ):
        st.session_state[key] = {} if key in ("player_crops", "player_names", "player_numbers") else None


def go_to_step(step: int) -> None:
    st.session_state["step"] = step
    st.rerun()


def render_step_indicator(current: int) -> None:
    labels = ["1. Видео", "2. Зоны и пороги", "3. Игроки", "4. Анализ"]
    parts = []
    for i, label in enumerate(labels, start=1):
        if i == current:
            parts.append(f"**➡️ {label}**")
        elif i < current:
            parts.append(f"✅ {label}")
        else:
            parts.append(f"◽ {label}")
    st.markdown(" &nbsp;→&nbsp; ".join(parts))
    st.divider()


# ---------------------------------------------------------------------------
# Шаг 1 — загрузка видео
# ---------------------------------------------------------------------------
def render_step1_upload() -> None:
    st.header("Шаг 1 — Загрузка видео")
    st.write("Загрузите видео тренировки со статичной камеры, снятое так, чтобы кольцо(-а) и площадка были в кадре.")

    if cv2 is None:
        st.error(f"Библиотека opencv-python не установлена: {CV2_IMPORT_ERROR}. Установите зависимости из requirements.txt.")

    uploaded_file = st.file_uploader("Видеофайл тренировки (MP4/MOV)", type=["mp4", "mov"])

    if uploaded_file is not None and cv2 is not None:
        if st.session_state.get("video_name") != uploaded_file.name:
            suffix = Path(uploaded_file.name).suffix or ".mp4"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.getbuffer())
                st.session_state["video_path"] = tmp.name
            st.session_state["video_name"] = uploaded_file.name
            reset_for_new_video()

    video_path = st.session_state.get("video_path")
    if video_path and Path(video_path).exists():
        meta = get_video_metadata(video_path)
        st.success(f"Видео загружено: {st.session_state.get('video_name')}")
        st.caption(
            f"Длительность: {meta['duration']:.1f} сек · {meta['fps']:.1f} fps · "
            f"{meta['width']}×{meta['height']} · {int(meta['total_frames'])} кадров"
        )
        if st.button("Далее → Настройка зон и порогов", type="primary"):
            go_to_step(2)
    else:
        st.info("Загрузите видеофайл тренировки, чтобы продолжить.")


# ---------------------------------------------------------------------------
# Шаг 2 — превью + зоны колец + пороги владения/передач
# ---------------------------------------------------------------------------
def render_step2_zones() -> None:
    st.header("Шаг 2 — Превью и настройка зон")
    video_path = st.session_state.get("video_path")
    if not video_path or not Path(video_path).exists():
        st.warning("Сначала загрузите видео на шаге 1.")
        if st.button("← Назад к загрузке"):
            go_to_step(1)
        return

    meta = get_video_metadata(video_path)

    if st.session_state.get("rings_initialized_for") != video_path:
        ring1, ring2 = default_ring_zones(int(meta["width"]), int(meta["height"]))
        st.session_state["ring1_x"], st.session_state["ring1_y"], st.session_state["ring1_r"] = (
            ring1["x"],
            ring1["y"],
            ring1["r"],
        )
        st.session_state["ring2_x"], st.session_state["ring2_y"], st.session_state["ring2_r"] = (
            ring2["x"],
            ring2["y"],
            ring2["r"],
        )
        st.session_state["rings_initialized_for"] = video_path

    max_t = max(meta["duration"] - 0.05, 0.0)
    st.session_state["preview_time"] = st.slider(
        "Кадр для превью (сек)", min_value=0.0, max_value=max_t if max_t > 0 else 0.1,
        value=min(st.session_state["preview_time"], max_t), step=0.5,
    )
    frame = extract_frame_at_time(video_path, st.session_state["preview_time"])

    st.subheader("🎯 Зоны обоих колец")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Кольцо №1**")
        st.session_state["ring1_x"] = st.number_input(
            "X1 (px)", min_value=0, max_value=int(meta["width"]), value=int(st.session_state["ring1_x"]), key="in_ring1_x"
        )
        st.session_state["ring1_y"] = st.number_input(
            "Y1 (px)", min_value=0, max_value=int(meta["height"]), value=int(st.session_state["ring1_y"]), key="in_ring1_y"
        )
        st.session_state["ring1_r"] = st.number_input(
            "Радиус 1 (px)", min_value=5, max_value=int(max(meta["width"], meta["height"])),
            value=int(st.session_state["ring1_r"]), key="in_ring1_r",
        )
    with col2:
        st.markdown("**Кольцо №2**")
        st.session_state["ring2_x"] = st.number_input(
            "X2 (px)", min_value=0, max_value=int(meta["width"]), value=int(st.session_state["ring2_x"]), key="in_ring2_x"
        )
        st.session_state["ring2_y"] = st.number_input(
            "Y2 (px)", min_value=0, max_value=int(meta["height"]), value=int(st.session_state["ring2_y"]), key="in_ring2_y"
        )
        st.session_state["ring2_r"] = st.number_input(
            "Радиус 2 (px)", min_value=5, max_value=int(max(meta["width"], meta["height"])),
            value=int(st.session_state["ring2_r"]), key="in_ring2_r",
        )

    if frame is not None:
        rings = [
            {"x": st.session_state["ring1_x"], "y": st.session_state["ring1_y"], "r": st.session_state["ring1_r"]},
            {"x": st.session_state["ring2_x"], "y": st.session_state["ring2_y"], "r": st.session_state["ring2_r"]},
        ]
        preview = draw_zones_preview(frame, rings)
        st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB), caption="Превью с зонами колец", use_container_width=True)
    else:
        st.error("Не удалось прочитать кадр из видео для превью.")

    st.subheader("🤝 Владение мячом, передачи и кулдаун")
    st.caption(
        "Если на реальном видео передачи не засчитываются — увеличьте порог владения "
        "и/или окно передачи ниже, и проверьте отладочный таймлайн на шаге 4."
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        st.session_state["possession_threshold"] = st.slider(
            "Порог владения мячом (px)",
            min_value=POSSESSION_THRESHOLD_MIN,
            max_value=POSSESSION_THRESHOLD_MAX,
            value=int(st.session_state["possession_threshold"]),
            step=5,
            help="Максимальное расстояние от центра мяча до рамки игрока, при котором игрок считается владеющим мячом.",
        )
    with c2:
        st.session_state["pass_window"] = st.slider(
            "Максимальное время передачи (сек)",
            min_value=PASS_WINDOW_MIN,
            max_value=PASS_WINDOW_MAX,
            value=float(st.session_state["pass_window"]),
            step=0.1,
            help="Сколько секунд может лететь мяч от одного игрока к другому, чтобы это засчиталось как пас.",
        )
    with c3:
        st.session_state["ball_memory"] = st.slider(
            "Память мяча при потере детекции (сек)",
            min_value=BALL_MEMORY_SECONDS_MIN,
            max_value=BALL_MEMORY_SECONDS_MAX,
            value=float(st.session_state["ball_memory"]),
            step=0.05,
            help="Сколько секунд считать мяч в последней известной точке, если детектор его временно не находит.",
        )

    st.session_state["shot_cooldown"] = st.slider(
        "Кулдаун события у кольца (сек)",
        min_value=SHOT_COOLDOWN_MIN,
        max_value=SHOT_COOLDOWN_MAX,
        value=float(st.session_state["shot_cooldown"]),
        step=0.5,
        help="Минимальный промежуток между двумя засчитанными бросками у одного и того же кольца.",
    )

    colA, colB = st.columns(2)
    with colA:
        if st.button("← Назад к загрузке"):
            go_to_step(1)
    with colB:
        if st.button("Далее → Сопоставление игроков", type="primary"):
            go_to_step(3)


# ---------------------------------------------------------------------------
# Шаг 3 — сопоставление игроков (ID → имя/номер)
# ---------------------------------------------------------------------------
def render_step3_players(device: str) -> None:
    st.header("Шаг 3 — Сопоставление игроков")
    video_path = st.session_state.get("video_path")
    if not video_path or not Path(video_path).exists():
        st.warning("Сначала загрузите видео на шаге 1.")
        if st.button("← Назад к загрузке"):
            go_to_step(1)
        return

    meta = get_video_metadata(video_path)
    st.write(
        "Короткое предварительное сканирование первых секунд видео собирает список ID игроков "
        "и по одному характерному кадру-кропу на каждого. Номера на форме не видны камере и "
        "распознать их автоматически невозможно — впишите имя/номер вручную по кропам."
    )

    max_scan = max(int(min(meta["duration"], 60)), 5)
    scan_seconds = st.slider(
        "Сколько секунд видео сканировать", min_value=5, max_value=max_scan,
        value=min(QUICK_SCAN_SECONDS_DEFAULT, max_scan), step=5,
    )

    if st.button("🔍 Запустить предварительное сканирование", type="primary"):
        model, model_error = load_model(device)
        if model is None:
            st.warning(
                f"⚠️ Модель YOLO11x не загружена ({model_error}). Список игроков не может быть собран "
                "автоматически в этой среде — на рабочем ПК с RTX 4080 и интернетом для первого "
                "скачивания весов сканирование сработает штатно. Этот шаг можно пропустить: имена по "
                "умолчанию останутся как «Игрок {ID}»."
            )
        else:
            with st.spinner("Сканирование..."):
                crops = quick_player_scan(video_path, model, device, max_seconds=scan_seconds, fps_hint=meta["fps"])
            st.session_state["player_crops"] = crops
            if not crops:
                st.warning("Игроки не найдены за это время — попробуйте увеличить длительность сканирования.")
            else:
                st.success(f"Найдено {len(crops)} уникальных ID игроков.")

    crops: Dict[int, Any] = st.session_state.get("player_crops") or {}
    if crops:
        cols_per_row = 4
        ids_sorted = sorted(crops.keys())
        for row_start in range(0, len(ids_sorted), cols_per_row):
            row_ids = ids_sorted[row_start : row_start + cols_per_row]
            cols = st.columns(len(row_ids))
            for col, pid in zip(cols, row_ids):
                with col:
                    st.image(cv2.cvtColor(crops[pid], cv2.COLOR_BGR2RGB), caption=f"ID {pid}", use_container_width=True)
                    st.text_input("Имя", key=f"player_name_{pid}", placeholder=f"Игрок {pid}")
                    st.text_input("Номер", key=f"player_number_{pid}", placeholder="—")
    else:
        st.info(
            "Пока нет данных — запустите сканирование выше, либо пропустите этот шаг: "
            "в итоговой таблице игроки будут отображаться как «Игрок {ID}»."
        )

    colA, colB = st.columns(2)
    with colA:
        if st.button("← Назад к зонам и порогам"):
            go_to_step(2)
    with colB:
        if st.button("Далее → Полный анализ", type="primary"):
            for pid in crops.keys():
                st.session_state["player_names"][pid] = st.session_state.get(f"player_name_{pid}", "")
                st.session_state["player_numbers"][pid] = st.session_state.get(f"player_number_{pid}", "")
            go_to_step(4)


# ---------------------------------------------------------------------------
# Шаг 4 — полный прогон и итоговая статистика
# ---------------------------------------------------------------------------
def run_full_analysis(video_path: str, device: str) -> None:
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

    rings = [
        {"x": st.session_state["ring1_x"], "y": st.session_state["ring1_y"], "r": st.session_state["ring1_r"]},
        {"x": st.session_state["ring2_x"], "y": st.session_state["ring2_y"], "r": st.session_state["ring2_r"]},
    ]

    progress_bar = st.progress(0.0)
    status_text = st.empty()

    try:
        stats, output_path, debug_log = process_video(
            video_path=video_path,
            model=model,
            device=device,
            rings=rings,
            possession_threshold=float(st.session_state["possession_threshold"]),
            pass_max_time=float(st.session_state["pass_window"]),
            shot_cooldown=float(st.session_state["shot_cooldown"]),
            ball_memory_seconds=float(st.session_state["ball_memory"]),
            progress_bar=progress_bar,
            status_text=status_text,
        )
    except Exception as exc:
        st.error(f"Ошибка при обработке видео: {exc}")
        return

    status_text.text("Обработка завершена ✅")
    st.session_state["box_score_df"] = build_box_score(
        stats, st.session_state.get("player_names"), st.session_state.get("player_numbers")
    )
    st.session_state["last_output_video"] = str(output_path)
    st.session_state["debug_log"] = debug_log
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

    debug_log = st.session_state.get("debug_log")
    if debug_log is not None:
        with st.expander("🔍 Отладка: таймлайн владения мячом (почему пас засчитан/отклонён)"):
            changes = debug_log.get("ownership_changes") or []
            if changes:
                changes_df = pd.DataFrame(changes)
                st.dataframe(changes_df, use_container_width=True, hide_index=True)
                accepted = sum(1 for c in changes if "засчитана" in c["Результат"])
                rejected = len(changes) - accepted
                st.caption(f"Смен владения: {len(changes)} · засчитано передач: {accepted} · отклонено: {rejected}")
            else:
                st.info("Смен владения мячом не зафиксировано — мяч либо не был обнаружен, либо всё время был у одного игрока.")

            sampled = debug_log.get("sampled_timeline") or []
            if sampled:
                st.caption("Владелец мяча (ID) во времени — разрывы означают, что мяч в этот момент ничей/не найден.")
                sampled_df = pd.DataFrame(sampled).set_index("Время, с")
                st.line_chart(sampled_df)

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


def render_step4_run(device: str) -> None:
    st.header("Шаг 4 — Полный анализ и статистика")
    video_path = st.session_state.get("video_path")
    if not video_path or not Path(video_path).exists():
        st.warning("Сначала загрузите видео на шаге 1.")
        if st.button("← Назад к загрузке"):
            go_to_step(1)
        return

    with st.expander("⚙️ Текущие настройки анализа", expanded=False):
        st.write(
            f"Кольцо 1: X={int(st.session_state['ring1_x'])}, Y={int(st.session_state['ring1_y'])}, "
            f"R={int(st.session_state['ring1_r'])} · Кольцо 2: X={int(st.session_state['ring2_x'])}, "
            f"Y={int(st.session_state['ring2_y'])}, R={int(st.session_state['ring2_r'])}"
        )
        st.write(
            f"Порог владения: {int(st.session_state['possession_threshold'])} px · "
            f"Окно передачи: {st.session_state['pass_window']:.1f} с · "
            f"Память мяча: {st.session_state['ball_memory']:.2f} с · "
            f"Кулдаун кольца: {st.session_state['shot_cooldown']:.1f} с"
        )
        n_players = len(st.session_state.get("player_names") or {})
        st.write(f"Сопоставлено игроков (имя/номер): {n_players}")

    if st.button("🚀 Запустить полный анализ", type="primary"):
        run_full_analysis(video_path, device)

    render_results_section()

    st.divider()
    colA, colB = st.columns(2)
    with colA:
        if st.button("← Назад к сопоставлению игроков"):
            go_to_step(3)
    with colB:
        if st.button("🔄 Начать заново с новым видео"):
            old_path = st.session_state.get("video_path")
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            if old_path:
                try:
                    os.remove(old_path)
                except OSError:
                    pass
            st.rerun()


def main() -> None:
    st.set_page_config(page_title="Basketball Tracking Analytics", page_icon="🏀", layout="wide")
    ensure_directories()
    ensure_tracker_config()
    init_session_state()

    st.title("🏀 Basketball Tracking Analytics")
    st.caption("Офлайн-аналитика баскетбольных тренировок по видео со статичной камеры (YOLO11x + BoT-SORT/ReID)")

    device, device_message, device_ok = resolve_device()
    if device_ok:
        st.success(device_message)
    else:
        st.warning(device_message)

    with st.sidebar:
        st.header("ℹ️ Конфигурация трекера")
        st.caption(f"Файл: `{TRACKER_CONFIG_PATH.name}` (генерируется автоматически)")
        if TRACKER_CONFIG_PATH.exists():
            st.code(TRACKER_CONFIG_PATH.read_text(encoding="utf-8"), language="yaml")
        st.caption(
            "Пошаговый процесс: загрузка видео → зоны колец и пороги → сопоставление "
            "игроков → полный анализ. Переключайтесь между шагами кнопками «Назад/Далее»."
        )

    step = st.session_state["step"]
    render_step_indicator(step)

    if step == 1:
        render_step1_upload()
    elif step == 2:
        render_step2_zones()
    elif step == 3:
        render_step3_players(device)
    else:
        render_step4_run(device)


if __name__ == "__main__":
    main()

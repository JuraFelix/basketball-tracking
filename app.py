"""
Basketball Tracking Analytics
==============================

Полностью автономное (офлайн) приложение на Streamlit для аналитики баскетбольных
тренировок по видео (статичная или панорамная камера).

Пошаговый пользовательский путь:
    1. Загрузка видео.
    2. Превью кадра + настройка ДВУХ зон колец и порогов владения/передач.
    3. Быстрое предварительное сканирование трекера для сбора списка ID
       игроков и ручное сопоставление ID → Имя/Номер (OCR номеров выключен
       по умолчанию — включайте в проф. режиме, только если номера читаются).
    4. Полный прогон трекинга + аналитики, итоговый Box Score и хайлайты.

Возможности:
    * Детекция и трекинг игроков и мяча моделью YOLO11x (Ultralytics) с трекером
      ByteTrack (игроки без номеров на майках — удержание ID за счёт трекера,
      а не OCR номеров).
    * Векторная детекция голов: пересечение траектории мяча с горизонтальной
      линией кольца (задаётся пользователем), с кулдауном в секундах.
    * Детекция передач (пасов): игрок владел мячом → мяч летел без владельца
      в заданном окне (секунды) → другой игрок получил владение.
    * Устойчивый трекинг мяча при пропусках YOLO: Kalman + интерполяция +
      цветовой fallback (оранжевый blob в ROI).
    * Отладочный таймлайн владения мячом — виден каждый переход владения и
      причина, по которой передача была засчитана или отклонена.
    * Автоматическая нарезка автономных MP4-хайлайтов (5 сек до события +
      2 сек после) с помощью OpenCV.
    * Итоговая статистика по игрокам (Box Score с именем/номером) и встроенный
      просмотр хайлайтов.

Запуск:
    streamlit run app.py

Первый запуск скачивает веса модели YOLO11x (~110 МБ) из интернета.
Все последующие запуски полностью офлайн.
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import subprocess
import tempfile
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import yaml
from PIL import Image, ImageDraw, ImageFont

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

try:
    import easyocr

    EASYOCR_IMPORT_ERROR: Optional[str] = None
except Exception as exc:  # pragma: no cover
    easyocr = None  # type: ignore[assignment]
    EASYOCR_IMPORT_ERROR = str(exc)

# streamlit-image-coordinates — необязательная лёгкая зависимость для клика
# мышкой по превью (шаг 2). Если пакета нет — GUI просто скрывает кликабельный
# режим и оставляет числовые поля/слайдеры как единственный способ ввода.
try:
    from streamlit_image_coordinates import streamlit_image_coordinates

    IMAGE_COORDINATES_IMPORT_ERROR: Optional[str] = None
except Exception as exc:  # pragma: no cover - защита от отсутствия пакета
    streamlit_image_coordinates = None  # type: ignore[assignment]
    IMAGE_COORDINATES_IMPORT_ERROR = str(exc)


# ---------------------------------------------------------------------------
# Константы, пути и настройки по умолчанию
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
HIGHLIGHTS_DIR = BASE_DIR / "highlights"
OUTPUT_DIR = BASE_DIR / "output_videos"
TRAINING_SEEDS_DIR = BASE_DIR / "training_seeds"
BALL_LABELS_JSONL = TRAINING_SEEDS_DIR / "ball_labels.jsonl"
BALL_TRAINING_CROPS_DIR = TRAINING_SEEDS_DIR / "crops"
TRACKER_CONFIG_PATH = BASE_DIR / "custom_bytetrack.yaml"

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

# Сколько последних точек траектории мяча хранить для отрисовки и анализа.
BALL_TRAJECTORY_DRAW_LEN = 10
BALL_HISTORY_MAXLEN = 20

# Дефолтные пороги инференса YOLO при вызове model.track() — низкий conf
# помогает замечать удалённых игроков на противоположной стороне площадки.
TRACK_CONF_DEFAULT = 0.15
TRACK_IOU_DEFAULT = 0.45

# Окно передачи в СЕКУНДАХ: мяч без владельца от min до max (конвертируется в кадры через fps).
PASS_MIN_SECONDS_DEFAULT = 0.35
PASS_MIN_SECONDS_MIN = 0.10
PASS_MIN_SECONDS_MAX = 1.50

PASS_MAX_SECONDS_DEFAULT = 2.0
PASS_MAX_SECONDS_MIN = 0.50
PASS_MAX_SECONDS_MAX = 4.0

# Минимальная дистанция полёта мяча (доля порога владения) — отсекает ведение.
PASS_FLIGHT_MIN_POSSESSION_FRACTION = 0.35

# Кулдаун между голами в одном эпизоде у кольца (секунды; ~3 с при 30 fps).
GOAL_COOLDOWN_SECONDS_DEFAULT = 3.0
GOAL_COOLDOWN_SECONDS_MIN = 1.0
GOAL_COOLDOWN_SECONDS_MAX = 6.0

# --- Дефолты и диапазоны настраиваемых через GUI порогов ---
# ВАЖНО: на реальном видео исходный жёсткий порог владения (65 px) оказался
# слишком строгим — при чуть более высоком разрешении/дальнем плане камеры
# рука/мяч игрока легко выходит за пределы этого расстояния от рамки, и
# смена владельца никогда не засчитывалась как пас. Дефолт увеличен, а сам
# порог и окно передачи полностью управляются слайдерами в GUI (шаг 2).
POSSESSION_THRESHOLD_DEFAULT = 90
POSSESSION_THRESHOLD_MIN = 20
POSSESSION_THRESHOLD_MAX = 250


# "Память" мяча: если детектор не нашёл мяч в текущем кадре (блики, смаз
# движения, быстрый полёт), сколько секунд продолжать считать его находящимся
# в последней известной точке. Без этого короткие пропуски детекции мяча
# ровно в момент приёма пасующим партнёром обрывали цепочку "владелец A → Б".
BALL_MEMORY_SECONDS_DEFAULT = 0.3
BALL_MEMORY_SECONDS_MIN = 0.0
BALL_MEMORY_SECONDS_MAX = 1.0

QUICK_SCAN_SECONDS_DEFAULT = 15

# --- Пороги уверенности детекции и разрешение инференса ---
# Мяч ('sports ball') — гораздо более мелкий и часто смазанный объект, чем
# игрок, и на видео низкого качества/при быстром полёте его confidence нередко
# заметно ниже, чем у игроков. Единый жёсткий порог (например, дефолтные
# 0.25 в ultralytics) в таком случае либо пропускает мяч (низкий порог, но
# тогда растёт число ложных срабатываний по игрокам/фону), либо теряет мяч
# почти всегда (высокий порог). Поэтому используются РАЗНЫЕ пороги: у
# model.track()/model.predict() нет параметра per-class conf, поэтому на
# инференс подаётся МИНИМАЛЬНЫЙ из двух порогов (чтобы мяч точно не был
# отфильтрован на этом этапе), а затем детекции класса person и sports ball
# фильтруются постфактум каждая своим порогом (см. parse_track_results).
BALL_CONF_DEFAULT = 0.18
BALL_CONF_MIN = 0.05
BALL_CONF_MAX = 0.5

PERSON_CONF_DEFAULT = 0.15
PERSON_CONF_MIN = 0.05
PERSON_CONF_MAX = 0.8

# Разрешение, до которого YOLO letterbox-ит кадр перед инференсом. Дефолт
# ultralytics — 640px, чего часто недостаточно для мелкого мяча на кадре
# высокого разрешения (после ресайза до 640px мяч может занимать буквально
# несколько пикселей). Увеличение imgsz даёт мячу больше пикселей ценой
# более медленного инференса — поэтому вынесено в настраиваемый GUI-параметр.
IMGSZ_DEFAULT = 640
IMGSZ_OPTIONS = [640, 768, 896, 1024, 1152, 1280]

# Диагностика видимости мяча (шаг 3): равномерный сэмпл по всему видео,
# жёсткий потолок инференсов — не гоняем сотни/тысячи кадров подряд.
BALL_DIAGNOSTIC_MAX_SAMPLES = 100
BALL_DIAGNOSTIC_MIN_SAMPLES = 80

# Быстрый скан игроков (шаг 3): макс. число кадров с model.track() за один запуск.
QUICK_SCAN_MAX_TRACK_FRAMES = 300

# --- Устойчивый трекинг мяча при пропусках YOLO ---
BALL_MAX_GAP_FRAMES_DEFAULT = 20
BALL_MAX_GAP_FRAMES_MIN = 10
BALL_MAX_GAP_FRAMES_MAX = 40

BALL_MAX_PREDICT_FRAMES_DEFAULT = 25
BALL_MAX_PREDICT_FRAMES_MIN = 15
BALL_MAX_PREDICT_FRAMES_MAX = 40

BALL_COLOR_FALLBACK_DEFAULT = True
BALL_COLOR_ROI_HALF_DEFAULT = 120
BALL_COLOR_ROI_HALF_MIN = 40
BALL_COLOR_ROI_HALF_MAX = 300

# HSV-диапазоны типичного оранжевого баскетбольного мяча (OpenCV H: 0–180).
ORANGE_HSV_LOWER1 = np.array([5, 70, 70], dtype=np.uint8)
ORANGE_HSV_UPPER1 = np.array([25, 255, 255], dtype=np.uint8)
ORANGE_HSV_LOWER2 = np.array([0, 70, 70], dtype=np.uint8)
ORANGE_HSV_UPPER2 = np.array([4, 255, 255], dtype=np.uint8)

BALL_SOURCE_COLORS_BGR = {
    "yolo": (0, 215, 255),
    "color": (0, 140, 255),
    "interp": (0, 255, 255),
    "kalman": (180, 255, 255),
    "csrt": (0, 255, 128),
    "tiled": (255, 200, 0),
    "roi": (255, 160, 0),
    "lost": (160, 160, 160),
    "user": (0, 120, 255),
    "user_interp": (0, 180, 255),
}
BALL_PREVIEW_COLOR_BGR = (0, 140, 255)
BALL_PREVIEW_INTERP_COLOR_BGR = (0, 200, 255)
CLICK_TARGET_OPTIONS = ("Кольцо 1", "Кольцо 2", "Мяч")
BALL_ANCHORS_MIN_RECOMMENDED = 2
BALL_INTERP_CHECKS_PER_GAP = 3
RING_ANCHOR_SNAP_FRAMES = 2
CAMERA_TRACK_MIN_INLIERS = 8
CAMERA_TRACK_MAX_SCALE_DELTA = 0.12
# Яркая траектория мяча на аннотированном видео (BGR) + чёрная обводка.
BALL_TRAJECTORY_COLOR_BGR = (0, 255, 255)
BALL_TRAJECTORY_THICKNESS = 6
BALL_TRAJECTORY_OUTLINE_BGR = (0, 0, 0)
BALL_DEFAULT_BBOX_HALF = 14
BALL_CSRT_MAX_JUMP_PX = 120
CSRT_REINIT_IOU_THRESHOLD = 0.35
CSRT_REINIT_CONF_MARGIN = 0.12
# Источники мяча, по которым можно считать гол/пас (не kalman/color/interp/user_interp).
EVENT_ELIGIBLE_BALL_SOURCES = frozenset({"yolo", "csrt", "user", "tiled", "roi"})
# Макс. разрыв между двумя event_ball-кадрами для гола (сек) — иначе не соединять хордой.
EVENT_BALL_MAX_GAP_SECONDS = 0.3
# CSRT-seed по клику: только на кадре якоря и коротком хвосте после него.
BALL_CSRT_SEED_TAIL_FRAMES = 5
BALL_TILED_IMGSZ = 1280
BALL_ROI_DETECT_IMGSZ = 960
JERSEY_OCR_ENABLED_DEFAULT = False
JERSEY_OCR_MIN_CONF = 0.45
REID_SIMILARITY_THRESHOLD = 0.78
# Жёсткий гейт внешности: ниже — не доверяем ID ByteTrack (лучше новый ID, чем swap).
APPEARANCE_SWAP_GATE = 0.58
APPEARANCE_MATCH_MIN = 0.62
APPEARANCE_MOTION_MAX_PX = 180
# Минимальная площадь bbox игрока (px²); 0 = не отбрасывать мелкие силуэты.
MIN_PERSON_BBOX_AREA_DEFAULT = 0
MIN_PERSON_BBOX_AREA_MIN = 0
MIN_PERSON_BBOX_AREA_MAX = 2000
# Линии колец на аннотированном видео: яркий cyan/lime + чёрная обводка (BGR).
HOOP_LINE_COLORS_BGR = [(255, 255, 0), (0, 255, 128)]
HOOP_LINE_THICKNESS = 7
HOOP_LINE_OUTLINE_THICKNESS = 12

# Appearance-matching поверх ByteTrack (гистограмма HSV майки, без ReID).
APPEARANCE_SIMILARITY_DEFAULT = 0.72
APPEARANCE_SIMILARITY_MIN = 0.50
APPEARANCE_SIMILARITY_MAX = 0.95
APPEARANCE_LOST_BUFFER_FRAMES = 180
APPEARANCE_HIST_EMA_ALPHA = 0.35

# --- Режим камеры ---
# "Статичная камера" — исходное поведение (зоны колец фиксированы во всех
# кадрах). "Камера в движении" — экспериментальный режим для видео с плавной
# панорамой/наклоном камеры: позиции обеих зон колец, заданные пользователем
# на кадре превью, пересчитываются в каждом кадре с учётом накопленного
# сдвига камеры (см. estimate_camera_transforms/compute_dynamic_rings).
CAMERA_MODE_STATIC = "static"
CAMERA_MODE_PANNING = "panning"
CAMERA_MODE_LABELS = {
    CAMERA_MODE_STATIC: "📷 Статичная камера (стандартный режим)",
    CAMERA_MODE_PANNING: "🎥 Камера в движении / панорама (экспериментально)",
}

# Доля от средней диагонали рамки игрока на видео, используемая как
# авто-предложенный порог владения мячом (вместо фиксированных 90 px) —
# на видео с дальней/близкой камерой игроки занимают разное число пикселей,
# и порог должен масштабироваться вместе с ними.
POSSESSION_THRESHOLD_DIAGONAL_FRACTION = 0.5

# Во сколько раз апскейлить кадр перед детекцией при включённой опции
# "улучшить качество кадра" (шаг 2). Помогает трекеру/ReID на видео низкого
# разрешения, но не является панацеей при изначально плохом качестве видео.
ENHANCE_UPSCALE_FACTOR = 1.6

# Высота, до которой приводятся все кропы игроков в сетке на шаге 3
# (пропорции сохраняются) — делает сетку компактной и аккуратной.
CROP_DISPLAY_HEIGHT = 160

# Максимальная ширина превью-изображения для кликабельного выбора кольца
# на шаге 2 (большие кадры уменьшаются для компактности интерфейса).
PREVIEW_MAX_DISPLAY_WIDTH = 900

# Путь к системному ffmpeg (если установлен) — используется для перекодировки
# видео в H.264, совместимый с браузерным <video> (см. reencode_for_browser).
FFMPEG_PATH = shutil.which("ffmpeg")


PlayerStats = Dict[str, int]
# Горизонтальная линия кольца: центр (x, y) и полуширина отрезка линии.
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


def ensure_training_seeds_dir() -> None:
    """Создаёт training_seeds/ для накопления разметки мяча (не для git)."""
    TRAINING_SEEDS_DIR.mkdir(parents=True, exist_ok=True)
    BALL_TRAINING_CROPS_DIR.mkdir(parents=True, exist_ok=True)
    if not BALL_LABELS_JSONL.exists():
        BALL_LABELS_JSONL.write_text("", encoding="utf-8")


def clear_highlights_directory() -> None:
    """Удаляет старые mp4 из highlights/ (оставляет .gitkeep и прочие файлы)."""
    ensure_directories()
    for path in HIGHLIGHTS_DIR.glob("*.mp4"):
        try:
            path.unlink()
        except OSError:
            pass


def reset_analysis_results() -> None:
    """Сбрасывает результаты последнего прогона анализа (шаг 4)."""
    st.session_state["box_score_df"] = None
    st.session_state["debug_log"] = None
    st.session_state["last_output_video"] = None
    st.session_state["highlight_files"] = []


def ensure_tracker_config(path: Path = TRACKER_CONFIG_PATH) -> Path:
    """Генерирует/обновляет конфиг ByteTrack: низкий new_track_thresh для дальних игроков."""
    tracker_cfg = {
        "tracker_type": "bytetrack",
        "track_high_thresh": 0.2,
        "track_low_thresh": 0.1,
        "new_track_thresh": 0.15,
        "track_buffer": 180,
        "match_thresh": 0.75,
        "fuse_score": True,
    }
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            "# Автоматически сгенерированный конфиг трекера ByteTrack.\n"
            "# track_buffer=180 — помнить игрока ~5–6 сек при окклюзии.\n"
            "# track_low_thresh=0.1 — не терять трек при частичном перекрытии.\n"
            "# new_track_thresh=0.15 — чаще стартовать трек для мелких/дальних силуэтов.\n"
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
    """Сбрасывает внутреннее состояние трекера ByteTrack перед новой независимой
    сессией трекинга (быстрое сканирование ID и финальный полный прогон должны
    оба стартовать "с чистого листа" на кадре 0 — иначе ID из предыдущего
    прогона исказят нумерацию и сопоставление игрок → имя/номер разойдётся
    с финальной статистикой).
    """
    if model is None:
        return
    try:
        predictor = getattr(model, "predictor", None)
        if predictor is not None:
            callbacks = getattr(predictor, "callbacks", None)
            if isinstance(callbacks, dict):
                for event in callbacks:
                    callbacks[event] = []
        model.predictor = None  # заставит ultralytics создать трекер заново
    except Exception:
        pass


def tracker_callback_count(model) -> int:
    """Число зарегистрированных callback-ов Ultralytics (для тестов F07)."""
    if model is None:
        return 0
    predictor = getattr(model, "predictor", None)
    if predictor is None:
        return 0
    callbacks = getattr(predictor, "callbacks", None)
    if not isinstance(callbacks, dict):
        return 0
    total = 0
    for handlers in callbacks.values():
        if isinstance(handlers, list):
            total += len(handlers)
    return total


# ---------------------------------------------------------------------------
# Геометрия и вспомогательные вычисления
# ---------------------------------------------------------------------------
def bbox_center(box: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def bbox_area(box: Tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = box
    return max(float(x2) - float(x1), 0.0) * max(float(y2) - float(y1), 0.0)


def bbox_iou(
    box_a: Tuple[float, float, float, float], box_b: Tuple[float, float, float, float]
) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(ix2 - ix1, 0.0), max(iy2 - iy1, 0.0)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = bbox_area(box_a)
    area_b = bbox_area(box_b)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


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


def segment_crosses_hoop_line_top_to_bottom(
    x_prev: float,
    y_prev: float,
    x_curr: float,
    y_curr: float,
    line_y: float,
    center_x: float,
    half_width: float,
) -> bool:
    """Проверяет пересечение отрезка движения мяча с горизонтальной линией кольца сверху вниз.

    В координатах кадра Y растёт вниз: мяч должен перейти из области выше линии
    (y_prev < line_y) в область на линии или ниже (y_curr >= line_y), а точка
    пересечения по X должна попадать в отрезок [center_x - half_width, center_x + half_width].
    """
    if y_prev >= line_y or y_curr < line_y:
        return False
    if abs(y_curr - y_prev) < 1e-6:
        cross_x = (x_prev + x_curr) / 2.0
    else:
        t = (line_y - y_prev) / (y_curr - y_prev)
        cross_x = x_prev + t * (x_curr - x_prev)
    return abs(cross_x - center_x) <= half_width


# ---------------------------------------------------------------------------
# Экспериментальный режим "камера в движении": оценка сдвига камеры между
# кадрами и пересчёт позиций зон колец под панораму/наклон.
# ---------------------------------------------------------------------------
def _affine2x3_to_3x3(m: np.ndarray) -> np.ndarray:
    T = np.eye(3, dtype=np.float64)
    T[:2, :] = m
    return T


def _is_degenerate_homography(H: Optional[np.ndarray]) -> bool:
    if H is None or H.shape != (3, 3):
        return True
    if not np.all(np.isfinite(H)):
        return True
    if abs(float(H[2, 2])) < 1e-12:
        return True
    try:
        det = float(np.linalg.det(H))
    except np.linalg.LinAlgError:
        return True
    return abs(det) < 1e-8 or abs(det) > 1e4


def _filter_flow_outliers(
    prev_pts: np.ndarray, next_pts: np.ndarray, mad_multiplier: float = 2.5
) -> Tuple[np.ndarray, np.ndarray]:
    """Оставляет треки с похожей величиной сдвига — движущиеся игроки отсекаются."""
    if len(prev_pts) < 8:
        return prev_pts, next_pts
    flow = next_pts.reshape(-1, 2) - prev_pts.reshape(-1, 2)
    mag = np.linalg.norm(flow, axis=1)
    med = float(np.median(mag))
    mad = float(np.median(np.abs(mag - med))) + 1e-6
    if med < 0.25:
        return prev_pts, next_pts
    keep = np.abs(mag - med) <= mad_multiplier * max(mad, 0.5)
    if int(np.count_nonzero(keep)) >= 6:
        return prev_pts[keep], next_pts[keep]
    return prev_pts, next_pts


def _homography_scale(H: np.ndarray) -> float:
    return float(math.hypot(float(H[0, 0]), float(H[1, 0])))


def _estimate_incremental_transform(
    prev_pts: np.ndarray, next_pts: np.ndarray
) -> Tuple[np.ndarray, int]:
    """Оценивает T_{i-1→i}: prev_pts (кадр i-1) → next_pts (кадр i). Homography, затем affine."""
    incremental = np.eye(3, dtype=np.float64)
    if len(prev_pts) < 6:
        return incremental, 0
    prev_f = prev_pts.reshape(-1, 1, 2).astype(np.float32)
    next_f = next_pts.reshape(-1, 1, 2).astype(np.float32)
    prev_f, next_f = _filter_flow_outliers(prev_f, next_f)
    if len(prev_f) < 6:
        return incremental, 0
    H, inlier_mask = cv2.findHomography(prev_f, next_f, cv2.RANSAC, 3.0, maxIters=2000, confidence=0.995)
    if H is not None and not _is_degenerate_homography(H):
        H = H.astype(np.float64)
        H /= H[2, 2]
        inliers = int(np.count_nonzero(inlier_mask)) if inlier_mask is not None else len(prev_f)
        return H, inliers
    m, inlier_mask = cv2.estimateAffinePartial2D(
        prev_f, next_f, method=cv2.RANSAC, ransacReprojThreshold=3.0
    )
    if m is not None:
        inliers = int(np.count_nonzero(inlier_mask)) if inlier_mask is not None else len(prev_f)
        return _affine2x3_to_3x3(m), inliers
    return incremental, 0


def _detect_local_motion_blobs(
    prev_gray: np.ndarray, gray: np.ndarray, dilate_kernel: np.ndarray
) -> np.ndarray:
    """Маскирует только локальные движущиеся объекты (игроки), не глобальный панорамный сдвиг."""
    h, w = prev_gray.shape
    diff = cv2.absdiff(prev_gray, gray)
    _, motion_bin = cv2.threshold(diff, 24, 255, cv2.THRESH_BINARY)
    motion_bin = cv2.dilate(motion_bin, dilate_kernel, iterations=1)
    contours, _ = cv2.findContours(motion_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros((h, w), dtype=np.uint8)
    frame_area = float(h * w)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 400.0 or area > frame_area * 0.12:
            continue
        cv2.drawContours(mask, [contour], -1, 255, thickness=-1)
    if np.count_nonzero(mask):
        mask = cv2.dilate(mask, dilate_kernel, iterations=2)
    return mask


def _create_background_feature_mask(
    gray: np.ndarray, motion_mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """Маска фич: весь кадр, кроме локально движущихся объектов (person-bbox-подобных blob)."""
    h, w = gray.shape
    mask = np.full((h, w), 255, dtype=np.uint8)
    if motion_mask is not None:
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(motion_mask))
    margin = max(int(min(w, h) * 0.03), 4)
    mask[:margin, :] = 255
    mask[h - margin :, :] = 255
    mask[:, :margin] = 255
    mask[:, w - margin :] = 255
    return mask


def _collect_matched_points(
    prev_gray: np.ndarray,
    gray: np.ndarray,
    bg_mask: np.ndarray,
    orb: Any,
    bf: Any,
    feature_params: Dict[str, Any],
    lk_params: Dict[str, Any],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """ORB-матчи + LK-поток только по фоновой маске."""
    prev_list: List[List[float]] = []
    next_list: List[List[float]] = []

    kp_prev, desc_prev = orb.detectAndCompute(prev_gray, bg_mask)
    kp_curr, desc_curr = orb.detectAndCompute(gray, bg_mask)
    if (
        desc_prev is not None
        and desc_curr is not None
        and len(desc_prev) >= 8
        and len(desc_curr) >= 8
    ):
        matches = bf.knnMatch(desc_prev, desc_curr, k=2)
        for pair in matches:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < 0.75 * n.distance:
                prev_list.append(kp_prev[m.queryIdx].pt)
                next_list.append(kp_curr[m.trainIdx].pt)

    prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=bg_mask, **feature_params)
    if prev_pts is not None and len(prev_pts) >= 6:
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, **lk_params)
        if next_pts is not None and status is not None:
            status_flat = status.flatten() == 1
            good_prev = prev_pts[status_flat]
            good_next = next_pts[status_flat]
            for pt_prev, pt_next in zip(good_prev.reshape(-1, 2), good_next.reshape(-1, 2)):
                prev_list.append([float(pt_prev[0]), float(pt_prev[1])])
                next_list.append([float(pt_next[0]), float(pt_next[1])])

    if len(prev_list) < 6:
        return None, None
    return np.array(prev_list, dtype=np.float32), np.array(next_list, dtype=np.float32)


def _matrix_to_affine_params(T: np.ndarray) -> Tuple[float, float, float, float]:
    tx, ty = float(T[0, 2]), float(T[1, 2])
    a, b = float(T[0, 0]), float(T[1, 0])
    scale = math.hypot(a, b)
    theta = math.atan2(b, a) if scale > 1e-9 else 0.0
    return tx, ty, scale, theta


def _affine_params_to_matrix(tx: float, ty: float, scale: float, theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    T = np.eye(3, dtype=np.float64)
    T[0, 0] = scale * c
    T[0, 1] = -scale * s
    T[1, 0] = scale * s
    T[1, 1] = scale * c
    T[0, 2] = tx
    T[1, 2] = ty
    return T


def smooth_cumulative_transforms(
    transforms: List[np.ndarray], window: int = 9
) -> List[np.ndarray]:
    """Сглаживает tx/ty/scale/rotation накопленной гомографии (не только сдвиг)."""
    if len(transforms) < 3 or window < 3:
        return transforms
    n = len(transforms)
    params = np.array([_matrix_to_affine_params(T) for T in transforms], dtype=np.float64)
    half = window // 2
    smoothed = params.copy()
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        smoothed[i] = np.mean(params[lo:hi], axis=0)
    return [
        _affine_params_to_matrix(float(p[0]), float(p[1]), float(p[2]), float(p[3]))
        for p in smoothed
    ]


def estimate_camera_transforms(
    video_path: str,
    progress_callback: Optional[Any] = None,
    frame_progress_callback: Optional[Callable[[int, int], None]] = None,
) -> List[np.ndarray]:
    """Оценивает покадровый сдвиг камеры по фоновым фичам (ORB + LK, RANSAC).

    Возвращает список накопленных 3×3 матриц длиной в число кадров, где
    transforms[i] переводит точку из системы координат КАДРА 0 в систему
    координат КАДРА i. transforms[0] — единичная матрица.

    Для якоря на кадре A и отрисовки на кадре t используйте
    transform_ring_to_frame (композиция T_0→t · inv(T_0→A)).
    """
    if cv2 is None:
        return [np.eye(3, dtype=np.float64)]
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [np.eye(3, dtype=np.float64)]

    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        return [np.eye(3, dtype=np.float64)]

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    cumulative: List[np.ndarray] = [np.eye(3, dtype=np.float64)]

    feature_params = dict(maxCorners=300, qualityLevel=0.01, minDistance=10, blockSize=7)
    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    orb = cv2.ORB_create(nfeatures=700, fastThreshold=12)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    motion_kernel = np.ones((13, 13), np.uint8)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        motion_mask = _detect_local_motion_blobs(prev_gray, gray, motion_kernel)
        bg_mask = _create_background_feature_mask(prev_gray, motion_mask)

        incremental = np.eye(3, dtype=np.float64)
        inlier_count = 0
        prev_pts, next_pts = _collect_matched_points(
            prev_gray, gray, bg_mask, orb, bf, feature_params, lk_params
        )
        if prev_pts is not None and next_pts is not None:
            incremental, inlier_count = _estimate_incremental_transform(prev_pts, next_pts)

        prev_cum = cumulative[-1]
        next_cum = incremental @ prev_cum
        prev_scale = _homography_scale(prev_cum)
        next_scale = _homography_scale(next_cum)
        scale_delta = abs(next_scale - prev_scale) / max(prev_scale, 1e-6)
        if inlier_count < CAMERA_TRACK_MIN_INLIERS or scale_delta > CAMERA_TRACK_MAX_SCALE_DELTA:
            cumulative.append(prev_cum.copy())
        else:
            cumulative.append(next_cum)
        prev_gray = gray

        current_frame = len(cumulative) - 1
        if frame_progress_callback is not None:
            try:
                frame_progress_callback(current_frame, total_frames)
            except Exception:
                pass
        elif progress_callback is not None and current_frame % 15 == 0:
            try:
                progress_callback(min(current_frame / total_frames, 1.0))
            except Exception:
                pass

    cap.release()
    if frame_progress_callback is not None:
        try:
            frame_progress_callback(total_frames, total_frames)
        except Exception:
            pass
    if progress_callback is not None:
        try:
            progress_callback(1.0)
        except Exception:
            pass
    return smooth_cumulative_transforms(cumulative, window=9)


def create_synthetic_pan_video(
    path: str,
    n_frames: int = 20,
    dx_per_frame: float = 10.0,
    width: int = 640,
    height: int = 480,
    moving_blobs: bool = False,
) -> None:
    """Синтетическое видео: каждый кадр контент сдвигается вправо на dx_per_frame px.

    Используется в геометрических тестах warp колец. moving_blobs — имитация игроков.
    """
    if cv2 is None:
        raise RuntimeError("opencv required")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, 25.0, (width, height))
    base = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            if (x // 20 + y // 20) % 2 == 0:
                base[y, x] = (40, 40, 40)
    cv2.line(base, (100, 0), (100, height), (255, 255, 255), 2)
    rng = np.random.default_rng(42)
    for i in range(n_frames):
        M = np.float32([[1, 0, i * dx_per_frame], [0, 1, 0]])
        frame = cv2.warpAffine(
            base, M, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
        )
        if moving_blobs:
            for _ in range(4):
                bx = int(rng.integers(80, width - 80))
                by = int(rng.integers(120, height - 80))
                br = int(rng.integers(18, 32))
                cv2.circle(frame, (bx + i * 3, by - i * 2), br, (0, 0, 200), -1)
        writer.write(frame)
    writer.release()


def transform_point_between_frames(
    x: float,
    y: float,
    source_frame_idx: int,
    target_frame_idx: int,
    camera_transforms: List[np.ndarray],
) -> Tuple[float, float]:
    """Переносит точку из системы координат source_frame_idx в target_frame_idx."""
    if not camera_transforms or source_frame_idx == target_frame_idx:
        return float(x), float(y)
    n = len(camera_transforms)
    source_idx = int(np.clip(int(source_frame_idx), 0, n - 1))
    target_idx = int(np.clip(int(target_frame_idx), 0, n - 1))
    try:
        source_inv = np.linalg.inv(camera_transforms[source_idx])
    except np.linalg.LinAlgError:
        return float(x), float(y)
    transform = camera_transforms[target_idx] @ source_inv
    point = transform @ np.array([x, y, 1.0], dtype=np.float64)
    return float(point[0]), float(point[1])


def transform_ring_to_frame(
    ring: RingZone,
    camera_transforms: List[np.ndarray],
    target_frame_idx: int,
) -> RingZone:
    """Переносит кольцо из координат якорного кадра (ring['anchor_frame']) в target_frame_idx.

    camera_transforms[i] — накопленная гомография из кадра 0 в кадр i.
    Композиция: T_0→target * inv(T_0→anchor).
    """
    if not camera_transforms:
        return dict(ring)
    n = len(camera_transforms)
    anchor_idx = int(np.clip(int(ring.get("anchor_frame", 0)), 0, n - 1))
    target_idx = int(np.clip(int(target_frame_idx), 0, n - 1))
    try:
        anchor_inv = np.linalg.inv(camera_transforms[anchor_idx])
    except np.linalg.LinAlgError:
        return dict(ring)
    transform = camera_transforms[target_idx] @ anchor_inv
    scale = math.hypot(transform[0, 0], transform[1, 0]) or 1.0
    point = transform @ np.array([ring["x"], ring["y"], 1.0], dtype=np.float64)
    half_w = float(ring.get("half_width", ring.get("r", 40.0)))
    return {
        "x": float(point[0]),
        "y": float(point[1]),
        "half_width": float(half_w * scale),
        "configured": bool(ring.get("configured", True)),
        "anchor_frame": anchor_idx,
    }


def normalize_ring_anchors(anchors: Optional[List[Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for item in anchors or []:
        if not isinstance(item, dict):
            continue
        out.append(
            {
                "x": float(item["x"]),
                "y": float(item["y"]),
                "half_width": float(item.get("half_width", item.get("r", 40.0))),
                "frame": int(item["frame"]),
            }
        )
    out.sort(key=lambda a: int(a["frame"]))
    return out


def ring_anchors_from_state(state: Dict[str, Any], ring_num: int) -> List[Dict[str, Any]]:
    """Список якорей кольца из session_state; при отсутствии — legacy single-anchor."""
    stored = normalize_ring_anchors(state.get(f"ring{ring_num}_anchors"))
    if stored:
        return stored
    if state.get(f"ring{ring_num}_configured"):
        return [
            {
                "x": float(state.get(f"ring{ring_num}_x", 0)),
                "y": float(state.get(f"ring{ring_num}_y", 0)),
                "half_width": float(state.get(f"ring{ring_num}_r", 40)),
                "frame": int(state.get(f"ring{ring_num}_frame", 0)),
            }
        ]
    return []


def ensure_ring_anchors_migrated(state: Dict[str, Any], ring_num: int) -> None:
    """Переносит legacy single-anchor в ring{N}_anchors при первом входе в динамику."""
    key = f"ring{ring_num}_anchors"
    if state.get(key):
        return
    legacy = ring_anchors_from_state(state, ring_num)
    if legacy:
        state[key] = legacy


def add_ring_anchor(
    state: Dict[str, Any],
    ring_num: int,
    x: int,
    y: int,
    frame_idx: int,
    half_width: Optional[float] = None,
) -> None:
    hw = float(half_width if half_width is not None else state.get(f"ring{ring_num}_r", 40))
    key = f"ring{ring_num}_anchors"
    anchors = [
        a for a in normalize_ring_anchors(state.get(key))
        if int(a["frame"]) != int(frame_idx)
    ]
    anchors.append({"x": float(x), "y": float(y), "half_width": hw, "frame": int(frame_idx)})
    anchors.sort(key=lambda a: int(a["frame"]))
    state[key] = anchors
    state[f"ring{ring_num}_configured"] = True
    state[f"ring{ring_num}_x"] = int(x)
    state[f"ring{ring_num}_y"] = int(y)
    state[f"ring{ring_num}_r"] = hw
    state[f"ring{ring_num}_frame"] = int(frame_idx)
    _set_pending_ring_widget_coords(state, ring_num, int(x), int(y), hw)


def remove_ring_anchor_at(state: Dict[str, Any], ring_num: int, index: int) -> None:
    key = f"ring{ring_num}_anchors"
    anchors = normalize_ring_anchors(state.get(key)) or ring_anchors_from_state(state, ring_num)
    if 0 <= index < len(anchors):
        anchors.pop(index)
    state[key] = anchors
    if anchors:
        latest = anchors[-1]
        state[f"ring{ring_num}_configured"] = True
        state[f"ring{ring_num}_x"] = int(latest["x"])
        state[f"ring{ring_num}_y"] = int(latest["y"])
        state[f"ring{ring_num}_r"] = float(latest["half_width"])
        state[f"ring{ring_num}_frame"] = int(latest["frame"])
    else:
        state[f"ring{ring_num}_configured"] = False


def _ring_anchor_as_zone(anchor: Dict[str, Any]) -> RingZone:
    return {
        "x": float(anchor["x"]),
        "y": float(anchor["y"]),
        "half_width": float(anchor["half_width"]),
        "anchor_frame": int(anchor["frame"]),
        "configured": True,
    }


def resolve_ring_from_anchors(
    anchors: List[Dict[str, Any]],
    frame_idx: int,
    camera_transforms: Optional[List[np.ndarray]],
) -> Optional[RingZone]:
    """Позиция линии кольца на frame_idx: warp якорей в текущий кадр + интерполяция."""
    sorted_anchors = normalize_ring_anchors(anchors)
    if not sorted_anchors:
        return None
    if not camera_transforms:
        zone = _ring_anchor_as_zone(sorted_anchors[0])
        zone["anchor_frame"] = int(frame_idx)
        return zone

    def project(anchor: Dict[str, Any], target: int) -> RingZone:
        return transform_ring_to_frame(_ring_anchor_as_zone(anchor), camera_transforms, target)

    if len(sorted_anchors) == 1:
        return project(sorted_anchors[0], frame_idx)

    first_f = int(sorted_anchors[0]["frame"])
    last_f = int(sorted_anchors[-1]["frame"])
    if frame_idx <= first_f:
        return project(sorted_anchors[0], frame_idx)
    if frame_idx >= last_f:
        return project(sorted_anchors[-1], frame_idx)

    for i in range(len(sorted_anchors) - 1):
        a0 = sorted_anchors[i]
        a1 = sorted_anchors[i + 1]
        f0, f1 = int(a0["frame"]), int(a1["frame"])
        if f0 <= frame_idx <= f1:
            if f1 == f0:
                return project(a0, frame_idx)
            if frame_idx - f0 <= RING_ANCHOR_SNAP_FRAMES:
                return project(a0, frame_idx)
            if f1 - frame_idx <= RING_ANCHOR_SNAP_FRAMES:
                return project(a1, frame_idx)
            t_frac = (frame_idx - f0) / float(f1 - f0)
            p0 = project(a0, frame_idx)
            p1 = project(a1, frame_idx)
            return {
                "x": p0["x"] + t_frac * (p1["x"] - p0["x"]),
                "y": p0["y"] + t_frac * (p1["y"] - p0["y"]),
                "half_width": p0["half_width"] + t_frac * (p1["half_width"] - p0["half_width"]),
                "configured": True,
                "anchor_frame": int(frame_idx),
            }
    return project(sorted_anchors[0], frame_idx)


def compute_dynamic_rings(
    rings: List[RingZone],
    camera_transforms: List[np.ndarray],
    frame_idx: int,
) -> List[RingZone]:
    """Пересчитывает кольца в frame_idx: multi-anchor интерполяция или single-anchor warp."""
    if not camera_transforms:
        return rings
    out: List[RingZone] = []
    for ring in rings:
        anchors = ring.get("anchors") or []
        if anchors and ring.get("configured"):
            resolved = resolve_ring_from_anchors(anchors, frame_idx, camera_transforms)
            if resolved is not None:
                out.append(resolved)
                continue
        out.append(transform_ring_to_frame(ring, camera_transforms, frame_idx))
    return out


def project_ball_point_to_frame(
    x: float,
    y: float,
    anchor_frame: int,
    target_frame_idx: int,
    camera_transforms: Optional[List[np.ndarray]],
) -> Tuple[float, float]:
    """Переносит точку мяча из координат якорного кадра в target_frame_idx."""
    if not camera_transforms:
        return float(x), float(y)
    ring = {
        "x": float(x),
        "y": float(y),
        "half_width": 40.0,
        "anchor_frame": int(anchor_frame),
        "configured": True,
    }
    projected = transform_ring_to_frame(ring, camera_transforms, target_frame_idx)
    return float(projected["x"]), float(projected["y"])


def normalize_ball_anchors(anchors: Optional[List[Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for item in anchors or []:
        if not isinstance(item, dict):
            continue
        out.append(
            {
                "x": float(item["x"]),
                "y": float(item["y"]),
                "frame": int(item["frame"]),
            }
        )
    out.sort(key=lambda a: int(a["frame"]))
    return out


def add_ball_anchor(state: Dict[str, Any], x: int, y: int, frame_idx: int) -> None:
    anchors = [
        a for a in normalize_ball_anchors(state.get("ball_anchors"))
        if int(a["frame"]) != int(frame_idx)
    ]
    anchors.append({"x": float(x), "y": float(y), "frame": int(frame_idx)})
    anchors.sort(key=lambda a: int(a["frame"]))
    state["ball_anchors"] = anchors


def remove_ball_anchor_at(state: Dict[str, Any], index: int) -> None:
    anchors = normalize_ball_anchors(state.get("ball_anchors"))
    if 0 <= index < len(anchors):
        anchors.pop(index)
    state["ball_anchors"] = anchors


def interpolate_ball_position(
    anchors: List[Dict[str, Any]],
    frame_idx: int,
    camera_transforms: Optional[List[np.ndarray]] = None,
) -> Optional[Tuple[float, float, str]]:
    """Позиция мяча на frame_idx: точный якорь или линейная интерполяция между соседними."""
    sorted_anchors = normalize_ball_anchors(anchors)
    if not sorted_anchors:
        return None

    for anchor in sorted_anchors:
        if int(anchor["frame"]) == int(frame_idx):
            x, y = project_ball_point_to_frame(
                anchor["x"], anchor["y"], int(anchor["frame"]), frame_idx, camera_transforms
            )
            return x, y, "user"

    if len(sorted_anchors) == 1:
        return None

    first_frame = int(sorted_anchors[0]["frame"])
    last_frame = int(sorted_anchors[-1]["frame"])
    if frame_idx < first_frame or frame_idx > last_frame:
        return None

    for i in range(len(sorted_anchors) - 1):
        a0 = sorted_anchors[i]
        a1 = sorted_anchors[i + 1]
        f0, f1 = int(a0["frame"]), int(a1["frame"])
        if f0 <= frame_idx <= f1:
            if f1 == f0:
                x, y = project_ball_point_to_frame(a0["x"], a0["y"], f0, frame_idx, camera_transforms)
                return x, y, "user"
            t_frac = (frame_idx - f0) / float(f1 - f0)
            p0 = project_ball_point_to_frame(a0["x"], a0["y"], f0, frame_idx, camera_transforms)
            p1 = project_ball_point_to_frame(a1["x"], a1["y"], f1, frame_idx, camera_transforms)
            x = p0[0] + t_frac * (p1[0] - p0[0])
            y = p0[1] + t_frac * (p1[1] - p0[1])
            return x, y, "user_interp"
    return None


def suggest_ball_interp_check_frames(
    anchors: Optional[List[Any]],
    max_per_gap: int = BALL_INTERP_CHECKS_PER_GAP,
) -> List[int]:
    """1–3 равномерно распределённых кадра между соседними якорями для проверки интерполяции."""
    sorted_anchors = normalize_ball_anchors(anchors)
    if len(sorted_anchors) < 2:
        return []
    frames: List[int] = []
    for i in range(len(sorted_anchors) - 1):
        f0 = int(sorted_anchors[i]["frame"])
        f1 = int(sorted_anchors[i + 1]["frame"])
        gap = f1 - f0
        if gap <= 1:
            continue
        n_checks = min(max_per_gap, gap - 1, 3)
        for j in range(1, n_checks + 1):
            frame = int(round(f0 + j * gap / (n_checks + 1)))
            if f0 < frame < f1:
                frames.append(frame)
    return frames


def get_pending_ball_interp_checks(
    anchors: Optional[List[Any]],
    skipped_frames: Optional[List[int]] = None,
) -> List[int]:
    """Кадры для проверки интерполяции: предложены, но ещё не якорь и не пропущены."""
    anchor_frames = {int(a["frame"]) for a in normalize_ball_anchors(anchors)}
    skipped = {int(f) for f in (skipped_frames or [])}
    return [
        frame
        for frame in suggest_ball_interp_check_frames(anchors)
        if frame not in anchor_frames and frame not in skipped
    ]


def append_ball_training_seed(
    video_path: Optional[str],
    video_name: Optional[str],
    frame_idx: int,
    x: float,
    y: float,
    frame_bgr: Any,
    source: str = "user_click",
) -> None:
    """Дописывает разметку мяча в training_seeds/ (jsonl + кроп) для будущего обучения."""
    if cv2 is None or frame_bgr is None or frame_bgr.size == 0:
        return
    ensure_training_seeds_dir()
    h, w = frame_bgr.shape[:2]
    half = BALL_DEFAULT_BBOX_HALF
    x1 = max(int(round(x)) - half, 0)
    y1 = max(int(round(y)) - half, 0)
    x2 = min(int(round(x)) + half, w)
    y2 = min(int(round(y)) + half, h)
    crop = frame_bgr[y1:y2, x1:x2]
    stem = Path(video_path or "video").stem
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    crop_name = f"{stem}_f{int(frame_idx)}_{stamp}.jpg"
    crop_path = BALL_TRAINING_CROPS_DIR / crop_name
    if crop.size > 0:
        cv2.imwrite(str(crop_path), crop)
    record = {
        "frame_idx": int(frame_idx),
        "center": [float(x), float(y)],
        "xyxy": [float(x1), float(y1), float(x2), float(y2)],
        "video_path": str(video_path or ""),
        "video_name": str(video_name or stem),
        "crop_path": str(crop_path.relative_to(BASE_DIR)),
        "source": source,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    with open(BALL_LABELS_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def make_id_resolver(manual_id_map: Optional[Dict[int, int]]) -> Callable[[int], int]:
    """Возвращает функцию raw_tracker_id → канонический ID (с учётом цепочек склейки)."""
    mapping = {int(k): int(v) for k, v in (manual_id_map or {}).items()}

    def resolve(player_id: int) -> int:
        current = int(player_id)
        seen: set = set()
        while current in mapping:
            if current in seen:
                break
            seen.add(current)
            current = mapping[current]
        return current

    return resolve


def build_id_former_labels(manual_id_map: Optional[Dict[int, int]]) -> Dict[int, List[int]]:
    """Канонический ID → список бывших raw ID для подписи на видео (=б. 12)."""
    resolve = make_id_resolver(manual_id_map)
    labels: Dict[int, List[int]] = {}
    for raw_id in (manual_id_map or {}):
        canonical = resolve(int(raw_id))
        if int(raw_id) != canonical:
            labels.setdefault(canonical, []).append(int(raw_id))
    for canonical in labels:
        labels[canonical] = sorted(set(labels[canonical]))
    return labels


def build_player_id_groups(
    crop_ids: List[int],
    manual_id_map: Optional[Dict[int, int]] = None,
) -> Dict[int, List[int]]:
    """Канонический ID → все ID группы (канонический первым)."""
    resolve = make_id_resolver(manual_id_map)
    groups: Dict[int, List[int]] = {}
    for pid in crop_ids:
        canonical = resolve(int(pid))
        members = groups.setdefault(canonical, [])
        if int(pid) not in members:
            members.append(int(pid))
    for canonical, members in groups.items():
        ordered = sorted(members)
        if canonical in ordered:
            groups[canonical] = [canonical] + [m for m in ordered if m != canonical]
        else:
            groups[canonical] = [canonical] + ordered
    return groups


def merge_player_ids_selection(state: Dict[str, Any], selected_ids: List[int]) -> None:
    """Склеивает выбранные карточки: канонический = минимальный ID."""
    unique = sorted({int(pid) for pid in selected_ids})
    if len(unique) < 2:
        return
    canonical = unique[0]
    apply_manual_id_merge(state, canonical, [pid for pid in unique if pid != canonical])


def unmerge_player_group(state: Dict[str, Any], canonical_id: int) -> None:
    """Разъединяет группу: убирает все записи manual_id_map, ведущие к canonical_id."""
    manual_map = {int(k): int(v) for k, v in (state.get("manual_id_map") or {}).items()}
    to_remove = [raw_id for raw_id, target in manual_map.items() if int(target) == int(canonical_id)]
    for raw_id in to_remove:
        manual_map.pop(raw_id, None)
    state["manual_id_map"] = manual_map


def consume_pending_step2_ui_state(state: Dict[str, Any]) -> None:
    """Применяет отложенные изменения навигации шага 2 до создания виджетов."""
    pending_frame = state.pop("_pending_preview_frame_idx", None)
    if pending_frame is not None:
        state["preview_frame_idx"] = int(pending_frame)
    pending_target = state.pop("_pending_click_target_ring", None)
    if pending_target is not None:
        state["click_target_ring"] = pending_target
        # radio с key=click_target_ring_radio хранит своё состояние отдельно от click_target_ring
        state["click_target_ring_radio"] = pending_target
    for ring_num in (1, 2):
        for field in ("x", "y", "r"):
            pending_key = f"_pending_wi_ring{ring_num}_{field}"
            if pending_key in state:
                state[f"wi_ring{ring_num}_{field}"] = state.pop(pending_key)


def consume_pending_step3_ui_state(state: Dict[str, Any]) -> None:
    """Сбрасывает merge_pick_* до создания чекбоксов (после склейки на прошлом прогоне)."""
    to_clear = state.pop("_pending_merge_pick_clear", None)
    if not to_clear:
        return
    for pid in to_clear:
        state[f"merge_pick_{int(pid)}"] = False


def apply_manual_id_merge(state: Dict[str, Any], canonical_id: int, source_ids: List[int]) -> None:
    """Склеивает выбранные tracker ID в один канонический (шаг 3)."""
    canonical = int(canonical_id)
    manual_map = {int(k): int(v) for k, v in (state.get("manual_id_map") or {}).items()}
    merged_sources: List[int] = []
    for sid in source_ids:
        sid = int(sid)
        if sid == canonical:
            continue
        manual_map[sid] = canonical
        merged_sources.append(sid)
        for raw_id, target in list(manual_map.items()):
            if target == sid:
                manual_map[raw_id] = canonical
    resolve = make_id_resolver(manual_map)
    flattened = {raw_id: resolve(raw_id) for raw_id in manual_map if resolve(raw_id) != raw_id}
    state["manual_id_map"] = flattened
    if not merged_sources:
        return
    merge_log = list(state.get("manual_id_merge_log") or [])
    merge_log.append({"Канонический ID": canonical, "Объединены ID": sorted(merged_sources)})
    state["manual_id_merge_log"] = merge_log
    crops = dict(state.get("player_crops") or {})
    names = dict(state.get("player_names") or {})
    numbers = dict(state.get("player_numbers") or {})
    for sid in merged_sources:
        if sid in crops and canonical not in crops:
            crops[canonical] = crops[sid]
        if (names.get(sid) or "").strip() and not (names.get(canonical) or "").strip():
            names[canonical] = names[sid]
        if (numbers.get(sid) or "").strip() and not (numbers.get(canonical) or "").strip():
            numbers[canonical] = numbers[sid]
    state["player_crops"] = crops
    state["player_names"] = names
    state["player_numbers"] = numbers


def lookup_player_meta(
    canonical_id: int,
    player_names: Optional[Dict[int, str]],
    player_numbers: Optional[Dict[int, str]],
    manual_id_map: Optional[Dict[int, int]] = None,
) -> Tuple[str, str]:
    """Имя/номер для канонического ID с учётом бывших raw ID после ручной склейки."""
    names = player_names or {}
    numbers = player_numbers or {}
    resolve = make_id_resolver(manual_id_map)
    candidates = [int(canonical_id)]
    for raw_id in (manual_id_map or {}):
        if resolve(int(raw_id)) == int(canonical_id):
            candidates.append(int(raw_id))
    name = ""
    for cid in candidates:
        n = (names.get(cid) or "").strip()
        if n:
            name = n
            break
    number = ""
    for cid in candidates:
        n = (numbers.get(cid) or "").strip()
        if n:
            number = n
            break
    return name, number


def get_camera_transforms_cached(
    video_path: str,
    state: Dict[str, Any],
    frame_progress_callback: Optional[Callable[[int, int], None]] = None,
) -> List[np.ndarray]:
    """Кэширует estimate_camera_transforms в session_state для превью шага 2."""
    if state.get("camera_transforms_video") != video_path or state.get("camera_transforms") is None:
        state["camera_transforms"] = estimate_camera_transforms(
            video_path, frame_progress_callback=frame_progress_callback
        )
        state["camera_transforms_video"] = video_path
    return state["camera_transforms"]


# ---------------------------------------------------------------------------
# Разбор результатов детекции/трекинга Ultralytics и отрисовка аннотаций
# ---------------------------------------------------------------------------
def bbox_iou(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> float:
    """IoU двух bbox (x1,y1,x2,y2)."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 1e-6 else 0.0


def compute_csrt_max_jump_px(
    fps: float,
    avg_player_diagonal: Optional[float] = None,
    imgsz: int = IMGSZ_DEFAULT,
) -> float:
    """Адаптивный порог скачка CSRT: ниже FPS / мельче игроки → больше px за кадр."""
    fps = max(float(fps), 1.0)
    fps_scale = 25.0 / fps
    diag = float(avg_player_diagonal) if avg_player_diagonal and avg_player_diagonal > 0 else 120.0
    size_scale = diag / 120.0
    imgsz_scale = max(float(imgsz) / 640.0, 0.5)
    jump = BALL_CSRT_MAX_JUMP_PX * fps_scale * size_scale * math.sqrt(imgsz_scale)
    return float(np.clip(jump, 45.0, 320.0))


def ball_state_counts_for_events(
    ball_state: Optional["BallTrackState"],
    ball_lost: bool = False,
) -> bool:
    """Гол/пас только по надёжным источникам (не kalman/color/interp с нулевым conf)."""
    if ball_state is None or ball_lost:
        return False
    if ball_state.source not in EVENT_ELIGIBLE_BALL_SOURCES:
        return False
    return float(ball_state.conf) > 0.0


def effective_prev_event_ball_xy(
    prev_xy: Optional[Tuple[float, float]],
    last_event_ball_frame: int,
    frame_idx: int,
    gap_max_frames: int,
    camera_transforms: Optional[List[np.ndarray]] = None,
) -> Optional[Tuple[float, float]]:
    """Сбрасывает предыдущую event_ball-точку, если дыра между кадрами слишком велика."""
    if prev_xy is None:
        return None
    if frame_idx - last_event_ball_frame > gap_max_frames:
        return None
    if camera_transforms is not None and last_event_ball_frame != frame_idx:
        return transform_point_between_frames(
            prev_xy[0],
            prev_xy[1],
            last_event_ball_frame,
            frame_idx,
            camera_transforms,
        )
    return prev_xy


def event_ball_from_tracking(
    model: Any,
    ball_state: Optional["BallTrackState"],
    ball_lost: bool,
    ball: Optional[Tuple[float, float]],
) -> Optional[Tuple[float, float]]:
    """Координаты мяча для гол/пас — только при загруженной модели и надёжном source."""
    if model is None:
        return None
    if ball_state_counts_for_events(ball_state, ball_lost):
        return ball
    return None


def seconds_to_frames(seconds: float, fps: float, minimum: int = 1) -> int:
    return max(int(round(float(seconds) * float(fps))), minimum)


def parse_track_results(
    results,
    person_conf_threshold: float = 0.0,
    ball_conf_threshold: float = 0.0,
) -> Tuple[
    List[Tuple[int, Tuple[float, float, float, float]]],
    Optional[Tuple[float, float]],
    float,
    Optional[Tuple[float, float, float, float]],
]:
    """Извлекает из результата YOLO список игроков (ID, рамка) и центр мяча.

    Координаты возвращаются в системе координат кадра, который был передан
    в model.track() — если перед детекцией применялось "улучшение качества"
    (апскейл), их нужно масштабировать обратно (см. process_video).

    person_conf_threshold/ball_conf_threshold — постфактум-фильтрация по
    классам: инференс (см. вызывающий код) намеренно идёт с ЕДИНЫМ низким
    conf (не выше минимального из двух порогов), чтобы не потерять мяч на
    этапе NMS модели, а затем каждый класс фильтруется своим порогом здесь —
    так игроки не "засоряются" ложными детекциями низкой уверенности, а мяч
    при этом всё ещё может быть обнаружен на низком пороге.
    """
    result = results[0]
    persons: List[Tuple[int, Tuple[float, float, float, float]]] = []
    ball: Optional[Tuple[float, float]] = None
    ball_bbox: Optional[Tuple[float, float, float, float]] = None

    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return persons, ball, 0.0, ball_bbox

    xyxy = boxes.xyxy.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)
    confs = boxes.conf.cpu().numpy()
    ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None

    best_ball_conf = -1.0
    for i, (box, c, conf) in enumerate(zip(xyxy, cls, confs)):
        conf = float(conf)
        if c == COCO_BALL_CLASS_ID:
            if conf < ball_conf_threshold:
                continue
            if conf > best_ball_conf:
                best_ball_conf = conf
                ball = ((float(box[0]) + float(box[2])) / 2.0, (float(box[1]) + float(box[3])) / 2.0)
                ball_bbox = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
        elif c == COCO_PERSON_CLASS_ID and ids is not None:
            if conf < person_conf_threshold:
                continue
            persons.append(
                (int(ids[i]), (float(box[0]), float(box[1]), float(box[2]), float(box[3])))
            )

    return persons, ball, best_ball_conf if ball is not None else 0.0, ball_bbox


def id_to_color(pid: int) -> Tuple[int, int, int]:
    """Детерминированный BGR-цвет по ID трека — чтобы игроки визуально
    отличались друг от друга на аннотированном видео."""
    hue = (int(pid) * 47) % 180
    hsv_pixel = np.uint8([[[hue, 220, 255]]])
    bgr_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0][0]
    return int(bgr_pixel[0]), int(bgr_pixel[1]), int(bgr_pixel[2])


_CYRILLIC_FONT_PATH: Optional[str] = None
_CYRILLIC_FONT_OBJECTS: Dict[int, ImageFont.FreeTypeFont] = {}


def resolve_cyrillic_font_path() -> Optional[str]:
    """DejaVuSans / Arial / Liberation — первый доступный TTF с кириллицей."""
    global _CYRILLIC_FONT_PATH
    if _CYRILLIC_FONT_PATH is not None:
        return _CYRILLIC_FONT_PATH
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).is_file():
            _CYRILLIC_FONT_PATH = candidate
            return candidate
    return None


def _get_cyrillic_font(size: int) -> Optional[ImageFont.FreeTypeFont]:
    path = resolve_cyrillic_font_path()
    if path is None:
        return None
    if size not in _CYRILLIC_FONT_OBJECTS:
        try:
            _CYRILLIC_FONT_OBJECTS[size] = ImageFont.truetype(path, size)
        except Exception:
            return None
    return _CYRILLIC_FONT_OBJECTS[size]


def draw_text_on_bgr(
    img_bgr: Any,
    text: str,
    xy: Tuple[int, int],
    font_size: int = 20,
    color_bgr: Tuple[int, int, int] = (255, 255, 255),
    outline_bgr: Optional[Tuple[int, int, int]] = (0, 0, 0),
    outline_width: int = 2,
) -> None:
    """Рисует Unicode/кириллицу на BGR-кадре (нижний левый угол текста в xy, как cv2.putText)."""
    if cv2 is None or not text:
        return
    font = _get_cyrillic_font(font_size)
    if font is None:
        cv2.putText(
            img_bgr, text, xy, cv2.FONT_HERSHEY_SIMPLEX, font_size / 30.0, color_bgr, 2, cv2.LINE_AA
        )
        return
    x, y = int(xy[0]), int(xy[1])
    rgb = (int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0]))
    outline_rgb = None
    if outline_bgr is not None:
        outline_rgb = (int(outline_bgr[2]), int(outline_bgr[1]), int(outline_bgr[0]))
    pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_h = bbox[3] - bbox[1]
    top_y = y - text_h
    if outline_rgb and outline_width > 0:
        for dx in range(-outline_width, outline_width + 1):
            for dy in range(-outline_width, outline_width + 1):
                if dx * dx + dy * dy <= outline_width * outline_width:
                    draw.text((x + dx, top_y + dy), text, font=font, fill=outline_rgb)
    draw.text((x, top_y), text, font=font, fill=rgb)
    img_bgr[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def draw_ball_interp_check_marker(preview: Any, x: float, y: float) -> None:
    """Яркий маркер предсказанной интерполированной позиции мяча на превью."""
    if cv2 is None:
        return
    center = (int(round(x)), int(round(y)))
    cv2.circle(preview, center, 15, (0, 0, 0), 3, lineType=cv2.LINE_AA)
    cv2.circle(preview, center, 12, BALL_PREVIEW_INTERP_COLOR_BGR, -1, lineType=cv2.LINE_AA)
    cross = 9
    cv2.line(
        preview, (center[0] - cross, center[1]), (center[0] + cross, center[1]),
        (255, 255, 255), 2, lineType=cv2.LINE_AA,
    )
    cv2.line(
        preview, (center[0], center[1] - cross), (center[0], center[1] + cross),
        (255, 255, 255), 2, lineType=cv2.LINE_AA,
    )
    draw_text_on_bgr(
        preview, "интерпол.", (center[0] + 18, max(center[1] - 10, 20)),
        font_size=18, color_bgr=BALL_PREVIEW_INTERP_COLOR_BGR,
        outline_bgr=(0, 0, 0), outline_width=2,
    )


def any_ring_configured(state: Dict[str, Any]) -> bool:
    return bool(state.get("ring1_configured") or state.get("ring2_configured"))


def ring_configuration_status(state: Dict[str, Any]) -> Tuple[bool, bool]:
    return bool(state.get("ring1_configured")), bool(state.get("ring2_configured"))


def _set_pending_ring_widget_coords(
    state: Dict[str, Any], ring_num: int, x: int, y: int, r: Optional[float] = None
) -> None:
    """Откладывает синхронизацию wi_ring* до consume_pending_step2_ui_state (до виджетов)."""
    state[f"_pending_wi_ring{ring_num}_x"] = int(x)
    state[f"_pending_wi_ring{ring_num}_y"] = int(y)
    if r is not None:
        state[f"_pending_wi_ring{ring_num}_r"] = int(round(r))


def mark_ring_configured(
    state: Dict[str, Any], ring_num: int, x: int, y: int, frame_idx: Optional[int] = None
) -> None:
    state[f"ring{ring_num}_configured"] = True
    state[f"ring{ring_num}_x"] = int(x)
    state[f"ring{ring_num}_y"] = int(y)
    _set_pending_ring_widget_coords(state, ring_num, int(x), int(y))
    if frame_idx is not None:
        state[f"ring{ring_num}_frame"] = int(frame_idx)


def init_ring_widget_keys(state: Dict[str, Any], ring_num: int) -> None:
    for field in ("x", "y", "r"):
        wkey = f"wi_ring{ring_num}_{field}"
        if wkey not in state:
            state[wkey] = int(state.get(f"ring{ring_num}_{field}", 0))


def sync_ring_widgets_to_canonical(state: Dict[str, Any], ring_num: int) -> None:
    prefix = f"ring{ring_num}"
    for field in ("x", "y", "r"):
        wkey = f"wi_{prefix}_{field}"
        if wkey in state:
            state[f"{prefix}_{field}"] = state[wkey]


def apply_default_ring_positions(state: Dict[str, Any], width: int, height: int) -> None:
    """Подставляет разумные дефолты для превью — без флага configured."""
    ring1, ring2 = default_ring_zones(width, height)
    for ring_num, ring in ((1, ring1), (2, ring2)):
        state[f"ring{ring_num}_x"] = int(ring["x"])
        state[f"ring{ring_num}_y"] = int(ring["y"])
        state[f"ring{ring_num}_r"] = int(ring.get("half_width", ring.get("r", 40)))
        init_ring_widget_keys(state, ring_num)
        state[f"wi_ring{ring_num}_x"] = int(ring["x"])
        state[f"wi_ring{ring_num}_y"] = int(ring["y"])
        state[f"wi_ring{ring_num}_r"] = int(ring.get("half_width", ring.get("r", 40)))
        state.setdefault(f"ring{ring_num}_frame", 0)


def ensure_ring_zones_for_video(video_path: str, state: Dict[str, Any]) -> None:
    """Дефолты только если пользователь ещё не задавал кольца; configured не затираем."""
    if state.get("rings_initialized_for") == video_path or cv2 is None:
        return
    if any_ring_configured(state):
        state["rings_initialized_for"] = video_path
        return
    meta = get_video_metadata(video_path)
    apply_default_ring_positions(state, int(meta["width"]), int(meta["height"]))
    state["rings_initialized_for"] = video_path


def rings_for_preview_display(
    state: Dict[str, Any],
    preview_frame_idx: int,
    camera_transforms: Optional[List[np.ndarray]] = None,
    panning_mode: bool = False,
) -> List[RingZone]:
    """Кольца для превью: в динамике — интерполяция/warp списка якорей на текущий кадр."""
    rings = rings_from_session_state(state)
    display: List[RingZone] = []
    for ring in rings:
        if ring.get("configured") and panning_mode and camera_transforms:
            anchors = ring.get("anchors") or []
            if anchors:
                projected = resolve_ring_from_anchors(anchors, preview_frame_idx, camera_transforms)
            else:
                projected = transform_ring_to_frame(ring, camera_transforms, preview_frame_idx)
            if projected is not None:
                projected["configured"] = True
                display.append(projected)
            else:
                display.append(dict(ring))
        elif ring.get("configured"):
            display.append(dict(ring))
        else:
            copy = dict(ring)
            if float(copy.get("y", 0.0)) > 0.0:
                copy["configured"] = True
            display.append(copy)
    return display


def rings_from_session_state(state: Dict[str, Any]) -> List[RingZone]:
    """Собирает зоны колец из session_state (с списком якорей для динамического видео)."""
    rings: List[RingZone] = []
    for ring_num in (1, 2):
        anchors = ring_anchors_from_state(state, ring_num)
        configured = bool(state.get(f"ring{ring_num}_configured"))
        if anchors:
            primary = anchors[0]
            rings.append(
                {
                    "x": float(primary["x"]),
                    "y": float(primary["y"]),
                    "half_width": float(primary["half_width"]),
                    "configured": configured,
                    "anchor_frame": int(primary["frame"]),
                    "anchors": anchors,
                }
            )
        else:
            rings.append(
                {
                    "x": float(state[f"ring{ring_num}_x"]),
                    "y": float(state[f"ring{ring_num}_y"]),
                    "half_width": float(state[f"ring{ring_num}_r"]),
                    "configured": configured,
                    "anchor_frame": int(state.get(f"ring{ring_num}_frame", 0)),
                    "anchors": [],
                }
            )
    return rings


def describe_ring_state_for_debug(state: Dict[str, Any]) -> str:
    r1, r2 = ring_configuration_status(state)
    n1 = len(ring_anchors_from_state(state, 1))
    n2 = len(ring_anchors_from_state(state, 2))
    return (
        f"ring1_configured={r1}, anchors={n1}, anchor_frame={state.get('ring1_frame')}, "
        f"ring1_x/y/r=({state.get('ring1_x')}, {state.get('ring1_y')}, {state.get('ring1_r')}); "
        f"ring2_configured={r2}, anchors={n2}, anchor_frame={state.get('ring2_frame')}, "
        f"ring2_x/y/r=({state.get('ring2_x')}, {state.get('ring2_y')}, {state.get('ring2_r')})"
    )


def prepare_rings_for_drawing(
    rings: List[RingZone], frame_w: int, frame_h: int
) -> Tuple[List[RingZone], List[str]]:
    """Нормализует координаты колец; «не задано» — только если configured=False."""
    warnings: List[str] = []
    prepared: List[RingZone] = []
    if not rings:
        warnings.append("Зоны колец не заданы — линии не рисуются.")
        return prepared, warnings

    if not any(bool(r.get("configured", True)) for r in rings):
        warnings.append(
            "Кольца не заданы пользователем (ring1_configured и ring2_configured = False) — линии не рисуются."
        )
        return prepared, warnings

    for idx, ring in enumerate(rings):
        label = f"Кольцо {idx + 1}"
        configured = bool(ring.get("configured", True))
        if not configured:
            continue

        x = float(ring.get("x", 0.0))
        y = float(ring.get("y", 0.0))
        half_w = float(ring.get("half_width", ring.get("r", 40.0)))
        if half_w <= 0.0:
            half_w = 40.0
            warnings.append(f"{label}: полуширина была 0 — использован fallback {int(half_w)} px.")

        half_w = max(half_w, 5.0)
        x = float(np.clip(x, 0, max(frame_w - 1, 0)))
        y = float(np.clip(y, 0, max(frame_h - 1, 0)))
        prepared.append({"x": x, "y": y, "half_width": half_w})
    return prepared, warnings


def draw_hoop_lines_on_frame(
    frame: Any,
    rings: List[RingZone],
    warnings_out: Optional[List[str]] = None,
) -> Any:
    """Рисует горизонтальные линии колец поверх всех остальных аннотаций.

    Координаты должны быть в системе исходного кадра (не уменьшенного превью).
    """
    if cv2 is None or frame is None:
        return frame
    h, w = frame.shape[:2]
    prepared, warnings = prepare_rings_for_drawing(rings, w, h)
    if warnings_out is not None:
        warnings_out.extend(warnings)

    for idx, ring in enumerate(prepared):
        color = HOOP_LINE_COLORS_BGR[idx % len(HOOP_LINE_COLORS_BGR)]
        center_x = int(round(ring["x"]))
        line_y = int(round(ring["y"]))
        half_w = int(round(ring["half_width"]))
        x1 = max(center_x - half_w, 0)
        x2 = min(center_x + half_w, w - 1)
        if x2 <= x1:
            x1, x2 = 0, w - 1
        _draw_line_outlined(
            frame, (x1, line_y), (x2, line_y), color, HOOP_LINE_THICKNESS,
            outline_thickness=HOOP_LINE_OUTLINE_THICKNESS,
        )
        cv2.drawMarker(
            frame,
            (center_x, line_y),
            (0, 0, 0),
            markerType=cv2.MARKER_CROSS,
            markerSize=22,
            thickness=HOOP_LINE_OUTLINE_THICKNESS // 2,
        )
        cv2.drawMarker(
            frame,
            (center_x, line_y),
            color,
            markerType=cv2.MARKER_CROSS,
            markerSize=18,
            thickness=3,
        )
        label = f"Кольцо {idx + 1}"
        label_pos = (max(center_x - 55, 4), max(line_y - 28, 24))
        draw_text_on_bgr(frame, label, label_pos, font_size=22, color_bgr=color, outline_bgr=(0, 0, 0), outline_width=3)
    return frame


def test_draw_hoop_lines_on_frame() -> bool:
    """Синтетическая проверка: после отрисовки пиксели линии ненулевые."""
    if cv2 is None:
        return True
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    rings = [{"x": 320.0, "y": 120.0, "half_width": 80.0, "configured": True}]
    draw_hoop_lines_on_frame(frame, rings)
    line_y = 120
    segment = frame[line_y, 240:401]
    return bool(np.any(segment))


def _draw_line_outlined(
    img: Any,
    p1: Tuple[int, int],
    p2: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int,
    outline_thickness: Optional[int] = None,
) -> None:
    outline = outline_thickness if outline_thickness is not None else thickness + 4
    cv2.line(img, p1, p2, BALL_TRAJECTORY_OUTLINE_BGR, outline, cv2.LINE_AA)
    cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)


def draw_annotations(
    frame,
    persons: List[Tuple[int, Tuple[float, float, float, float]]],
    ball: Optional[Tuple[float, float]],
    ball_trajectory: Optional[List[Tuple[float, float]]] = None,
    ball_source: Optional[str] = None,
    excluded_ids: Optional[set] = None,
    ball_lost: bool = False,
    last_ball: Optional[Tuple[float, float]] = None,
    id_former_labels: Optional[Dict[int, List[int]]] = None,
):
    """Рисует рамки игроков (с ID), траекторию мяча и маркер мяча на копии кадра.

    Заменяет result.plot() из ultralytics: при включённом "улучшении
    качества" детекция идёт на увеличенном кадре, а рисовать нужно на
    ОРИГИНАЛЬНОМ (после пересчёта координат обратно) — иначе выходное видео
    получилось бы в другом разрешении, чем входное.
    """
    annotated = frame.copy()
    excluded_ids = excluded_ids or set()
    id_former_labels = id_former_labels or {}
    traj_pts: List[Tuple[int, int]] = []
    if ball_trajectory:
        traj_pts = [(int(x), int(y)) for x, y in ball_trajectory]
    if ball is not None:
        traj_pts.append((int(ball[0]), int(ball[1])))
    if len(traj_pts) >= 2:
        for i in range(1, len(traj_pts)):
            _draw_line_outlined(
                annotated, traj_pts[i - 1], traj_pts[i],
                BALL_TRAJECTORY_COLOR_BGR, BALL_TRAJECTORY_THICKNESS,
            )
    for pid, (x1, y1, x2, y2) in persons:
        excluded = pid in excluded_ids
        color = (128, 128, 128) if excluded else id_to_color(pid)
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        thickness = 1 if excluded else 2
        cv2.rectangle(annotated, p1, p2, color, thickness)
        former = id_former_labels.get(pid) or []
        label = f"ID {pid}"
        if former:
            label += f" (=б. {', '.join(str(x) for x in former)})"
        if excluded:
            label += " [excl]"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        label_y1 = max(p1[1] - th - 8, 0)
        cv2.rectangle(annotated, (p1[0], label_y1), (p1[0] + tw + 6, p1[1]), color, -1)
        cv2.putText(
            annotated, label, (p1[0] + 3, max(p1[1] - 5, th)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
        )
    draw_ball = ball
    src = ball_source or ("lost" if ball_lost else "yolo")
    if draw_ball is None and ball_lost and last_ball is not None:
        draw_ball = last_ball
    if draw_ball is not None:
        center = (int(draw_ball[0]), int(draw_ball[1]))
        color = BALL_SOURCE_COLORS_BGR.get(src, (0, 215, 255))
        if ball_lost and ball is None:
            overlay = annotated.copy()
            cv2.circle(overlay, center, 12, color, -1)
            annotated = cv2.addWeighted(overlay, 0.45, annotated, 0.55, 0)
            cv2.circle(annotated, center, 12, color, 2)
            label = "ball/lost"
        else:
            cv2.circle(annotated, center, 11, (0, 0, 0), -1)
            cv2.circle(annotated, center, 9, color, -1)
            cv2.circle(annotated, center, 9, (0, 0, 0), 2)
            label = f"ball/{src}"
        cv2.putText(
            annotated, label, (center[0] + 14, center[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA,
        )
    return annotated


def enhance_frame_for_detection(frame):
    """Апскейл + лёгкая резкость перед детекцией (опция "улучшить качество").

    Может помочь трекеру/ReID отличать игроков на видео низкого разрешения
    за счёт более крупных/чётких признаков на входе модели, но НЕ панацея:
    если исходное видео сильно сжато или изначально низкого качества, апскейл
    не восстановит потерянные детали.
    """
    upscaled = cv2.resize(
        frame, None, fx=ENHANCE_UPSCALE_FACTOR, fy=ENHANCE_UPSCALE_FACTOR, interpolation=cv2.INTER_LANCZOS4
    )
    blurred = cv2.GaussianBlur(upscaled, (0, 0), sigmaX=1.0)
    sharpened = cv2.addWeighted(upscaled, 1.5, blurred, -0.5, 0)
    return sharpened


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


def reencode_for_browser(path: Path) -> Path:
    """Перекодирует mp4 в H.264 (yuv420p, +faststart) через системный ffmpeg.

    cv2.VideoWriter пишет валидный MP4, но кодеком по умолчанию (обычно
    mp4v/MPEG-4 Part 2), который многие браузеры НЕ умеют проигрывать через
    HTML5 <video> (а именно так работает st.video) — файл при этом открывается
    внешними плеерами, но выглядит "битым" во встроенном плеере Streamlit.
    H.264 поддерживается практически всеми браузерами. Если ffmpeg не найден
    в системе или перекодирование не удалось — возвращаем исходный файл без
    изменений; GUI в этом случае покажет предупреждение и кнопку скачивания.
    """
    if not FFMPEG_PATH or not path.exists() or path.stat().st_size == 0:
        return path
    tmp_out = path.with_name(path.stem + "_h264" + path.suffix)
    try:
        result = subprocess.run(
            [
                FFMPEG_PATH, "-y", "-i", str(path),
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an",
                str(tmp_out),
            ],
            capture_output=True,
            timeout=600,
        )
        if result.returncode == 0 and tmp_out.exists() and tmp_out.stat().st_size > 0:
            path.unlink(missing_ok=True)
            tmp_out.rename(path)
        else:
            tmp_out.unlink(missing_ok=True)
    except Exception:
        tmp_out.unlink(missing_ok=True)
    return path


def save_highlight_clip(highlight: PendingHighlight, fps: float, width: int, height: int) -> Path:
    """Сохраняет буфер прошлого + будущего в автономный MP4-файл в highlights/."""
    frames = highlight.past_frames + highlight.future_frames
    out_path = HIGHLIGHTS_DIR / highlight.filename
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps if fps > 1 else 25.0, (width, height))
    for frame in frames:
        writer.write(frame)
    writer.release()
    return reencode_for_browser(out_path)


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


def extract_frame_at_index(video_path: str, frame_idx: int):
    """Извлекает кадр по индексу (для превью настройки колец)."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(int(frame_idx), 0))
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def extract_frame_at_time(video_path: str, t_seconds: float):
    """Извлекает один кадр видео на заданной секунде (для превью настройки зон)."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_idx = max(int(t_seconds * fps), 0)
    cap.release()
    return extract_frame_at_index(video_path, frame_idx)


def default_ring_zones(width: int, height: int) -> Tuple[RingZone, RingZone]:
    """Разумные дефолтные горизонтальные линии двух колец (по краям площадки)."""
    half_w = max(int(min(width, height) * 0.08), 40)
    ring1 = {"x": float(int(width * 0.10)), "y": float(int(height * 0.35)), "half_width": float(half_w)}
    ring2 = {"x": float(int(width * 0.90)), "y": float(int(height * 0.35)), "half_width": float(half_w)}
    return ring1, ring2


def draw_ball_anchors_on_preview(
    preview: Any,
    ball_anchors: List[Dict[str, Any]],
    preview_frame_idx: int,
    camera_transforms: Optional[List[np.ndarray]] = None,
    panning_mode: bool = False,
) -> None:
    """Оранжевые маркеры якорей мяча и интерполированная позиция на текущем кадре превью."""
    if cv2 is None or not ball_anchors:
        return
    transforms = camera_transforms if panning_mode else None
    for anchor in normalize_ball_anchors(ball_anchors):
        if int(anchor["frame"]) != int(preview_frame_idx):
            continue
        cx = int(round(anchor["x"]))
        cy = int(round(anchor["y"]))
        cv2.circle(preview, (cx, cy), 12, (0, 0, 0), 2, lineType=cv2.LINE_AA)
        cv2.circle(preview, (cx, cy), 10, BALL_PREVIEW_COLOR_BGR, -1, lineType=cv2.LINE_AA)
        draw_text_on_bgr(
            preview, "мяч", (cx + 14, max(cy - 16, 18)),
            font_size=16, color_bgr=BALL_PREVIEW_COLOR_BGR,
            outline_bgr=(0, 0, 0), outline_width=2,
        )

    interp = interpolate_ball_position(ball_anchors, preview_frame_idx, transforms)
    if interp is None:
        return
    ix, iy, kind = interp
    center = (int(round(ix)), int(round(iy)))
    if kind == "user":
        return
    color = BALL_PREVIEW_INTERP_COLOR_BGR
    cv2.circle(preview, center, 11, (0, 0, 0), 2, lineType=cv2.LINE_AA)
    cv2.circle(preview, center, 9, color, 2, lineType=cv2.LINE_AA)
    cross = 7
    cv2.line(
        preview, (center[0] - cross, center[1]), (center[0] + cross, center[1]),
        color, 1, lineType=cv2.LINE_AA,
    )
    cv2.line(
        preview, (center[0], center[1] - cross), (center[0], center[1] + cross),
        color, 1, lineType=cv2.LINE_AA,
    )
    draw_text_on_bgr(
        preview, "мяч↔", (center[0] + 12, max(center[1] - 18, 16)),
        font_size=15, color_bgr=color, outline_bgr=(0, 0, 0), outline_width=2,
    )


def draw_zones_preview(
    frame,
    rings: List[RingZone],
    possession_threshold: Optional[float] = None,
    preview_frame_idx: Optional[int] = None,
    show_anchor_debug: bool = False,
    ball_anchors: Optional[List[Dict[str, Any]]] = None,
    camera_transforms: Optional[List[np.ndarray]] = None,
    panning_mode: bool = False,
) -> Any:
    """Рисует горизонтальные линии колец на копии кадра для наглядной проверки в GUI.

    Если передан possession_threshold — дополнительно рисует в углу кадра
    эталонный полупрозрачный круг такого радиуса с подписью в пикселях, чтобы
    пользователь видел порог владения мячом в реальном масштабе кадра, а не
    гадал по числу пикселей.

    show_anchor_debug — крест якоря и подпись «як.A→t» в режиме динамической камеры.
    """
    preview = frame.copy()
    h, w = preview.shape[:2]
    prepared, _ = prepare_rings_for_drawing(rings, w, h)
    for idx, ring in enumerate(prepared):
        color = HOOP_LINE_COLORS_BGR[idx % len(HOOP_LINE_COLORS_BGR)]
        source = rings[idx] if idx < len(rings) else ring
        center_x = int(round(ring["x"]))
        line_y = int(round(ring["y"]))
        half_w = int(round(ring["half_width"]))
        x1 = max(center_x - half_w, 0)
        x2 = min(center_x + half_w, w - 1)
        if x2 <= x1:
            x1, x2 = 0, w - 1
        _draw_line_outlined(
            preview, (x1, line_y), (x2, line_y), color, HOOP_LINE_THICKNESS,
            outline_thickness=HOOP_LINE_OUTLINE_THICKNESS,
        )
        label_pos = (max(center_x - 45, 0), max(line_y - 18, 20))
        draw_text_on_bgr(
            preview, f"Кольцо {idx + 1}", label_pos, font_size=20, color_bgr=color,
            outline_bgr=(0, 0, 0), outline_width=3,
        )
        if show_anchor_debug and preview_frame_idx is not None:
            anchor_frame = int(source.get("anchor_frame", preview_frame_idx))
            cross = 10
            cv2.line(
                preview, (center_x - cross, line_y), (center_x + cross, line_y),
                (255, 255, 255), 1, lineType=cv2.LINE_AA,
            )
            cv2.line(
                preview, (center_x, line_y - cross), (center_x, line_y + cross),
                (255, 255, 255), 1, lineType=cv2.LINE_AA,
            )
            cv2.circle(preview, (center_x, line_y), 4, color, -1, lineType=cv2.LINE_AA)
            debug_label = f"як.{anchor_frame}→{preview_frame_idx}"
            draw_text_on_bgr(
                preview, debug_label,
                (min(center_x + 12, w - 120), max(line_y - 28, 16)),
                font_size=16, color_bgr=(255, 255, 255),
                outline_bgr=(0, 0, 0), outline_width=2,
            )

    if ball_anchors and preview_frame_idx is not None:
        draw_ball_anchors_on_preview(
            preview, ball_anchors, preview_frame_idx, camera_transforms, panning_mode
        )

    if possession_threshold and possession_threshold > 0:
        h, w = preview.shape[:2]
        r = int(possession_threshold)
        margin = r + 20
        hint_center = (min(margin, w - 1), max(h - margin, 1))
        overlay = preview.copy()
        cv2.circle(overlay, hint_center, r, (0, 255, 0), -1)
        preview = cv2.addWeighted(overlay, 0.18, preview, 0.82, 0)
        cv2.circle(preview, hint_center, r, (0, 255, 0), 2, lineType=cv2.LINE_AA)
        label = f"Порог владения: {r}px"
        draw_text_on_bgr(
            preview, label,
            (min(margin - r, w - 10), max(h - margin - r - 10, 20)),
            font_size=18, color_bgr=(0, 255, 0), outline_bgr=(0, 0, 0), outline_width=2,
        )
    return preview


def resize_crop_to_height(crop: Any, target_height: int = CROP_DISPLAY_HEIGHT) -> Any:
    """Приводит кроп игрока к фиксированной высоте с сохранением пропорций —
    чтобы сетка кропов на шаге 3 выглядела аккуратно и компактно."""
    h, w = crop.shape[:2]
    if h <= 0 or w <= 0:
        return crop
    scale = target_height / float(h)
    new_w = max(int(round(w * scale)), 1)
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(crop, (new_w, target_height), interpolation=interpolation)


def estimate_player_scale(
    frame, model, device: str, person_conf: float = PERSON_CONF_DEFAULT, imgsz: int = IMGSZ_DEFAULT
) -> Optional[float]:
    """Быстрый однократный проход детектора по кадру (без трекинга) — считает
    средний размер (диагональ рамки) игроков, чтобы предложить адаптивный
    дефолт порога владения мячом под масштаб конкретного видео вместо
    фиксированных пикселей."""
    if model is None or frame is None:
        return None
    try:
        results = model.predict(
            frame, classes=[COCO_PERSON_CLASS_ID], conf=person_conf, imgsz=imgsz, device=device, verbose=False
        )
    except Exception:
        return None
    if not results:
        return None
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return None
    xyxy = boxes.xyxy.cpu().numpy()
    diagonals = [math.hypot(x2 - x1, y2 - y1) for x1, y1, x2, y2 in xyxy]
    if not diagonals:
        return None
    return float(np.mean(diagonals))


def suggest_possession_threshold(avg_player_diagonal: Optional[float]) -> int:
    """Переводит средний размер игрока в рекомендованный порог владения мячом."""
    if not avg_player_diagonal or avg_player_diagonal <= 0:
        return POSSESSION_THRESHOLD_DEFAULT
    suggestion = avg_player_diagonal * POSSESSION_THRESHOLD_DIAGONAL_FRACTION
    return int(np.clip(suggestion, POSSESSION_THRESHOLD_MIN, POSSESSION_THRESHOLD_MAX))


# ---------------------------------------------------------------------------
# Устойчивый трекинг мяча: Kalman + CSRT + tiled/ROI YOLO + цветовой fallback
# ---------------------------------------------------------------------------
@dataclass
class BallTrackState:
    x: float
    y: float
    conf: float
    source: str  # yolo | csrt | tiled | roi | color | interp | kalman | user | user_interp


class BallAnchorGuide:
    """Пользовательские клики по мячу: приоритет на якорных кадрах и интерполяция между ними."""

    def __init__(
        self,
        anchors: Optional[List[Dict[str, Any]]] = None,
        camera_transforms: Optional[List[np.ndarray]] = None,
    ) -> None:
        self.anchors = normalize_ball_anchors(anchors)
        self.camera_transforms = camera_transforms

    def get_override(self, frame_idx: int) -> Optional[BallTrackState]:
        """Только точный кадр клика — не подменяет YOLO/CSRT между якорями."""
        hit = interpolate_ball_position(self.anchors, frame_idx, self.camera_transforms)
        if hit is None:
            return None
        x, y, source = hit
        if source != "user":
            return None
        return BallTrackState(x, y, 1.0, source)

    def get_csrt_seed_bbox(self, frame_idx: int) -> Optional[Tuple[float, float, float, float]]:
        """Seed CSRT только на кадре клика и коротком хвосте после него."""
        if not self.anchors:
            return None
        best: Optional[Dict[str, Any]] = None
        best_delta: Optional[int] = None
        for anchor in self.anchors:
            delta = int(frame_idx) - int(anchor["frame"])
            if 0 <= delta <= BALL_CSRT_SEED_TAIL_FRAMES:
                if best_delta is None or delta < best_delta:
                    best = anchor
                    best_delta = delta
        if best is None:
            return None
        x, y = project_ball_point_to_frame(
            best["x"], best["y"], int(best["frame"]), frame_idx, self.camera_transforms
        )
        return _bbox_from_center(x, y)


class BallKalmanFilter:
    """Простой Kalman (x,y,vx,vy) для предсказания позиции мяча между детекциями."""

    def __init__(self) -> None:
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32
        )
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.05
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.8
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        self.initialized = False

    def init(self, x: float, y: float) -> None:
        self.kf.statePost = np.array([[x], [y], [0.0], [0.0]], dtype=np.float32)
        self.initialized = True

    def predict(self) -> Optional[Tuple[float, float]]:
        if not self.initialized:
            return None
        pred = self.kf.predict()
        return float(pred[0]), float(pred[1])

    def correct(self, x: float, y: float) -> None:
        if not self.initialized:
            self.init(x, y)
            return
        self.kf.correct(np.array([[x], [y]], dtype=np.float32))


@dataclass
class BallColorCalibration:
    """HSV-диапазоны мяча, откалиброванные по кликам-якорям шага 2."""

    lower1: np.ndarray
    upper1: np.ndarray
    lower2: Optional[np.ndarray] = None
    upper2: Optional[np.ndarray] = None


def build_ball_color_calibration_from_anchors(
    video_path: str,
    anchors: Optional[List[Any]],
    camera_transforms: Optional[List[np.ndarray]] = None,
    patch_radius: int = 12,
) -> Optional[BallColorCalibration]:
    """Строит HSV-маску мяча по кликам пользователя; без якорей — None (fallback на оранжевый)."""
    if cv2 is None:
        return None
    norm = normalize_ball_anchors(anchors)
    if not norm:
        return None
    h_vals: List[float] = []
    s_vals: List[float] = []
    v_vals: List[float] = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    try:
        for anchor in norm:
            fidx = int(anchor["frame"])
            cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ret, frame = cap.read()
            if not ret:
                continue
            x, y = project_ball_point_to_frame(
                anchor["x"], anchor["y"], fidx, fidx, camera_transforms
            )
            cx, cy = int(round(x)), int(round(y))
            fh, fw = frame.shape[:2]
            x1, y1 = max(cx - patch_radius, 0), max(cy - patch_radius, 0)
            x2, y2 = min(cx + patch_radius, fw), min(cy + patch_radius, fh)
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            patch = frame[y1:y2, x1:x2]
            hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
            h_vals.extend(hsv[:, :, 0].astype(np.float32).flatten().tolist())
            s_vals.extend(hsv[:, :, 1].astype(np.float32).flatten().tolist())
            v_vals.extend(hsv[:, :, 2].astype(np.float32).flatten().tolist())
    finally:
        cap.release()
    if len(h_vals) < 16:
        return None
    h_arr = np.array(h_vals, dtype=np.float32)
    s_arr = np.array(s_vals, dtype=np.float32)
    v_arr = np.array(v_vals, dtype=np.float32)
    h_med = float(np.median(h_arr))
    h_pad = 16.0
    s_lo = float(np.clip(np.percentile(s_arr, 8) - 30, 25, 255))
    s_hi = float(np.clip(np.percentile(s_arr, 92) + 30, 0, 255))
    v_lo = float(np.clip(np.percentile(v_arr, 8) - 35, 30, 255))
    v_hi = float(np.clip(np.percentile(v_arr, 92) + 35, 0, 255))
    if h_med < 12:
        lower1 = np.array([0, s_lo, v_lo], dtype=np.uint8)
        upper1 = np.array([min(180, h_med + h_pad), s_hi, v_hi], dtype=np.uint8)
        lower2 = np.array([max(0, 180 - (12 - h_med)), s_lo, v_lo], dtype=np.uint8)
        upper2 = np.array([179, s_hi, v_hi], dtype=np.uint8)
        return BallColorCalibration(lower1=lower1, upper1=upper1, lower2=lower2, upper2=upper2)
    lower1 = np.array([max(0, h_med - h_pad), s_lo, v_lo], dtype=np.uint8)
    upper1 = np.array([min(180, h_med + h_pad), s_hi, v_hi], dtype=np.uint8)
    return BallColorCalibration(lower1=lower1, upper1=upper1)


def _ball_color_hsv_mask(hsv: Any, calibration: Optional[BallColorCalibration]) -> Any:
    if calibration is not None:
        mask = cv2.inRange(hsv, calibration.lower1, calibration.upper1)
        if calibration.lower2 is not None and calibration.upper2 is not None:
            mask = mask | cv2.inRange(hsv, calibration.lower2, calibration.upper2)
        return mask
    return cv2.inRange(hsv, ORANGE_HSV_LOWER1, ORANGE_HSV_UPPER1) | cv2.inRange(
        hsv, ORANGE_HSV_LOWER2, ORANGE_HSV_UPPER2
    )


def detect_ball_color_blob(
    frame_bgr: Any,
    hint_xy: Tuple[float, float],
    roi_half: int,
    persons: List[Tuple[int, Tuple[float, float, float, float]]],
    color_calibration: Optional[BallColorCalibration] = None,
) -> Optional[Tuple[float, float, float]]:
    """Ищет круглый blob мяча в ROI (HSV из калибровки по кликам или дефолтный оранжевый)."""
    if cv2 is None or frame_bgr is None:
        return None
    h, w = frame_bgr.shape[:2]
    cx, cy = int(hint_xy[0]), int(hint_xy[1])
    x1, y1 = max(cx - roi_half, 0), max(cy - roi_half, 0)
    x2, y2 = min(cx + roi_half, w), min(cy + roi_half, h)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None

    roi = frame_bgr[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = _ball_color_hsv_mask(hsv, color_calibration)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    person_diags = [math.hypot(b[2] - b[0], b[3] - b[1]) for _, b in persons]
    ref_size = float(np.mean(person_diags)) if person_diags else min(h, w) * 0.12
    min_area = max(16.0, (ref_size * 0.04) ** 2)
    max_area = max(min_area * 4.0, (ref_size * 0.22) ** 2)

    best: Optional[Tuple[float, float, float]] = None
    best_score = -1.0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter < 1e-3:
            continue
        circularity = 4.0 * math.pi * area / (perimeter * perimeter)
        if circularity < 0.45:
            continue
        moments = cv2.moments(cnt)
        if abs(moments["m00"]) < 1e-3:
            continue
        bx = moments["m10"] / moments["m00"] + x1
        by = moments["m01"] / moments["m00"] + y1
        dist = math.hypot(bx - hint_xy[0], by - hint_xy[1])
        if dist > roi_half * 1.25:
            continue
        score = circularity * math.sqrt(area) / (1.0 + dist * 0.05)
        if score > best_score:
            best_score = score
            best = (float(bx), float(by), float(min(1.0, circularity)))
    return best


def _create_cv_ball_tracker() -> Any:
    """CSRT с fallback на KCF (opencv-contrib)."""
    if cv2 is None:
        return None
    factories = []
    if hasattr(cv2, "TrackerCSRT_create"):
        factories.append(cv2.TrackerCSRT_create)
    legacy = getattr(cv2, "legacy", None)
    if legacy is not None and hasattr(legacy, "TrackerCSRT_create"):
        factories.append(legacy.TrackerCSRT_create)
    if hasattr(cv2, "TrackerKCF_create"):
        factories.append(cv2.TrackerKCF_create)
    if legacy is not None and hasattr(legacy, "TrackerKCF_create"):
        factories.append(legacy.TrackerKCF_create)
    for factory in factories:
        try:
            tracker = factory()
            if tracker is not None:
                return tracker
        except Exception:
            continue
    return None


def _bbox_from_center(x: float, y: float, half: float = BALL_DEFAULT_BBOX_HALF) -> Tuple[float, float, float, float]:
    return (x - half, y - half, x + half, y + half)


def _clip_bbox(
    bbox: Tuple[float, float, float, float], frame_w: int, frame_h: int
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    xi1 = max(int(x1), 0)
    yi1 = max(int(y1), 0)
    xi2 = min(int(x2), frame_w)
    yi2 = min(int(y2), frame_h)
    if xi2 <= xi1 + 2:
        xi2 = min(xi1 + 4, frame_w)
    if yi2 <= yi1 + 2:
        yi2 = min(yi1 + 4, frame_h)
    return xi1, yi1, xi2, yi2


def _ball_center_from_bbox(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def detect_ball_yolo_on_crop(
    crop: Any,
    offset_xy: Tuple[int, int],
    model,
    device: str,
    ball_conf_threshold: float,
    imgsz: int,
) -> Tuple[Optional[Tuple[float, float]], float, Optional[Tuple[float, float, float, float]]]:
    """YOLO predict на кропе; координаты возвращаются в системе полного кадра.

    Detection-only: временно отключает ByteTrack на predictor, чтобы ROI/тайлы
    не подмешивали координаты кропа во внутреннее состояние трекера.
    """
    if model is None or crop is None or crop.size == 0:
        return None, 0.0, None
    saved_trackers: Any = None
    predictor = getattr(model, "predictor", None)
    if predictor is not None and hasattr(predictor, "trackers"):
        saved_trackers = predictor.trackers
        predictor.trackers = None
    try:
        results = model.predict(
            crop,
            classes=[COCO_BALL_CLASS_ID],
            conf=ball_conf_threshold,
            imgsz=imgsz,
            device=device,
            verbose=False,
        )
    except Exception as exc:
        logger.warning("Ball ROI/tile YOLO predict failed: %s", exc)
        return None, 0.0, None
    finally:
        if predictor is not None and saved_trackers is not None:
            predictor.trackers = saved_trackers
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return None, 0.0, None
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    best_i = int(np.argmax(confs))
    box = xyxy[best_i]
    ox, oy = offset_xy
    bbox = (float(box[0]) + ox, float(box[1]) + oy, float(box[2]) + ox, float(box[3]) + oy)
    center = _ball_center_from_bbox(bbox)
    return center, float(confs[best_i]), bbox


def detect_ball_yolo_tiled_and_roi(
    frame_bgr: Any,
    model,
    device: str,
    ball_conf_threshold: float,
    hint_xy: Optional[Tuple[float, float]],
    roi_half: int,
) -> Tuple[Optional[Tuple[float, float]], float, Optional[Tuple[float, float, float, float]], str]:
    """ROI YOLO вокруг подсказки, затем 2×2 тайлы по всему кадру."""
    if model is None or frame_bgr is None:
        return None, 0.0, None, "roi"
    h, w = frame_bgr.shape[:2]
    best: Optional[Tuple[float, float]] = None
    best_conf = -1.0
    best_bbox: Optional[Tuple[float, float, float, float]] = None
    best_source = "roi"

    if hint_xy is not None:
        cx, cy = int(hint_xy[0]), int(hint_xy[1])
        x1, y1 = max(cx - roi_half, 0), max(cy - roi_half, 0)
        x2, y2 = min(cx + roi_half, w), min(cy + roi_half, h)
        if x2 - x1 >= 16 and y2 - y1 >= 16:
            center, conf, bbox = detect_ball_yolo_on_crop(
                frame_bgr[y1:y2, x1:x2], (x1, y1), model, device, ball_conf_threshold, BALL_ROI_DETECT_IMGSZ
            )
            if center is not None and conf > best_conf:
                best, best_conf, best_bbox, best_source = center, conf, bbox, "roi"

    mid_x, mid_y = w // 2, h // 2
    tiles = [
        (0, 0, mid_x, mid_y),
        (mid_x, 0, w, mid_y),
        (0, mid_y, mid_x, h),
        (mid_x, mid_y, w, h),
    ]
    for tx1, ty1, tx2, ty2 in tiles:
        if tx2 - tx1 < 16 or ty2 - ty1 < 16:
            continue
        center, conf, bbox = detect_ball_yolo_on_crop(
            frame_bgr[ty1:ty2, tx1:tx2],
            (tx1, ty1),
            model,
            device,
            ball_conf_threshold,
            BALL_TILED_IMGSZ,
        )
        if center is not None and conf > best_conf:
            best, best_conf, best_bbox, best_source = center, conf, bbox, "tiled"

    return best, best_conf if best is not None else 0.0, best_bbox, best_source


class BallTracker:
    """Сглаживает пропуски YOLO: CSRT, ROI/tiled YOLO, Kalman, цвет."""

    def __init__(
        self,
        max_gap_frames: int = BALL_MAX_GAP_FRAMES_DEFAULT,
        max_predict_frames: int = BALL_MAX_PREDICT_FRAMES_DEFAULT,
        color_fallback: bool = BALL_COLOR_FALLBACK_DEFAULT,
        color_roi_half: int = BALL_COLOR_ROI_HALF_DEFAULT,
        model=None,
        device: str = "cpu",
        ball_conf: float = BALL_CONF_DEFAULT,
        imgsz: int = IMGSZ_DEFAULT,
    ) -> None:
        self.max_gap_frames = max_gap_frames
        self.max_predict_frames = max_predict_frames
        self.color_fallback = color_fallback
        self.color_roi_half = color_roi_half
        self.model = model
        self.device = device
        self.ball_conf = ball_conf
        self.imgsz = imgsz
        self.csrt_max_jump_px = float(BALL_CSRT_MAX_JUMP_PX)
        self.color_calibration: Optional[BallColorCalibration] = None
        self.kalman = BallKalmanFilter()
        self.csrt_tracker: Any = None
        self.last_bbox: Optional[Tuple[float, float, float, float]] = None
        self.last_confident_frame: Optional[int] = None
        self.last_confident_pos: Optional[Tuple[float, float]] = None
        self.prev_confident_frame: Optional[int] = None
        self.prev_confident_pos: Optional[Tuple[float, float]] = None
        self.last_committed_conf: float = 0.0
        self.frames_since_yolo = 10**6
        self.frames_since_any = 10**6

    def _reset_csrt(self) -> None:
        self.csrt_tracker = None

    def _init_csrt(self, frame_bgr: Any, bbox: Tuple[float, float, float, float]) -> bool:
        if cv2 is None:
            return False
        tracker = _create_cv_ball_tracker()
        if tracker is None:
            return False
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = _clip_bbox(bbox, w, h)
        try:
            tracker.init(frame_bgr, (x1, y1, x2 - x1, y2 - y1))
        except Exception:
            return False
        self.csrt_tracker = tracker
        self.last_bbox = (float(x1), float(y1), float(x2), float(y2))
        return True

    def _update_csrt(self, frame_bgr: Any) -> Optional[Tuple[float, float, Tuple[float, float, float, float]]]:
        if self.csrt_tracker is None:
            return None
        try:
            ok, rect = self.csrt_tracker.update(frame_bgr)
        except Exception:
            self._reset_csrt()
            return None
        if not ok:
            self._reset_csrt()
            return None
        x, y, rw, rh = rect
        bbox = (float(x), float(y), float(x + rw), float(y + rh))
        center = _ball_center_from_bbox(bbox)
        if self.last_confident_pos is not None:
            jump = math.hypot(center[0] - self.last_confident_pos[0], center[1] - self.last_confident_pos[1])
            if jump > self.csrt_max_jump_px:
                self._reset_csrt()
                return None
        self.last_bbox = bbox
        return center[0], center[1], bbox

    def _record_confident(self, frame_idx: int, x: float, y: float) -> None:
        if self.last_confident_frame is not None:
            self.prev_confident_frame = self.last_confident_frame
            self.prev_confident_pos = self.last_confident_pos
        self.last_confident_frame = frame_idx
        self.last_confident_pos = (x, y)

    def _linear_extrapolate(self, frame_idx: int) -> Optional[Tuple[float, float]]:
        if (
            self.prev_confident_frame is None
            or self.prev_confident_pos is None
            or self.last_confident_frame is None
            or self.last_confident_pos is None
        ):
            return None
        f0, p0 = self.prev_confident_frame, self.prev_confident_pos
        f1, p1 = self.last_confident_frame, self.last_confident_pos
        if f1 <= f0 or frame_idx <= f1:
            return None
        gap = frame_idx - f1
        if gap > self.max_gap_frames:
            return None
        dt = float(f1 - f0)
        vx = (p1[0] - p0[0]) / dt
        vy = (p1[1] - p0[1]) / dt
        return p1[0] + vx * gap, p1[1] + vy * gap

    def _should_reinit_csrt(
        self,
        new_bbox: Tuple[float, float, float, float],
        conf: float,
        source: str,
        prev_bbox: Optional[Tuple[float, float, float, float]] = None,
    ) -> bool:
        compare_bbox = prev_bbox if prev_bbox is not None else self.last_bbox
        if self.csrt_tracker is None or compare_bbox is None:
            return True
        if bbox_iou(new_bbox, compare_bbox) >= CSRT_REINIT_IOU_THRESHOLD:
            return True
        if source in ("yolo", "tiled", "roi") and conf > self.last_committed_conf + CSRT_REINIT_CONF_MARGIN:
            return True
        return False

    def _commit_detection(
        self,
        frame_idx: int,
        frame_bgr: Any,
        x: float,
        y: float,
        conf: float,
        source: str,
        bbox: Optional[Tuple[float, float, float, float]] = None,
    ) -> BallTrackState:
        bbox = bbox or _bbox_from_center(x, y)
        prev_bbox = self.last_bbox
        reinit_csrt = self._should_reinit_csrt(bbox, conf, source, prev_bbox=prev_bbox)
        self.last_bbox = bbox
        self._record_confident(frame_idx, x, y)
        if self.kalman.initialized:
            self.kalman.predict()
        self.kalman.correct(x, y)
        if reinit_csrt:
            self._init_csrt(frame_bgr, bbox)
        self.last_committed_conf = float(conf)
        if source in ("yolo", "user", "user_interp", "tiled", "roi"):
            self.frames_since_yolo = 0
        self.frames_since_any = 0
        return BallTrackState(x, y, conf, source)

    def _roi_detection_commits_confident(self, x: float, y: float, conf: float) -> bool:
        if conf < self.ball_conf:
            return False
        if self.last_confident_pos is None:
            return True
        jump = math.hypot(x - self.last_confident_pos[0], y - self.last_confident_pos[1])
        if jump > self.csrt_max_jump_px and conf <= self.last_committed_conf + CSRT_REINIT_CONF_MARGIN:
            return False
        return True

    def update(
        self,
        frame_idx: int,
        frame_bgr: Any,
        yolo_ball: Optional[Tuple[float, float]],
        yolo_conf: float,
        persons: List[Tuple[int, Tuple[float, float, float, float]]],
        yolo_bbox: Optional[Tuple[float, float, float, float]] = None,
        ball_anchor_guide: Optional[BallAnchorGuide] = None,
    ) -> Optional[BallTrackState]:
        if yolo_ball is not None:
            bbox = yolo_bbox or _bbox_from_center(yolo_ball[0], yolo_ball[1])
            return self._commit_detection(
                frame_idx, frame_bgr, yolo_ball[0], yolo_ball[1], yolo_conf, "yolo", bbox=bbox
            )

        if ball_anchor_guide is not None:
            override = ball_anchor_guide.get_override(frame_idx)
            if override is not None:
                return self._commit_detection(
                    frame_idx, frame_bgr, override.x, override.y, override.conf, override.source
                )
            if self.csrt_tracker is None:
                seed_bbox = ball_anchor_guide.get_csrt_seed_bbox(frame_idx)
                if seed_bbox is not None:
                    self._init_csrt(frame_bgr, seed_bbox)

        self.frames_since_yolo += 1
        if self.frames_since_any >= self.max_predict_frames:
            return None

        predicted = self.kalman.predict() if self.kalman.initialized else None
        hint = predicted or self.last_confident_pos

        csrt_hit = self._update_csrt(frame_bgr)
        if csrt_hit is not None:
            x, y, bbox = csrt_hit
            self.last_bbox = bbox
            self._record_confident(frame_idx, x, y)
            self.kalman.correct(x, y)
            self.frames_since_any = 0
            return BallTrackState(x, y, 0.55, "csrt")

        if self.model is not None and hint is not None and self.frames_since_yolo >= 2:
            det_center, det_conf, det_bbox, det_src = detect_ball_yolo_tiled_and_roi(
                frame_bgr,
                self.model,
                self.device,
                self.ball_conf,
                hint,
                self.color_roi_half,
            )
            if det_center is not None:
                x, y = det_center
                if self._roi_detection_commits_confident(x, y, det_conf):
                    return self._commit_detection(
                        frame_idx,
                        frame_bgr,
                        x,
                        y,
                        det_conf,
                        det_src,
                        bbox=det_bbox,
                    )
                self.frames_since_any = 0
                return BallTrackState(x, y, det_conf, det_src)

        if self.color_fallback and hint is not None:
            color_hit = detect_ball_color_blob(
                frame_bgr, hint, self.color_roi_half, persons, self.color_calibration
            )
            if color_hit is not None:
                x, y, score = color_hit
                bbox = _bbox_from_center(x, y)
                self.last_bbox = bbox
                if self.kalman.initialized:
                    self.kalman.predict()
                self.kalman.correct(x, y)
                if self._should_reinit_csrt(bbox, score * 0.5, "color"):
                    self._init_csrt(frame_bgr, bbox)
                self.frames_since_any = 0
                return BallTrackState(x, y, score * 0.5, "color")

        if self.frames_since_yolo > self.max_gap_frames:
            self.frames_since_any += 1
            return None

        linear = self._linear_extrapolate(frame_idx)
        if linear is not None:
            self.frames_since_any += 1
            return BallTrackState(linear[0], linear[1], 0.0, "interp")

        if predicted is not None:
            self.frames_since_any += 1
            return BallTrackState(predicted[0], predicted[1], 0.0, "kalman")

        if self.last_confident_pos is not None and self.frames_since_yolo <= self.max_gap_frames:
            self.frames_since_any += 1
            return BallTrackState(
                self.last_confident_pos[0], self.last_confident_pos[1], 0.0, "interp"
            )

        self.frames_since_any += 1
        return None


def extract_jersey_histogram(
    frame_bgr: Any, box: Tuple[float, float, float, float]
) -> Optional[np.ndarray]:
    """HSV-гистограмма верхней части bbox (майка) для appearance-matching."""
    if cv2 is None or frame_bgr is None:
        return None
    x1, y1, x2, y2 = box
    h = max(y2 - y1, 1.0)
    torso_y2 = y1 + h * 0.55
    xi1, yi1 = max(int(x1), 0), max(int(y1), 0)
    xi2, yi2 = min(int(x2), frame_bgr.shape[1]), min(int(torso_y2), frame_bgr.shape[0])
    if xi2 <= xi1 or yi2 <= yi1:
        return None
    crop = frame_bgr[yi1:yi2, xi1:xi2]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten().astype(np.float32)


def histogram_similarity(h1: np.ndarray, h2: np.ndarray) -> float:
    return float(cv2.compareHist(h1.reshape(-1, 1), h2.reshape(-1, 1), cv2.HISTCMP_CORREL))


_REID_MODEL: Any = None
_REID_DEVICE: Optional[str] = None
_REID_LAST_ERROR: Optional[str] = None
_OCR_READER: Any = None


def _resolve_reid_device(device: Optional[str] = None) -> str:
    if device:
        return device
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_reid_error_message() -> Optional[str]:
    """Последняя ошибка ReID для отображения в UI (шаг 3/4 или sidebar)."""
    return _REID_LAST_ERROR


def render_reid_status_warning() -> None:
    """Показывает предупреждение, если MobileNet ReID недоступен."""
    reid_error = get_reid_error_message()
    if reid_error:
        st.warning(
            f"⚠️ ReID (MobileNet) недоступен — склейка ID работает только по HSV/OCR. "
            f"Причина: {reid_error}"
        )


def _get_reid_model(device: Optional[str] = None) -> Any:
    global _REID_MODEL, _REID_DEVICE, _REID_LAST_ERROR
    if torch is None:
        return None
    target_device = _resolve_reid_device(device)
    if _REID_MODEL is not None and _REID_DEVICE == target_device:
        return _REID_MODEL
    try:
        from torchvision import models
        import torch.nn as nn

        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        model.classifier = nn.Identity()
        model.eval()
        model.to(target_device)
        _REID_MODEL = model
        _REID_DEVICE = target_device
        _REID_LAST_ERROR = None
    except Exception as exc:
        _REID_LAST_ERROR = str(exc)
        logger.warning("Failed to load MobileNet ReID model: %s", exc)
        _REID_MODEL = None
        _REID_DEVICE = None
    return _REID_MODEL


def _get_ocr_reader() -> Any:
    global _OCR_READER
    if _OCR_READER is not None:
        return _OCR_READER
    if easyocr is None:
        return None
    try:
        _OCR_READER = easyocr.Reader(["en"], gpu=torch is not None and torch.cuda.is_available(), verbose=False)
    except Exception:
        _OCR_READER = None
    return _OCR_READER


def extract_reid_embedding(
    frame_bgr: Any,
    box: Tuple[float, float, float, float],
    device: Optional[str] = None,
) -> Optional[np.ndarray]:
    global _REID_LAST_ERROR
    target_device = _resolve_reid_device(device)
    model = _get_reid_model(target_device)
    if model is None or torch is None or cv2 is None or frame_bgr is None:
        return None
    x1, y1, x2, y2 = [int(v) for v in box]
    h, w = frame_bgr.shape[:2]
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, w), min(y2, h)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    try:
        from torchvision import transforms

        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((128, 64)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        tensor = transform(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(target_device)
        with torch.no_grad():
            emb = model(tensor).cpu().numpy().flatten().astype(np.float32)
        norm = np.linalg.norm(emb)
        if norm > 1e-6:
            emb /= norm
        return emb
    except Exception as exc:
        _REID_LAST_ERROR = str(exc)
        logger.warning("ReID embedding failed: %s", exc)
        return None


def embedding_similarity(e1: np.ndarray, e2: np.ndarray) -> float:
    return float(np.dot(e1, e2))


def read_jersey_number_from_box(
    frame_bgr: Any, box: Tuple[float, float, float, float]
) -> Optional[str]:
    reader = _get_ocr_reader()
    if reader is None or cv2 is None or frame_bgr is None:
        return None
    x1, y1, x2, y2 = box
    torso_y2 = y1 + max(y2 - y1, 1.0) * 0.65
    xi1, yi1 = max(int(x1), 0), max(int(y1), 0)
    xi2, yi2 = min(int(x2), frame_bgr.shape[1]), min(int(torso_y2), frame_bgr.shape[0])
    if xi2 - xi1 < 12 or yi2 - yi1 < 12:
        return None
    crop = frame_bgr[yi1:yi2, xi1:xi2]
    try:
        results = reader.readtext(crop, allowlist="0123456789", detail=1, paragraph=False)
    except Exception:
        return None
    best_num: Optional[str] = None
    best_conf = -1.0
    for _bbox, text, conf in results:
        digits = "".join(ch for ch in str(text) if ch.isdigit())
        if not digits or len(digits) > 2:
            continue
        conf = float(conf)
        if conf >= JERSEY_OCR_MIN_CONF and conf > best_conf:
            best_conf = conf
            best_num = digits
    return best_num


@dataclass
class _DetectionFeatures:
    raw_id: int
    box: Tuple[float, float, float, float]
    center: Tuple[float, float]
    hist: Optional[np.ndarray]
    emb: Optional[np.ndarray]
    jersey: Optional[str]
    area: float


@dataclass
class _TrackProfile:
    canonical_id: int
    histogram: Optional[np.ndarray] = None
    embedding: Optional[np.ndarray] = None
    jersey_number: Optional[str] = None
    last_center: Tuple[float, float] = (0.0, 0.0)
    last_box: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    velocity: Tuple[float, float] = (0.0, 0.0)
    last_seen_frame: int = -1


class AppearanceMerger:
    """Стабилизация ID поверх ByteTrack: motion + ReID/HSV + OCR, анти-swap при пересечении."""

    def __init__(
        self,
        similarity_threshold: float = APPEARANCE_SIMILARITY_DEFAULT,
        lost_buffer_frames: int = APPEARANCE_LOST_BUFFER_FRAMES,
        jersey_ocr_enabled: bool = JERSEY_OCR_ENABLED_DEFAULT,
        min_person_bbox_area: float = MIN_PERSON_BBOX_AREA_DEFAULT,
        motion_max_px: float = APPEARANCE_MOTION_MAX_PX,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.lost_buffer_frames = lost_buffer_frames
        self.jersey_ocr_enabled = jersey_ocr_enabled
        self.min_person_bbox_area = float(min_person_bbox_area)
        self.motion_max_px = float(motion_max_px)
        self.profiles: Dict[int, _TrackProfile] = {}
        self.prev_active_ids: List[int] = []
        self._next_synthetic_id = 50_000
        self.merge_log: List[Dict[str, Any]] = []

    def _allocate_new_id(self, raw_id: int) -> int:
        if raw_id not in self.profiles:
            return int(raw_id)
        while self._next_synthetic_id in self.profiles:
            self._next_synthetic_id += 1
        cid = self._next_synthetic_id
        self._next_synthetic_id += 1
        return cid

    def _jersey_compatible(self, det_jersey: Optional[str], stored: Optional[str]) -> bool:
        if not det_jersey or not stored:
            return True
        return det_jersey == stored

    def _appearance_similarity(
        self,
        profile: _TrackProfile,
        hist: Optional[np.ndarray],
        emb: Optional[np.ndarray],
    ) -> float:
        scores: List[float] = []
        if hist is not None and profile.histogram is not None:
            scores.append(histogram_similarity(hist, profile.histogram))
        if emb is not None and profile.embedding is not None:
            scores.append(embedding_similarity(emb, profile.embedding))
        if not scores:
            return -1.0
        return float(np.mean(scores))

    def _match_score(self, det: _DetectionFeatures, profile: _TrackProfile, frame_idx: int) -> float:
        if not self._jersey_compatible(det.jersey, profile.jersey_number):
            return -1.0
        app = self._appearance_similarity(profile, det.hist, det.emb)
        if app < 0:
            app = 0.0
        if profile.last_seen_frame >= 0 and frame_idx - profile.last_seen_frame <= 2 and app < APPEARANCE_SWAP_GATE:
            return -1.0
        dt = max(frame_idx - profile.last_seen_frame, 1)
        pred_x = profile.last_center[0] + profile.velocity[0] * dt
        pred_y = profile.last_center[1] + profile.velocity[1] * dt
        dist = math.hypot(det.center[0] - pred_x, det.center[1] - pred_y)
        diag = math.hypot(profile.last_box[2] - profile.last_box[0], profile.last_box[3] - profile.last_box[1])
        max_dist = max(self.motion_max_px, diag * 1.6, 40.0)
        motion = max(0.0, 1.0 - dist / max_dist)
        return 0.65 * app + 0.35 * motion

    def _greedy_assign(
        self,
        detections: List[_DetectionFeatures],
        candidate_ids: List[int],
        frame_idx: int,
        min_score: float,
    ) -> Dict[int, int]:
        pairs: List[Tuple[float, int, int]] = []
        for di, det in enumerate(detections):
            for cid in candidate_ids:
                profile = self.profiles.get(cid)
                if profile is None:
                    continue
                score = self._match_score(det, profile, frame_idx)
                if score >= min_score:
                    pairs.append((score, di, cid))
        pairs.sort(key=lambda item: item[0], reverse=True)
        det_to_cid: Dict[int, int] = {}
        used_dets: set = set()
        used_cids: set = set()
        for score, di, cid in pairs:
            if di in used_dets or cid in used_cids:
                continue
            det_to_cid[di] = cid
            used_dets.add(di)
            used_cids.add(cid)
        return det_to_cid

    def _correct_pair_swap(
        self,
        detections: List[_DetectionFeatures],
        assignments: Dict[int, int],
        frame_idx: int,
    ) -> Dict[int, int]:
        if len(detections) != 2 or len(assignments) != 2:
            return assignments
        di0, di1 = 0, 1
        if di0 not in assignments or di1 not in assignments:
            return assignments
        c0, c1 = assignments[di0], assignments[di1]
        p0, p1 = self.profiles[c0], self.profiles[c1]
        direct = self._match_score(detections[di0], p0, frame_idx) + self._match_score(detections[di1], p1, frame_idx)
        crossed = self._match_score(detections[di0], p1, frame_idx) + self._match_score(detections[di1], p0, frame_idx)
        if crossed > direct + 0.04 and min(
            self._match_score(detections[di0], p1, frame_idx),
            self._match_score(detections[di1], p0, frame_idx),
        ) >= APPEARANCE_SWAP_GATE:
            assignments = {di0: c1, di1: c0}
            self.merge_log.append(
                {
                    "Кадр": frame_idx,
                    "Событие": "swap_corrected",
                    "ID A": int(c0),
                    "ID B": int(c1),
                    "Причина": "внешность+движение",
                }
            )
        return assignments

    def _update_profile(
        self,
        canonical_id: int,
        det: _DetectionFeatures,
        frame_idx: int,
    ) -> None:
        profile = self.profiles.get(canonical_id)
        if profile is None:
            profile = _TrackProfile(canonical_id=canonical_id)
            self.profiles[canonical_id] = profile
        if det.hist is not None:
            if profile.histogram is None:
                profile.histogram = det.hist.copy()
            else:
                alpha = APPEARANCE_HIST_EMA_ALPHA
                profile.histogram = (1.0 - alpha) * profile.histogram + alpha * det.hist
        if det.emb is not None:
            if profile.embedding is None:
                profile.embedding = det.emb.copy()
            else:
                alpha = APPEARANCE_HIST_EMA_ALPHA
                merged = (1.0 - alpha) * profile.embedding + alpha * det.emb
                norm = np.linalg.norm(merged)
                if norm > 1e-6:
                    merged /= norm
                profile.embedding = merged.astype(np.float32)
        if det.jersey:
            profile.jersey_number = det.jersey
        dt = max(frame_idx - profile.last_seen_frame, 1)
        if profile.last_seen_frame >= 0 and dt <= 30:
            profile.velocity = (
                (det.center[0] - profile.last_center[0]) / dt,
                (det.center[1] - profile.last_center[1]) / dt,
            )
        profile.last_center = det.center
        profile.last_box = det.box
        profile.last_seen_frame = frame_idx

    def remap(
        self,
        frame_idx: int,
        frame_bgr: Any,
        persons: List[Tuple[int, Tuple[float, float, float, float]]],
        device: Optional[str] = None,
    ) -> List[Tuple[int, Tuple[float, float, float, float]]]:
        detections: List[_DetectionFeatures] = []
        for raw_id, box in persons:
            area = bbox_area(box)
            if area < self.min_person_bbox_area:
                continue
            hist = extract_jersey_histogram(frame_bgr, box)
            emb = extract_reid_embedding(frame_bgr, box, device=device)
            jersey = read_jersey_number_from_box(frame_bgr, box) if self.jersey_ocr_enabled else None
            detections.append(
                _DetectionFeatures(
                    raw_id=int(raw_id),
                    box=box,
                    center=bbox_center(box),
                    hist=hist,
                    emb=emb,
                    jersey=jersey,
                    area=area,
                )
            )

        prev_active = list(self.prev_active_ids)
        match_min = max(self.similarity_threshold, APPEARANCE_MATCH_MIN)
        assignments = self._greedy_assign(detections, prev_active, frame_idx, match_min)
        assignments = self._correct_pair_swap(detections, assignments, frame_idx)

        assigned_cids = set(assignments.values())
        for di, det in enumerate(detections):
            if di in assignments:
                continue
            lost_candidates = [
                cid
                for cid, profile in self.profiles.items()
                if cid not in assigned_cids
                and cid not in prev_active
                and frame_idx - profile.last_seen_frame <= self.lost_buffer_frames
            ]
            lost_assign = self._greedy_assign([det], lost_candidates, frame_idx, match_min)
            if lost_assign:
                assignments[di] = lost_assign[0]
                assigned_cids.add(lost_assign[0])
                self.merge_log.append(
                    {
                        "Кадр": frame_idx,
                        "Событие": "reid_after_lost",
                        "Новый ID трекера": int(det.raw_id),
                        "Склеен с ID": int(lost_assign[0]),
                    }
                )

        remapped: List[Tuple[int, Tuple[float, float, float, float]]] = []
        active_this_frame: List[int] = []
        for di, det in enumerate(detections):
            canonical_id = assignments.get(di)
            if canonical_id is None:
                canonical_id = self._allocate_new_id(det.raw_id)
                self.merge_log.append(
                    {
                        "Кадр": frame_idx,
                        "Событие": "new_id",
                        "ID трекера ByteTrack": int(det.raw_id),
                        "Назначен ID": int(canonical_id),
                    }
                )
            self._update_profile(canonical_id, det, frame_idx)
            active_this_frame.append(canonical_id)
            remapped.append((canonical_id, det.box))

        self.prev_active_ids = active_this_frame
        return remapped


def detect_yolo_ball_on_frame(
    frame: Any,
    model,
    device: str,
    ball_conf_threshold: float,
    imgsz: int,
) -> Tuple[Optional[Tuple[float, float]], float]:
    """Однокадровая детекция мяча YOLO (для диагностики без трекера)."""
    if model is None:
        return None, 0.0
    try:
        results = model.predict(
            frame,
            classes=[COCO_BALL_CLASS_ID],
            conf=ball_conf_threshold,
            imgsz=imgsz,
            device=device,
            verbose=False,
        )
    except Exception:
        return None, 0.0
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return None, 0.0
    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    best_i = int(np.argmax(confs))
    box = xyxy[best_i]
    conf = float(confs[best_i])
    center = ((float(box[0]) + float(box[2])) / 2.0, (float(box[1]) + float(box[3])) / 2.0)
    return center, conf


def build_sparse_frame_indices(
    total_frames: int,
    max_samples: int = BALL_DIAGNOSTIC_MAX_SAMPLES,
    min_samples: int = BALL_DIAGNOSTIC_MIN_SAMPLES,
) -> List[int]:
    """Равномерные индексы кадров по всему видео с жёстким потолком инференсов."""
    if total_frames <= 0:
        return []
    cap = min(max(max_samples, min_samples), 120)
    if total_frames <= cap:
        return list(range(total_frames))
    step = max(total_frames // cap, 1)
    indices = list(range(0, total_frames, step))
    if len(indices) > cap:
        indices = indices[:cap]
    if indices[-1] != total_frames - 1 and len(indices) < cap:
        indices.append(total_frames - 1)
    return indices


# ---------------------------------------------------------------------------
# Диагностика видимости мяча (шаг 3) — отличить "проблема в порогах" от
# "модель в принципе не видит мяч на этом видео".
# ---------------------------------------------------------------------------
def diagnose_ball_visibility(
    video_path: str,
    model,
    device: str,
    ball_conf_threshold: float,
    imgsz: int,
    enhance_quality: bool = False,
    max_gap_frames: int = BALL_MAX_GAP_FRAMES_DEFAULT,
    max_predict_frames: int = BALL_MAX_PREDICT_FRAMES_DEFAULT,
    color_fallback: bool = BALL_COLOR_FALLBACK_DEFAULT,
    color_roi_half: int = BALL_COLOR_ROI_HALF_DEFAULT,
    progress_callback: Optional[Callable[[float], Any]] = None,
    status_callback: Optional[Callable[[str], Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Сэмплирует до ~100 кадров равномерно по всему видео (model.predict на
    каждом) и сравнивает видимость мяча: YOLO vs YOLO + Kalman/цвет."""
    if model is None or cv2 is None:
        return None
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total_frames <= 0:
        cap.release()
        return None

    sample_indices = build_sparse_frame_indices(total_frames)
    if not sample_indices:
        cap.release()
        return None

    tracker = BallTracker(
        max_gap_frames=max_gap_frames,
        max_predict_frames=max_predict_frames,
        color_fallback=color_fallback,
        color_roi_half=color_roi_half,
        model=model,
        device=device,
        ball_conf=ball_conf_threshold,
        imgsz=imgsz,
    )

    yolo_hits = 0
    enhanced_hits = 0
    yolo_conf_sum = 0.0
    source_counts: Dict[str, int] = {
        "yolo": 0, "csrt": 0, "tiled": 0, "roi": 0, "color": 0, "interp": 0, "kalman": 0,
    }
    gaps: List[int] = []
    gap_start_frame: Optional[int] = None
    n_samples = len(sample_indices)

    for sample_i, frame_idx in enumerate(sample_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        if status_callback is not None:
            try:
                status_callback(
                    f"Сэмпл {sample_i + 1}/{n_samples} · кадр {frame_idx} / {total_frames}"
                )
            except Exception:
                pass
        if progress_callback is not None:
            try:
                progress_callback(min((sample_i + 1) / n_samples, 1.0))
            except Exception:
                pass

        detect_frame = enhance_frame_for_detection(frame) if enhance_quality else frame
        yolo_ball, yolo_conf = detect_yolo_ball_on_frame(
            detect_frame, model, device, ball_conf_threshold, imgsz
        )
        if enhance_quality and yolo_ball is not None:
            inv = 1.0 / ENHANCE_UPSCALE_FACTOR
            yolo_ball = (yolo_ball[0] * inv, yolo_ball[1] * inv)

        if yolo_ball is not None:
            yolo_hits += 1
            yolo_conf_sum += yolo_conf
            if gap_start_frame is not None:
                gaps.append(frame_idx - gap_start_frame)
                gap_start_frame = None
        elif gap_start_frame is None:
            gap_start_frame = frame_idx

        state = tracker.update(frame_idx, frame, yolo_ball, yolo_conf, [])
        if state is not None:
            enhanced_hits += 1
            source_counts[state.source] = source_counts.get(state.source, 0) + 1

    cap.release()
    if n_samples <= 0:
        return None

    yolo_rate = yolo_hits / n_samples
    enhanced_rate = enhanced_hits / n_samples
    return {
        "sampled_frames": n_samples,
        "total_video_frames": total_frames,
        "sample_step_approx": max(total_frames // max(n_samples, 1), 1),
        "frames_with_ball_yolo": yolo_hits,
        "yolo_detection_rate": yolo_rate,
        "frames_with_ball_enhanced": enhanced_hits,
        "enhanced_detection_rate": enhanced_rate,
        "avg_confidence": (yolo_conf_sum / yolo_hits) if yolo_hits else 0.0,
        "avg_gap_frames": float(np.mean(gaps)) if gaps else 0.0,
        "max_gap_frames": int(max(gaps)) if gaps else 0,
        "source_counts": source_counts,
        "frames_with_ball": yolo_hits,
        "detection_rate": yolo_rate,
    }


# ---------------------------------------------------------------------------
# Быстрое предварительное сканирование для сбора списка игроков (шаг 3)
# ---------------------------------------------------------------------------
def compute_quick_scan_frame_count(fps: float, max_seconds: float) -> int:
    """Непрерывный префикс кадров для шага 3 — те же кадры, что видит process_video с 0."""
    requested = max(int(float(fps) * float(max_seconds)), 1)
    return min(requested, QUICK_SCAN_MAX_TRACK_FRAMES)


def quick_player_scan(
    video_path: str,
    model,
    device: str,
    max_seconds: float,
    fps_hint: float,
    enhance_quality: bool = False,
    person_conf: float = PERSON_CONF_DEFAULT,
    ball_conf: float = BALL_CONF_DEFAULT,
    imgsz: int = IMGSZ_DEFAULT,
    appearance_similarity: float = APPEARANCE_SIMILARITY_DEFAULT,
    jersey_ocr_enabled: bool = JERSEY_OCR_ENABLED_DEFAULT,
    min_person_bbox_area: float = MIN_PERSON_BBOX_AREA_DEFAULT,
    progress_callback: Optional[Callable[[float], Any]] = None,
    status_callback: Optional[Callable[[str], Any]] = None,
) -> Tuple[Dict[int, Any], List[Dict[str, Any]]]:
    """Короткий прогон трекера по первым max_seconds секундам видео.

    Цель — не полноценная аналитика, а быстрый сбор всех уникальных ID
    игроков и одного репрезентативного кропа (самой уверенной/крупной
    детекции) на каждого, чтобы пользователь мог вручную вписать имя/номер.
    Кадры не пропускаются, а enhance_quality/person_conf/ball_conf/imgsz
    применяются точно так же, как в финальном прогоне (process_video) —
    чтобы ID трекера совпадали с ID, которые получатся на шаге 4 (тот же
    трекер, тот же конфиг, те же пороги, тот же сброс состояния — см.
    reset_tracker).
    """
    if model is None or cv2 is None:
        return {}, []

    reset_tracker(model)
    appearance = AppearanceMerger(
        similarity_threshold=appearance_similarity,
        jersey_ocr_enabled=jersey_ocr_enabled,
        min_person_bbox_area=min_person_bbox_area,
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {}, []

    fps = fps_hint or 25.0
    max_frames = compute_quick_scan_frame_count(fps, max_seconds)
    n_track = max_frames

    best_crops: Dict[int, Tuple[float, Any]] = {}
    frame_idx = 0
    track_done = 0
    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        track_done += 1
        if status_callback is not None:
            try:
                status_callback(
                    f"Сканирование: кадр {frame_idx + 1}/{max_frames} "
                    f"({track_done}/{n_track} с трекингом)"
                )
            except Exception:
                pass
        if progress_callback is not None:
            try:
                progress_callback(min(track_done / n_track, 1.0))
            except Exception:
                pass

        detect_frame = enhance_frame_for_detection(frame) if enhance_quality else frame
        track_conf = min(person_conf, ball_conf)
        results = model.track(
            detect_frame,
            persist=True,
            tracker=str(TRACKER_CONFIG_PATH),
            device=device,
            classes=[COCO_PERSON_CLASS_ID, COCO_BALL_CLASS_ID],
            conf=track_conf if track_conf > 0 else TRACK_CONF_DEFAULT,
            iou=TRACK_IOU_DEFAULT,
            imgsz=imgsz,
            verbose=False,
        )
        persons, _, _, _ = parse_track_results(
            results, person_conf_threshold=person_conf, ball_conf_threshold=ball_conf
        )
        if enhance_quality:
            inv_scale = 1.0 / ENHANCE_UPSCALE_FACTOR
            persons = [(pid, tuple(v * inv_scale for v in box)) for pid, box in persons]
        persons = appearance.remap(frame_idx, frame, persons, device=device)
        h, w = frame.shape[:2]
        for pid, box in persons:
            x1, y1, x2, y2 = [int(v) for v in box]
            area = max(x2 - x1, 0) * max(y2 - y1, 0)
            score = float(area)
            prev = best_crops.get(int(pid))
            if prev is None or score > prev[0]:
                x1c, y1c = max(x1, 0), max(y1, 0)
                x2c, y2c = min(x2, w), min(y2, h)
                crop = frame[y1c:y2c, x1c:x2c].copy()
                if crop.size > 0:
                    best_crops[int(pid)] = (score, crop)
        frame_idx += 1

    cap.release()
    reset_tracker(model)  # не оставляем состояние "подвешенным" перед шагом 4
    return {tid: crop for tid, (_, crop) in best_crops.items()}, appearance.merge_log


# ---------------------------------------------------------------------------
# Основной цикл обработки видео (шаг 4)
# ---------------------------------------------------------------------------
def process_video(
    video_path: str,
    model,
    device: str,
    rings: List[RingZone],
    possession_threshold: float,
    pass_min_seconds: float,
    pass_max_seconds: float,
    goal_cooldown_seconds: float,
    ball_memory_seconds: float,
    progress_bar,
    status_text,
    enhance_quality: bool = False,
    person_conf: float = PERSON_CONF_DEFAULT,
    ball_conf: float = BALL_CONF_DEFAULT,
    imgsz: int = IMGSZ_DEFAULT,
    max_gap_frames: int = BALL_MAX_GAP_FRAMES_DEFAULT,
    max_predict_frames: int = BALL_MAX_PREDICT_FRAMES_DEFAULT,
    color_fallback: bool = BALL_COLOR_FALLBACK_DEFAULT,
    color_roi_half: int = BALL_COLOR_ROI_HALF_DEFAULT,
    appearance_similarity: float = APPEARANCE_SIMILARITY_DEFAULT,
    jersey_ocr_enabled: bool = JERSEY_OCR_ENABLED_DEFAULT,
    min_person_bbox_area: float = MIN_PERSON_BBOX_AREA_DEFAULT,
    excluded_player_ids: Optional[set] = None,
    camera_transforms: Optional[List[np.ndarray]] = None,
    ball_anchors: Optional[List[Dict[str, Any]]] = None,
    manual_id_map: Optional[Dict[int, int]] = None,
    avg_player_diagonal: Optional[float] = None,
) -> Tuple[Dict[int, PlayerStats], Path, Dict[str, List[Dict[str, Any]]]]:
    """Обрабатывает видео покадрово: детекция, трекинг, события, хайлайты.

    person_conf/ball_conf/imgsz — см. константы BALL_CONF_*/PERSON_CONF_*/
    IMGSZ_* выше: инференс идёт с единым низким conf=min(person_conf,
    ball_conf), а классы фильтруются раздельно постфактум в
    parse_track_results (мяч — более мелкий и часто менее уверенный объект,
    чем игрок, поэтому его порог обычно значительно ниже).

    camera_transforms — экспериментальный режим «камера в движении»: если передан
    список накопленных матриц из estimate_camera_transforms, каждое кольцо
    пересчитывается из своего якорного кадра (ring*_frame) в текущий кадр,
    см. transform_ring_to_frame / compute_dynamic_rings. Если None — статичный
    режим (линии фиксированы).

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
    sample_every = max(int(fps // 2), 1)  # ~2 отсчёта в секунду для таймлайна отладки

    frame_buffer: Deque[Any] = deque(maxlen=buffer_len)
    ball_history: Deque[Tuple[float, float, float]] = deque(maxlen=BALL_HISTORY_MAXLEN)
    ball_trajectory: Deque[Tuple[float, float]] = deque(maxlen=BALL_TRAJECTORY_DRAW_LEN)
    pending_highlights: List[PendingHighlight] = []

    stats: Dict[int, PlayerStats] = {}
    debug_log: Dict[str, List[Dict[str, Any]]] = {
        "ownership_changes": [],
        "sampled_timeline": [],
        "ball_sources": [],
    }
    pass_min_frames = seconds_to_frames(pass_min_seconds, fps)
    pass_max_frames = max(seconds_to_frames(pass_max_seconds, fps), pass_min_frames + 1)
    goal_cooldown_frames = seconds_to_frames(goal_cooldown_seconds, fps)
    pass_flight_min_px = possession_threshold * PASS_FLIGHT_MIN_POSSESSION_FRACTION

    predict_limit = max(max_predict_frames, int(ball_memory_seconds * fps))
    ball_color_cal = build_ball_color_calibration_from_anchors(
        video_path, ball_anchors, camera_transforms
    )
    ball_tracker = BallTracker(
        max_gap_frames=max_gap_frames,
        max_predict_frames=predict_limit,
        color_fallback=color_fallback,
        color_roi_half=color_roi_half,
        model=model,
        device=device,
        ball_conf=ball_conf,
        imgsz=imgsz,
    )
    ball_tracker.csrt_max_jump_px = compute_csrt_max_jump_px(fps, avg_player_diagonal, imgsz)
    ball_tracker.color_calibration = ball_color_cal
    ball_anchor_guide = (
        BallAnchorGuide(ball_anchors, camera_transforms)
        if normalize_ball_anchors(ball_anchors)
        else None
    )
    appearance_merger = AppearanceMerger(
        similarity_threshold=appearance_similarity,
        jersey_ocr_enabled=jersey_ocr_enabled,
        min_person_bbox_area=min_person_bbox_area,
    )
    resolve_player_id = make_id_resolver(manual_id_map)
    id_former_labels = build_id_former_labels(manual_id_map)
    excluded_ids: set = {resolve_player_id(int(pid)) for pid in (excluded_player_ids or [])}
    ring_draw_warnings: List[str] = []

    # Состояние владения мячом (для пасов и для "кто владел мячом перед голом").
    last_owner: Optional[int] = None
    pass_origin: Optional[int] = None
    free_ball_frames: int = 0
    pass_free_start_xy: Optional[Tuple[float, float]] = None
    last_goal_episode_frame: int = -10**9
    prev_event_ball_xy: Optional[Tuple[float, float]] = None
    last_event_ball_frame: int = -10**9
    event_ball_gap_max_frames = max(
        seconds_to_frames(EVENT_BALL_MAX_GAP_SECONDS, fps),
        pass_max_frames,
    )

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
        ball: Optional[Tuple[float, float]] = None
        ball_state: Optional[BallTrackState] = None
        ball_source: Optional[str] = None
        ball_lost = False

        if model is not None:
            # Опционально апскейлим+резчим кадр перед детекцией (помогает
            # трекеру/ReID на видео низкого разрешения), затем пересчитываем
            # координаты обратно в исходный масштаб и рисуем на ОРИГИНАЛЬНОМ
            # кадре, чтобы выходное видео сохранило исходное разрешение.
            detect_frame = enhance_frame_for_detection(frame) if enhance_quality else frame
            # persist=True сохраняет ID треков между кадрами одного видео.
            # conf=min(person_conf, ball_conf) — намеренно ЕДИНЫЙ низкий порог на
            # инференсе (per-class conf в ultralytics track()/predict() не
            # поддерживается), чтобы не потерять мяч на этапе NMS модели; более
            # строгая фильтрация по каждому классу — ниже, в parse_track_results.
            track_conf = min(person_conf, ball_conf)
            results = model.track(
                detect_frame,
                persist=True,
                tracker=tracker_path,
                device=device,
                classes=[COCO_PERSON_CLASS_ID, COCO_BALL_CLASS_ID],
                conf=track_conf if track_conf > 0 else TRACK_CONF_DEFAULT,
                iou=TRACK_IOU_DEFAULT,
                imgsz=imgsz,
                verbose=False,
            )
            persons, yolo_ball, yolo_ball_conf, yolo_ball_bbox = parse_track_results(
                results, person_conf_threshold=person_conf, ball_conf_threshold=ball_conf
            )
            if enhance_quality:
                inv_scale = 1.0 / ENHANCE_UPSCALE_FACTOR
                persons = [(pid, tuple(v * inv_scale for v in box)) for pid, box in persons]
                if yolo_ball is not None:
                    yolo_ball = (yolo_ball[0] * inv_scale, yolo_ball[1] * inv_scale)
                if yolo_ball_bbox is not None:
                    yolo_ball_bbox = tuple(v * inv_scale for v in yolo_ball_bbox)
            persons = appearance_merger.remap(frame_idx, frame, persons, device=device)
            persons = [(resolve_player_id(pid), box) for pid, box in persons]
            ball_state = ball_tracker.update(
                frame_idx,
                frame,
                yolo_ball,
                yolo_ball_conf,
                persons,
                yolo_bbox=yolo_ball_bbox,
                ball_anchor_guide=ball_anchor_guide,
            )
            if ball_state is not None:
                ball = (ball_state.x, ball_state.y)
                ball_source = ball_state.source
            elif ball_tracker.last_confident_pos is not None and ball_tracker.frames_since_any < ball_tracker.max_predict_frames:
                ball_lost = True

        event_ball = event_ball_from_tracking(model, ball_state, ball_lost, ball)
        if model is None:
            persons = []
            annotated = frame.copy()
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

        if camera_transforms is not None:
            current_rings = compute_dynamic_rings(rings, camera_transforms, frame_idx)
        else:
            current_rings = rings

        effective_ball: Optional[Tuple[float, float]] = ball
        if effective_ball is not None:
            ball_history.append((t, effective_ball[0], effective_ball[1]))
            ball_trajectory.append((effective_ball[0], effective_ball[1]))

        event_persons = [(pid, box) for pid, box in persons if pid not in excluded_ids]
        if model is not None:
            annotated = draw_annotations(
                frame,
                persons,
                effective_ball,
                list(ball_trajectory),
                ball_source=ball_source,
                excluded_ids=excluded_ids,
                ball_lost=ball_lost if model is not None else False,
                last_ball=ball_tracker.last_confident_pos if model is not None else None,
                id_former_labels=id_former_labels,
            )
            annotated = draw_hoop_lines_on_frame(annotated, current_rings, warnings_out=ring_draw_warnings)
        elif effective_ball is not None:
            annotated = draw_hoop_lines_on_frame(annotated, current_rings, warnings_out=ring_draw_warnings)

        # -------------------------------------------------------------
        # ВЛАДЕНИЕ МЯЧОМ И ДЕТЕКЦИЯ ПЕРЕДАЧ (ПАСОВ)
        #
        # Игрок владеет мячом, если расстояние от центра мяча до его bbox
        # меньше possession_threshold px (дефолт 90). Передача: Игрок_1 владел
        # мячом → мяч летел без владельца pass_min_frames..pass_max_frames
        # кадров → Игрок_2 получил владение → +1 пас Игроку_1.
        # -------------------------------------------------------------
        current_owner: Optional[int] = None
        if event_ball is not None and event_persons:
            best_dist = None
            for pid, box in event_persons:
                d = distance_point_to_bbox(event_ball[0], event_ball[1], box)
                if best_dist is None or d < best_dist:
                    best_dist, current_owner = d, pid
            if best_dist is not None and best_dist > possession_threshold:
                current_owner = None  # мяч ничейный/в полёте — слишком далеко от всех игроков

        if current_owner is not None:
            pass_flight_ok = True
            if pass_origin is not None and pass_free_start_xy is not None and event_ball is not None:
                flight_dist = math.hypot(
                    event_ball[0] - pass_free_start_xy[0],
                    event_ball[1] - pass_free_start_xy[1],
                )
                pass_flight_ok = flight_dist >= pass_flight_min_px
            if (
                pass_origin is not None
                and pass_origin not in excluded_ids
                and current_owner not in excluded_ids
                and pass_origin != current_owner
                and pass_min_frames <= free_ball_frames <= pass_max_frames
                and pass_flight_ok
            ):
                reason = "✅ передача засчитана"
                stats.setdefault(pass_origin, blank_stats())["passes"] += 1
                stats.setdefault(current_owner, blank_stats())
                pending_highlights.append(
                    PendingHighlight(
                        filename=f"pass_from_ID{pass_origin}_to_ID{current_owner}_frame_{frame_idx}.mp4",
                        past_frames=list(frame_buffer),
                        frames_needed=future_frames_needed,
                    )
                )
                debug_log["ownership_changes"].append(
                    {
                        "Кадр": frame_idx,
                        "Время, с": round(t, 2),
                        "От игрока": pass_origin,
                        "К игроку": current_owner,
                        "Кадров без владельца": free_ball_frames,
                        "Результат": reason,
                    }
                )
            elif current_owner != last_owner:
                reason = "смена владельца (не пас)"
                if pass_origin is not None and free_ball_frames > 0:
                    if free_ball_frames < pass_min_frames:
                        reason = (
                            f"❌ отклонено (< {pass_min_seconds:.2f} с без владельца)"
                        )
                    elif free_ball_frames > pass_max_frames:
                        reason = (
                            f"❌ отклонено (> {pass_max_seconds:.2f} с без владельца)"
                        )
                    elif not pass_flight_ok:
                        reason = (
                            f"❌ отклонено (полёт < {pass_flight_min_px:.0f} px — вероятно ведение)"
                        )
                debug_log["ownership_changes"].append(
                    {
                        "Кадр": frame_idx,
                        "Время, с": round(t, 2),
                        "От игрока": last_owner if last_owner is not None else "—",
                        "К игроку": current_owner,
                        "Кадров без владельца": free_ball_frames if free_ball_frames else None,
                        "Результат": reason,
                    }
                )
            pass_origin = None
            free_ball_frames = 0
            pass_free_start_xy = None
            last_owner = current_owner
        elif event_ball is not None:
            if last_owner is not None and last_owner not in excluded_ids:
                if pass_origin is None:
                    pass_origin = last_owner
                    pass_free_start_xy = event_ball
                free_ball_frames += 1
                if free_ball_frames > pass_max_frames:
                    pass_origin = None
                    pass_free_start_xy = None
        elif pass_origin is not None:
            free_ball_frames += 1
            if free_ball_frames > pass_max_frames:
                pass_origin = None
                pass_free_start_xy = None
        else:
            if free_ball_frames > pass_max_frames:
                pass_origin = None
                pass_free_start_xy = None
                free_ball_frames = 0

        if frame_idx % sample_every == 0:
            debug_log["sampled_timeline"].append(
                {
                    "Время, с": round(t, 2),
                    "Владелец (ID)": float(current_owner) if current_owner is not None else float("nan"),
                }
            )
            if ball_source:
                debug_log["ball_sources"].append(
                    {"Кадр": frame_idx, "Время, с": round(t, 2), "Источник мяча": ball_source}
                )

        # -------------------------------------------------------------
        # ГОЛЫ: пересечение траектории мяча с линией кольца (только надёжный source).
        # Общий кулдаун эпизода у кольца в секундах (не per-ring).
        # -------------------------------------------------------------
        goal_prev_xy = effective_prev_event_ball_xy(
            prev_event_ball_xy,
            last_event_ball_frame,
            frame_idx,
            event_ball_gap_max_frames,
            camera_transforms=camera_transforms,
        )
        if event_ball is not None and goal_prev_xy is not None:
            bx_prev, by_prev = goal_prev_xy
            bx_curr, by_curr = event_ball[0], event_ball[1]
            if (frame_idx - last_goal_episode_frame) >= goal_cooldown_frames:
                for ring_idx, ring in enumerate(current_rings):
                    if not ring.get("configured", True):
                        continue
                    half_w = ring.get("half_width", ring.get("r", 40.0))
                    if not segment_crosses_hoop_line_top_to_bottom(
                        bx_prev, by_prev, bx_curr, by_curr, ring["y"], ring["x"], half_w
                    ):
                        continue
                    last_goal_episode_frame = frame_idx
                    credited_player = last_owner
                    if credited_player is None:
                        credited_player = nearest_player_to_point(event_persons, (ring["x"], ring["y"]))
                    if credited_player is not None and credited_player not in excluded_ids:
                        stats.setdefault(credited_player, blank_stats())["makes"] += 1
                        pending_highlights.append(
                            PendingHighlight(
                                filename=f"goal_ring{ring_idx + 1}_ID{credited_player}_frame_{frame_idx}.mp4",
                                past_frames=list(frame_buffer),
                                frames_needed=future_frames_needed,
                            )
                        )
                    break

        if event_ball is not None:
            prev_event_ball_xy = (event_ball[0], event_ball[1])
            last_event_ball_frame = frame_idx

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
    output_path = reencode_for_browser(output_path)
    if appearance_merger.merge_log:
        debug_log["id_merges"] = appearance_merger.merge_log
    if manual_id_map:
        debug_log["manual_id_map"] = dict(manual_id_map)
    if ring_draw_warnings:
        debug_log["ring_draw_warnings"] = list(dict.fromkeys(ring_draw_warnings))
    return stats, output_path, debug_log


def build_box_score(
    stats: Dict[int, PlayerStats],
    player_names: Optional[Dict[int, str]] = None,
    player_numbers: Optional[Dict[int, str]] = None,
    excluded_player_ids: Optional[set] = None,
    manual_id_map: Optional[Dict[int, int]] = None,
) -> pd.DataFrame:
    """Формирует итоговую таблицу статистики (Box Score) по игрокам."""
    resolve_player_id = make_id_resolver(manual_id_map)
    excluded = {resolve_player_id(int(pid)) for pid in (excluded_player_ids or set())}
    columns = ["ID игрока", "Имя", "Номер", "Попадания", "Сделано передач"]
    if not stats:
        return pd.DataFrame(columns=columns)

    rows = []
    for pid, s in sorted(stats.items()):
        if pid in excluded:
            continue
        makes, passes = s["makes"], s["passes"]
        name, number = lookup_player_meta(pid, player_names, player_numbers, manual_id_map)
        name = name or f"Игрок {pid}"
        number = number or "—"
        rows.append(
            {
                "ID игрока": pid,
                "Имя": name,
                "Номер": number,
                "Попадания": makes,
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
        "ring1_configured": False,
        "ring1_frame": 0,
        "ring2_x": 0.0,
        "ring2_y": 0.0,
        "ring2_r": 40.0,
        "ring2_configured": False,
        "ring2_frame": 0,
        "ring1_anchors": [],
        "ring2_anchors": [],
        "preview_frame_idx": 0,
        "possession_threshold": float(POSSESSION_THRESHOLD_DEFAULT),
        "pass_min_seconds": float(PASS_MIN_SECONDS_DEFAULT),
        "pass_max_seconds": float(PASS_MAX_SECONDS_DEFAULT),
        "ball_memory": float(BALL_MEMORY_SECONDS_DEFAULT),
        "goal_cooldown_seconds": float(GOAL_COOLDOWN_SECONDS_DEFAULT),
        "enhance_quality": False,
        "person_conf": float(PERSON_CONF_DEFAULT),
        "ball_conf": float(BALL_CONF_DEFAULT),
        "imgsz": int(IMGSZ_DEFAULT),
        "ball_max_gap_frames": int(BALL_MAX_GAP_FRAMES_DEFAULT),
        "ball_max_predict_frames": int(BALL_MAX_PREDICT_FRAMES_DEFAULT),
        "ball_color_fallback": BALL_COLOR_FALLBACK_DEFAULT,
        "ball_color_roi_half": int(BALL_COLOR_ROI_HALF_DEFAULT),
        "camera_mode": CAMERA_MODE_STATIC,
        "ball_diagnostics": None,
        "preview_time": 0.0,
        "camera_transforms": None,
        "camera_transforms_video": None,
        "avg_player_diagonal": None,
        "auto_threshold_computed_for": None,
        "click_target_ring": "Кольцо 1",
        "ball_anchors": [],
        "_last_ring_click_time": None,
        "player_crops": {},
        "player_names": {},
        "player_numbers": {},
        "box_score_df": None,
        "debug_log": None,
        "last_output_video": None,
        "highlight_files": [],
        "excluded_player_ids": [],
        "appearance_similarity": float(APPEARANCE_SIMILARITY_DEFAULT),
        "jersey_ocr_enabled": JERSEY_OCR_ENABLED_DEFAULT,
        "min_person_bbox_area": int(MIN_PERSON_BBOX_AREA_DEFAULT),
        "id_merge_log": [],
        "manual_id_map": {},
        "manual_id_merge_log": [],
        "ball_interp_skipped": [],
        "ball_interp_fix_frame": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def reset_for_new_video() -> None:
    """Сбрасывает всё, что зависит от конкретного видео (при загрузке нового)."""
    clear_highlights_directory()
    reset_analysis_results()
    for key in (
        "rings_initialized_for",
        "ring1_configured",
        "ring2_configured",
        "camera_transforms",
        "camera_transforms_video",
        "player_crops",
        "player_names",
        "player_numbers",
        "avg_player_diagonal",
        "auto_threshold_computed_for",
        "_last_ring_click_time",
        "ball_diagnostics",
        "excluded_player_ids",
        "id_merge_log",
        "ball_anchors",
        "ring1_anchors",
        "ring2_anchors",
        "manual_id_map",
        "manual_id_merge_log",
        "ball_interp_skipped",
        "ball_interp_fix_frame",
    ):
        if key in ("player_crops", "player_names", "player_numbers"):
            st.session_state[key] = {}
        elif key in (
            "excluded_player_ids", "id_merge_log", "ball_anchors", "ball_interp_skipped",
            "ring1_anchors", "ring2_anchors",
        ):
            st.session_state[key] = []
        elif key in ("manual_id_map",):
            st.session_state[key] = {}
        elif key == "manual_id_merge_log":
            st.session_state[key] = []
        elif key == "ball_interp_fix_frame":
            st.session_state[key] = None
        elif key in ("ring1_configured", "ring2_configured"):
            st.session_state[key] = False
        else:
            st.session_state[key] = None
    for widget_key in list(st.session_state.keys()):
        if str(widget_key).startswith("wi_ring"):
            del st.session_state[widget_key]


def _make_ring_widget_change_handler(ring_num: int):
    def _handler() -> None:
        sync_ring_widgets_to_canonical(st.session_state, ring_num)
        if st.session_state.get("camera_mode") == CAMERA_MODE_PANNING:
            st.session_state[f"ring{ring_num}_frame"] = int(st.session_state.get("preview_frame_idx", 0))
        else:
            st.session_state[f"ring{ring_num}_frame"] = 0

    return _handler


def _ring_anchor_frame_idx() -> int:
    """Номер кадра-якоря для кольца: в статике всегда 0, в динамике — текущий кадр превью."""
    if st.session_state.get("camera_mode") == CAMERA_MODE_PANNING:
        return int(st.session_state.get("preview_frame_idx", 0))
    return 0


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

    st.subheader("📷 Режим камеры")
    camera_mode_label = st.radio(
        "Как снято видео?",
        options=[CAMERA_MODE_LABELS[CAMERA_MODE_STATIC], CAMERA_MODE_LABELS[CAMERA_MODE_PANNING]],
        index=0 if st.session_state.get("camera_mode", CAMERA_MODE_STATIC) == CAMERA_MODE_STATIC else 1,
        help="Выбор влияет на то, как обрабатываются зоны колец на шаге 4.",
    )
    st.session_state["camera_mode"] = (
        CAMERA_MODE_PANNING if camera_mode_label == CAMERA_MODE_LABELS[CAMERA_MODE_PANNING] else CAMERA_MODE_STATIC
    )
    if st.session_state["camera_mode"] == CAMERA_MODE_PANNING:
        st.warning(
            "⚠️ Экспериментальный режим. Позиции обеих зон колец, заданные на шаге 2, будут "
            "автоматически пересчитываться под сдвиг камеры на каждом кадре (оптический поток "
            "по фоновым фичам). Надёжно работает для **плавной** панорамы/наклона камеры — при "
            "резком зуме, сильной тряске или смене плана оценка сдвига может потерять точность, "
            "и виртуальные зоны колец разъедутся с реальными кольцами на площадке."
        )
    else:
        st.caption("Зоны колец, заданные на шаге 2, останутся неподвижными во всех кадрах (как раньше).")

    st.write("Загрузите видео тренировки, снятое так, чтобы кольцо(-а) и площадка были в кадре.")

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
def render_step2_professional_settings(device: str, video_path: str, meta: Dict[str, float]) -> None:
    """Дополнительные пороги и трекинг — только для опытных пользователей."""
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Кольцо №1 (точные координаты)**")
        st.number_input(
            "X1 (px)", min_value=0, max_value=int(meta["width"]), key="wi_ring1_x",
            on_change=_make_ring_widget_change_handler(1),
        )
        st.number_input(
            "Y1 (px)", min_value=0, max_value=int(meta["height"]), key="wi_ring1_y",
            on_change=_make_ring_widget_change_handler(1),
        )
    with col2:
        st.markdown("**Кольцо №2 (точные координаты)**")
        st.number_input(
            "X2 (px)", min_value=0, max_value=int(meta["width"]), key="wi_ring2_x",
            on_change=_make_ring_widget_change_handler(2),
        )
        st.number_input(
            "Y2 (px)", min_value=0, max_value=int(meta["height"]), key="wi_ring2_y",
            on_change=_make_ring_widget_change_handler(2),
        )
    sync_ring_widgets_to_canonical(st.session_state, 1)
    sync_ring_widgets_to_canonical(st.session_state, 2)

    colcal1, colcal2 = st.columns([3, 1])
    with colcal1:
        if st.session_state.get("avg_player_diagonal"):
            st.caption(
                f"📏 Средний размер игрока на кадре превью: ~{st.session_state['avg_player_diagonal']:.0f}px → "
                f"авто-порог владения ~{suggest_possession_threshold(st.session_state['avg_player_diagonal'])}px."
            )
        else:
            st.caption(
                "Авто-калибровка порога недоступна (модель не загружена или игроки не найдены) — "
                f"дефолт {POSSESSION_THRESHOLD_DEFAULT}px."
            )
    with colcal2:
        if st.button("🔄 Пересчитать порог по кадру", use_container_width=True):
            st.session_state["auto_threshold_computed_for"] = None
            st.rerun()

    st.markdown("**Владение мячом, передачи и кулдаун**")
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
        st.session_state["pass_min_seconds"] = st.slider(
            "Мин. время без владельца для паса (сек)",
            min_value=PASS_MIN_SECONDS_MIN,
            max_value=PASS_MIN_SECONDS_MAX,
            value=float(st.session_state.get("pass_min_seconds", PASS_MIN_SECONDS_DEFAULT)),
            step=0.05,
        )
        st.session_state["pass_max_seconds"] = st.slider(
            "Макс. время без владельца для паса (сек)",
            min_value=PASS_MAX_SECONDS_MIN,
            max_value=PASS_MAX_SECONDS_MAX,
            value=float(st.session_state.get("pass_max_seconds", PASS_MAX_SECONDS_DEFAULT)),
            step=0.1,
        )
    with c3:
        st.session_state["ball_memory"] = st.slider(
            "Память мяча при потере детекции (сек)",
            min_value=BALL_MEMORY_SECONDS_MIN,
            max_value=BALL_MEMORY_SECONDS_MAX,
            value=float(st.session_state["ball_memory"]),
            step=0.05,
        )

    st.session_state["goal_cooldown_seconds"] = st.slider(
        "Кулдаун гола у кольца (сек, общий на эпизод)",
        min_value=GOAL_COOLDOWN_SECONDS_MIN,
        max_value=GOAL_COOLDOWN_SECONDS_MAX,
        value=float(st.session_state.get("goal_cooldown_seconds", GOAL_COOLDOWN_SECONDS_DEFAULT)),
        step=0.25,
    )

    st.markdown("**Детекция мяча и производительность (YOLO)**")
    cconf1, cconf2 = st.columns(2)
    with cconf1:
        st.session_state["ball_conf"] = st.slider(
            "Порог уверенности для мяча (conf)",
            min_value=BALL_CONF_MIN, max_value=BALL_CONF_MAX,
            value=float(st.session_state["ball_conf"]), step=0.01,
        )
    with cconf2:
        st.session_state["person_conf"] = st.slider(
            "Порог уверенности для игроков (conf)",
            min_value=PERSON_CONF_MIN, max_value=PERSON_CONF_MAX,
            value=float(st.session_state["person_conf"]), step=0.05,
        )
    st.session_state["imgsz"] = st.select_slider(
        "Разрешение инференса (imgsz, px)",
        options=IMGSZ_OPTIONS,
        value=int(st.session_state["imgsz"]) if int(st.session_state["imgsz"]) in IMGSZ_OPTIONS else IMGSZ_DEFAULT,
    )

    st.markdown("**Устойчивый трекинг мяча (Kalman + цвет)**")
    st.session_state["ball_color_fallback"] = st.checkbox(
        "Цветовой fallback мяча в ROI (HSV из кликов шага 2 или оранжевый дефолт)",
        value=bool(st.session_state.get("ball_color_fallback", BALL_COLOR_FALLBACK_DEFAULT)),
    )
    bgap1, bgap2, broi = st.columns(3)
    with bgap1:
        st.session_state["ball_max_gap_frames"] = st.slider(
            "Макс. кадров интерполяции",
            min_value=BALL_MAX_GAP_FRAMES_MIN,
            max_value=BALL_MAX_GAP_FRAMES_MAX,
            value=int(st.session_state.get("ball_max_gap_frames", BALL_MAX_GAP_FRAMES_DEFAULT)),
            step=1,
        )
    with bgap2:
        st.session_state["ball_max_predict_frames"] = st.slider(
            "Макс. кадров виртуального мяча",
            min_value=BALL_MAX_PREDICT_FRAMES_MIN,
            max_value=BALL_MAX_PREDICT_FRAMES_MAX,
            value=int(st.session_state.get("ball_max_predict_frames", BALL_MAX_PREDICT_FRAMES_DEFAULT)),
            step=1,
        )
    with broi:
        st.session_state["ball_color_roi_half"] = st.slider(
            "Полуразмер ROI цвета (px)",
            min_value=BALL_COLOR_ROI_HALF_MIN,
            max_value=BALL_COLOR_ROI_HALF_MAX,
            value=int(st.session_state.get("ball_color_roi_half", BALL_COLOR_ROI_HALF_DEFAULT)),
            step=10,
        )

    st.session_state["min_person_bbox_area"] = st.slider(
        "Мин. площадь bbox игрока (px², 0 = не фильтровать)",
        min_value=MIN_PERSON_BBOX_AREA_MIN,
        max_value=MIN_PERSON_BBOX_AREA_MAX,
        value=int(st.session_state.get("min_person_bbox_area", MIN_PERSON_BBOX_AREA_DEFAULT)),
        step=16,
    )
    st.session_state["appearance_similarity"] = st.slider(
        "Порог похожести игроков (склейка ID / ReID)",
        min_value=APPEARANCE_SIMILARITY_MIN,
        max_value=APPEARANCE_SIMILARITY_MAX,
        value=float(st.session_state.get("appearance_similarity", APPEARANCE_SIMILARITY_DEFAULT)),
        step=0.01,
    )
    st.session_state["jersey_ocr_enabled"] = st.checkbox(
        "EasyOCR номеров на форме (только если номера реально читаются на видео)",
        value=bool(st.session_state.get("jersey_ocr_enabled", JERSEY_OCR_ENABLED_DEFAULT)),
    )
    if not st.session_state["jersey_ocr_enabled"]:
        st.caption(
            "По умолчанию OCR выключен — на большинстве тренировочных видео номера не читаются "
            "и OCR даёт ложные цифры. Включайте только при крупном плане и чётких номерах."
        )
    elif easyocr is None:
        st.caption(f"⚠️ EasyOCR недоступен: {EASYOCR_IMPORT_ERROR or 'не установлен'}")

    st.session_state["enhance_quality"] = st.checkbox(
        "Улучшить качество кадра перед детекцией (апскейл + резкость)",
        value=st.session_state.get("enhance_quality", False),
    )


def render_step2_zones(device: str) -> None:
    st.header("Шаг 2 — Превью и настройка зон")
    video_path = st.session_state.get("video_path")
    if not video_path or not Path(video_path).exists():
        st.warning("Сначала загрузите видео на шаге 1.")
        if st.button("← Назад к загрузке"):
            go_to_step(1)
        return

    consume_pending_step2_ui_state(st.session_state)

    # Клик по превью обновляет session_state["ring*_x"/"_y"] ДО того, как ниже
    # инстанциируются number_input с теми же ключами (иначе — StreamlitAPIException
    # "cannot be modified after the widget... is instantiated", см. комментарий
    # у блока клика ниже). НО сам st.rerun() после этого откладывается до самого
    # конца функции (см. ring_click_triggered_rerun ниже) — если вызвать rerun
    # сразу, скрипт обрывается ДО того, как в этом прогоне вообще были созданы
    # number_input для колец, и Streamlit считает их состояние "осиротевшим"
    # (виджет не был отрисован в прогоне) и сбрасывает его к дефолтам на
    # следующем прогоне. Поэтому rerun обязательно должен произойти ПОСЛЕ того,
    # как все виджеты этого прогона (включая все 6 number_input) уже созданы.
    ring_click_triggered_rerun = False

    meta = get_video_metadata(video_path)

    if st.session_state.get("rings_initialized_for") != video_path:
        apply_default_ring_positions(st.session_state, int(meta["width"]), int(meta["height"]))
        st.session_state["rings_initialized_for"] = video_path
    init_ring_widget_keys(st.session_state, 1)
    init_ring_widget_keys(st.session_state, 2)

    max_frame_idx = max(int(meta["total_frames"]) - 1, 0)

    # --- Авто-калибровка порога владения по среднему размеру игрока на кадре ---
    # Фиксированный порог в пикселях не учитывает масштаб конкретного видео
    # (камера близко/далеко, разное разрешение) — поэтому один раз на видео
    # (и по кнопке повторно) считаем средний размер рамки игрока и предлагаем
    # адаптивный дефолт вместо жёстких 90px.
    panning_mode = st.session_state.get("camera_mode") == CAMERA_MODE_PANNING
    camera_transforms_preview: Optional[List[np.ndarray]] = None
    if panning_mode:
        ensure_ring_anchors_migrated(st.session_state, 1)
        ensure_ring_anchors_migrated(st.session_state, 2)
        st.info(
            "🎥 **Динамическое видео:** каждое кольцо привязывается к **номеру кадра**, "
            "на котором вы кликнули. Сдвигайте ползунок ниже, чтобы проверить, как линия "
            "проецируется на другие кадры."
        )
        total_frames_cam = max(int(meta["total_frames"]), 1)
        if st.session_state.get("camera_transforms_video") != video_path:
            st.caption("Оценка движения камеры для превью колец…")
            cam_progress = st.progress(0.0)
            cam_status = st.empty()

            def _camera_preview_progress(current: int, total: int) -> None:
                cam_progress.progress(min(current / max(total, 1), 1.0))
                cam_status.caption(f"Оценка движения камеры: кадр **{current}** / **{total}**")

            get_camera_transforms_cached(
                video_path, st.session_state, frame_progress_callback=_camera_preview_progress
            )
            cam_progress.progress(1.0)
            cam_status.caption(f"Оценка движения камеры завершена ({total_frames_cam} кадров).")
        else:
            st.caption("Оценка движения камеры готова (используется кэш для этого видео).")
        camera_transforms_preview = st.session_state.get("camera_transforms")

    st.subheader("🎯 Положение колец и мяча")
    if streamlit_image_coordinates is not None and cv2 is not None:
        current_target = st.session_state.get("click_target_ring", "Кольцо 1")
        if current_target not in CLICK_TARGET_OPTIONS:
            current_target = "Кольцо 1"
        st.session_state["click_target_ring"] = st.radio(
            "Сейчас клик по превью задаёт:",
            list(CLICK_TARGET_OPTIONS),
            horizontal=True,
            key="click_target_ring_radio",
            index=list(CLICK_TARGET_OPTIONS).index(current_target),
        )
    else:
        current_target = st.session_state.get("click_target_ring", "Кольцо 1")
        st.caption(
            "Пакет streamlit-image-coordinates не установлен — доступна только точная настройка "
            "числовыми полями ниже (см. requirements.txt)."
        )

    click_target = st.session_state.get("click_target_ring", current_target)
    show_frame_slider = panning_mode or click_target == "Мяч"

    if click_target == "Мяч":
        st.caption(
            "Кликните мяч на **2–10 кадрах**, где он хорошо виден — так трекер реже теряет его "
            "между детекциями YOLO. Особенно полезны кадры **броска** и **прилёта в кольцо**."
        )
    elif click_target == "Кольцо 1":
        if panning_mode:
            st.caption(
                "На **нескольких кадрах** выберите кадр и **кликните по ободу** кольца 1 — "
                "каждый клик добавляет якорь (повтор на том же кадре заменяет). Между якорями "
                "линия интерполируется с учётом движения камеры."
            )
        else:
            st.caption(
                "**Кликните по ободу** кольца 1 на превью — кадр выбирать не нужно, "
                "линия на всех кадрах в одном месте."
            )
    else:
        if panning_mode:
            st.caption(
                "На **нескольких кадрах** выберите кадр и **кликните по ободу** кольца 2 — "
                "каждый клик добавляет якорь (повтор на том же кадре заменяет). Между якорями "
                "линия интерполируется с учётом движения камеры."
            )
        else:
            st.caption(
                "**Кликните по ободу** кольца 2 на превью — кадр выбирать не нужно, "
                "линия на всех кадрах в одном месте."
            )

    half_w_col1, half_w_col2 = st.columns(2)
    with half_w_col1:
        st.number_input(
            "Полуширина линии кольца 1 (px)",
            min_value=5,
            max_value=int(max(meta["width"], meta["height"])),
            key="wi_ring1_r",
            on_change=_make_ring_widget_change_handler(1),
            help="Половина длины горизонтального отрезка линии (от центра влево/вправо).",
        )
    with half_w_col2:
        st.number_input(
            "Полуширина линии кольца 2 (px)",
            min_value=5,
            max_value=int(max(meta["width"], meta["height"])),
            key="wi_ring2_r",
            on_change=_make_ring_widget_change_handler(2),
            help="Половина длины горизонтального отрезка линии (от центра влево/вправо).",
        )
    sync_ring_widgets_to_canonical(st.session_state, 1)
    sync_ring_widgets_to_canonical(st.session_state, 2)

    if show_frame_slider:
        if int(st.session_state.get("preview_frame_idx", 0)) > max_frame_idx:
            st.session_state["preview_frame_idx"] = max_frame_idx
        slider_label = (
            "Кадр для настройки (кольца и мяч привязываются к этому кадру)"
            if panning_mode
            else "Кадр для якорей мяча"
        )
        st.slider(
            slider_label,
            min_value=0,
            max_value=max_frame_idx,
            step=1,
            key="preview_frame_idx",
        )
        preview_frame_idx = int(st.session_state["preview_frame_idx"])
        st.session_state["preview_time"] = float(preview_frame_idx) / float(meta["fps"] or 25.0)
        st.caption(f"Текущий кадр превью: **{preview_frame_idx}** (~{st.session_state['preview_time']:.2f} с)")
    else:
        st.session_state["preview_frame_idx"] = 0
        preview_frame_idx = 0
        st.session_state["preview_time"] = 0.0
        st.caption(
            "Статичное видео: кольца задаются в одной позиции на всех кадрах — "
            "ползунок кадра не нужен (переключитесь на «Мяч» для выбора кадра)."
        )

    frame = extract_frame_at_index(video_path, preview_frame_idx)
    if frame is not None and st.session_state.get("auto_threshold_computed_for") != video_path:
        model, _ = load_model(device)
        if model is not None:
            with st.spinner("Авто-калибровка порога владения по кадру..."):
                avg_diag = estimate_player_scale(
                    frame, model, device,
                    person_conf=float(st.session_state.get("person_conf", PERSON_CONF_DEFAULT)),
                    imgsz=int(st.session_state.get("imgsz", IMGSZ_DEFAULT)),
                )
            if avg_diag:
                st.session_state["avg_player_diagonal"] = avg_diag
                st.session_state["possession_threshold"] = suggest_possession_threshold(avg_diag)
        st.session_state["auto_threshold_computed_for"] = video_path

    # ВАЖНО: превью + обработка клика по картинке (streamlit_image_coordinates)
    # ДОЛЖНЫ идти строго ДО того, как ниже создаются number_input с ключами
    # "ring1_x"/"ring1_y"/"ring2_x"/"ring2_y". Обработчик клика программно
    # пишет в st.session_state[f"{target}_x"]/["_y"] — если бы это происходило
    # ПОСЛЕ инстанциирования виджетов с теми же ключами в этом же прогоне
    # скрипта, Streamlit кидает StreamlitAPIException "cannot be modified
    # after the widget ... is instantiated" (ровно так и было до этого
    # фикса: блок клика был ниже number_input). Здесь клик только читает
    # текущие ring*_x/y/r из session_state (виджеты с этими ключами в
    # текущем прогоне ещё не создавались) — значит запись в эти ключи всё
    # ещё разрешена. Сам st.rerun() при этом откладывается флагом
    # ring_click_triggered_rerun до конца функции — см. комментарий там.
    if frame is not None:
        rings = rings_for_preview_display(
            st.session_state,
            preview_frame_idx,
            camera_transforms=camera_transforms_preview,
            panning_mode=panning_mode,
        )
        preview_bgr = draw_zones_preview(
            frame,
            rings,
            possession_threshold=st.session_state["possession_threshold"],
            preview_frame_idx=preview_frame_idx,
            show_anchor_debug=panning_mode,
            ball_anchors=st.session_state.get("ball_anchors"),
            camera_transforms=camera_transforms_preview,
            panning_mode=panning_mode,
        )
        preview_rgb = cv2.cvtColor(preview_bgr, cv2.COLOR_BGR2RGB)

        if streamlit_image_coordinates is not None:
            display_width = min(int(meta["width"]), PREVIEW_MAX_DISPLAY_WIDTH)
            click_value = streamlit_image_coordinates(
                preview_rgb, key=f"ring_click_canvas_{preview_frame_idx}", width=display_width
            )
            if click_value is not None and click_value.get("x") is not None:
                click_time = click_value.get("unix_time")
                if click_time != st.session_state.get("_last_ring_click_time"):
                    st.session_state["_last_ring_click_time"] = click_time
                    disp_w = click_value.get("width") or display_width
                    disp_h = click_value.get("height") or int(meta["height"] * display_width / meta["width"])
                    scale_x = meta["width"] / disp_w if disp_w else 1.0
                    scale_y = meta["height"] / disp_h if disp_h else 1.0
                    orig_x = int(np.clip(click_value["x"] * scale_x, 0, meta["width"]))
                    orig_y = int(np.clip(click_value["y"] * scale_y, 0, meta["height"]))
                    target = st.session_state["click_target_ring"]
                    if target == "Мяч":
                        anchor_frame = int(st.session_state["preview_frame_idx"])
                        add_ball_anchor(st.session_state, orig_x, orig_y, frame_idx=anchor_frame)
                        append_ball_training_seed(
                            video_path,
                            st.session_state.get("video_name"),
                            anchor_frame,
                            orig_x,
                            orig_y,
                            frame,
                            source="user_click",
                        )
                        st.session_state["ball_interp_fix_frame"] = None
                    else:
                        ring_num = 1 if target == "Кольцо 1" else 2
                        if panning_mode:
                            sync_ring_widgets_to_canonical(st.session_state, ring_num)
                            add_ring_anchor(
                                st.session_state,
                                ring_num,
                                orig_x,
                                orig_y,
                                frame_idx=int(st.session_state["preview_frame_idx"]),
                                half_width=float(st.session_state.get(f"ring{ring_num}_r", 40)),
                            )
                        else:
                            mark_ring_configured(
                                st.session_state, ring_num, orig_x, orig_y,
                                frame_idx=_ring_anchor_frame_idx(),
                            )
                    ring_click_triggered_rerun = True
        else:
            st.image(preview_rgb, caption="Превью с зонами колец", use_container_width=True)
    else:
        st.error("Не удалось прочитать кадр из видео для превью.")

    ring1_anchor_count = len(ring_anchors_from_state(st.session_state, 1))
    ring2_anchor_count = len(ring_anchors_from_state(st.session_state, 2))
    status_cols = st.columns(2)
    with status_cols[0]:
        if st.session_state.get("ring1_configured"):
            if panning_mode:
                ring1_extra = f" · **задано {ring1_anchor_count} кадр.**"
            else:
                ring1_extra = " · **на всех кадрах**"
            st.success(
                f"✅ **Кольцо 1 задано:** X={int(st.session_state['ring1_x'])}, "
                f"линия Y={int(st.session_state['ring1_y'])}, полуширина={int(st.session_state['ring1_r'])} px"
                f"{ring1_extra}"
            )
        else:
            if panning_mode:
                st.caption("Кольцо 1: выберите кадр и кликните по ободу на превью.")
            else:
                st.caption("Кольцо 1: кликните по ободу на превью.")
    with status_cols[1]:
        if st.session_state.get("ring2_configured"):
            if panning_mode:
                ring2_extra = f" · **задано {ring2_anchor_count} кадр.**"
            else:
                ring2_extra = " · **на всех кадрах**"
            st.success(
                f"✅ **Кольцо 2 задано:** X={int(st.session_state['ring2_x'])}, "
                f"линия Y={int(st.session_state['ring2_y'])}, полуширина={int(st.session_state['ring2_r'])} px"
                f"{ring2_extra}"
            )
        else:
            if panning_mode:
                st.caption("Кольцо 2: выберите кадр и кликните по ободу на превью.")
            else:
                st.caption("Кольцо 2: кликните по ободу на превью.")

    if panning_mode:
        for ring_num, ring_label in ((1, "Кольцо 1"), (2, "Кольцо 2")):
            anchors = ring_anchors_from_state(st.session_state, ring_num)
            st.caption(f"**{ring_label}:** задано **{len(anchors)}** кадр.")
            if anchors:
                st.markdown(f"**Якоря {ring_label.lower()}**")
                for idx, anchor in enumerate(anchors):
                    row_cols = st.columns([5, 1])
                    with row_cols[0]:
                        st.text(
                            f"Кадр {int(anchor['frame'])}: "
                            f"X={int(anchor['x'])}, Y={int(anchor['y'])}, "
                            f"полуширина={int(anchor['half_width'])} px"
                        )
                    with row_cols[1]:
                        if st.button("✕", key=f"del_ring{ring_num}_anchor_{int(anchor['frame'])}_{idx}"):
                            remove_ring_anchor_at(st.session_state, ring_num, idx)
                            st.rerun()

    ball_anchor_count = len(normalize_ball_anchors(st.session_state.get("ball_anchors")))
    st.caption(
        f"Мяч: кликните минимум **{BALL_ANCHORS_MIN_RECOMMENDED}** кадра (задано **{ball_anchor_count}**)."
    )

    ball_anchors = normalize_ball_anchors(st.session_state.get("ball_anchors"))
    st.markdown("**Якоря мяча**")
    if not ball_anchors:
        st.caption("Пока нет — выберите режим «Мяч» и кликните по мячу на превью.")
    else:
        for idx, anchor in enumerate(ball_anchors):
            row_cols = st.columns([5, 1])
            with row_cols[0]:
                st.text(
                    f"Кадр {int(anchor['frame'])}: "
                    f"({int(anchor['x'])}, {int(anchor['y'])}) px"
                )
            with row_cols[1]:
                if st.button("✕", key=f"del_ball_anchor_{int(anchor['frame'])}_{idx}"):
                    remove_ball_anchor_at(st.session_state, idx)
                    st.rerun()

    if len(ball_anchors) >= BALL_ANCHORS_MIN_RECOMMENDED:
        skipped_interp = st.session_state.get("ball_interp_skipped") or []
        pending_interp = get_pending_ball_interp_checks(ball_anchors, skipped_interp)
        if pending_interp:
            check_frame = pending_interp[0]
            interp_pos = interpolate_ball_position(
                ball_anchors, check_frame, camera_transforms_preview if panning_mode else None
            )
            st.markdown("**Проверка интерполяции мяча**")
            st.caption(
                f"Между якорями предлагается проверить кадр **{check_frame}** "
                f"(осталось проверок: {len(pending_interp)}). Кнопка «Далее» не блокируется."
            )
            if interp_pos is not None:
                ix, iy, _ = interp_pos
                check_bgr = extract_frame_at_index(video_path, check_frame)
                if check_bgr is not None:
                    check_preview = draw_zones_preview(
                        check_bgr,
                        rings_for_preview_display(
                            st.session_state,
                            check_frame,
                            camera_transforms=camera_transforms_preview,
                            panning_mode=panning_mode,
                        ),
                        ball_anchors=ball_anchors,
                        preview_frame_idx=check_frame,
                        camera_transforms=camera_transforms_preview,
                        panning_mode=panning_mode,
                    )
                    draw_ball_interp_check_marker(check_preview, ix, iy)
                    st.image(
                        cv2.cvtColor(check_preview, cv2.COLOR_BGR2RGB),
                        caption=f"Кадр {check_frame}: предсказанная позиция мяча (интерполяция)",
                        use_container_width=True,
                    )
                btn_ok, btn_fix, btn_skip = st.columns(3)
                with btn_ok:
                    if st.button("✅ Верно", key=f"ball_interp_ok_{check_frame}"):
                        add_ball_anchor(st.session_state, int(round(ix)), int(round(iy)), check_frame)
                        if check_bgr is not None:
                            append_ball_training_seed(
                                video_path,
                                st.session_state.get("video_name"),
                                check_frame,
                                ix,
                                iy,
                                check_bgr,
                                source="interp_confirm",
                            )
                        st.rerun()
                with btn_fix:
                    if st.button("✏️ Поправить кликом", key=f"ball_interp_fix_{check_frame}"):
                        st.session_state["_pending_click_target_ring"] = "Мяч"
                        st.session_state["_pending_preview_frame_idx"] = int(check_frame)
                        st.session_state["ball_interp_fix_frame"] = int(check_frame)
                        ring_click_triggered_rerun = True
                with btn_skip:
                    if st.button("Пропустить", key=f"ball_interp_skip_{check_frame}"):
                        skipped = list(st.session_state.get("ball_interp_skipped") or [])
                        if check_frame not in skipped:
                            skipped.append(check_frame)
                        st.session_state["ball_interp_skipped"] = skipped
                        ring_click_triggered_rerun = True
            else:
                st.caption("Для этого кадра интерполяция недоступна — нажмите «Пропустить».")
                if st.button("Пропустить", key=f"ball_interp_skip_empty_{check_frame}"):
                    skipped = list(st.session_state.get("ball_interp_skipped") or [])
                    if check_frame not in skipped:
                        skipped.append(check_frame)
                    st.session_state["ball_interp_skipped"] = skipped
                    ring_click_triggered_rerun = True

        fix_frame = st.session_state.get("ball_interp_fix_frame")
        if fix_frame is not None:
            st.info(
                f"Режим правки: переключитесь на превью выше, выберите «Мяч» и кликните "
                f"по мячу на кадре **{fix_frame}**."
            )

    rings_ready = any_ring_configured(st.session_state)
    if not rings_ready:
        st.warning("Задайте **хотя бы одно кольцо** кликом по превью или числовыми полями, чтобы перейти дальше.")

    colA, colB = st.columns(2)
    with colA:
        if st.button("← Назад к загрузке"):
            go_to_step(1)
    with colB:
        if st.button("Далее → Сопоставление игроков", type="primary", disabled=not rings_ready):
            sync_ring_widgets_to_canonical(st.session_state, 1)
            sync_ring_widgets_to_canonical(st.session_state, 2)
            go_to_step(3)

    with st.expander(
        "⚙️ Профессиональный режим — пороги YOLO, трекинг, точные координаты",
        expanded=False,
    ):
        st.caption(
            "Открывайте только если на шагах 3–4 плохо считаются броски/передачи, теряется мяч "
            "или путаются ID игроков. Для настройки колец достаточно клика по превью выше."
        )
        render_step2_professional_settings(device, video_path, meta)

    # Отложенный rerun после клика по превью (см. комментарий у
    # ring_click_triggered_rerun в начале функции) — на этом этапе ВСЕ виджеты
    # текущего прогона (все 6 number_input, слайдеры, чекбокс, кнопки) уже
    # инстанциированы, поэтому rerun здесь больше не "осиротит" их состояние.
    if ring_click_triggered_rerun:
        st.rerun()


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
        "и по одному характерному кадру-кропу на каждого. **Имена и номера** впишите вручную "
        "по кропам. EasyOCR номеров **выключен по умолчанию** (шаг 2 → проф. режим) — "
        "включайте только если на вашем видео номера действительно читаются."
    )
    render_reid_status_warning()

    max_scan = max(int(min(meta["duration"], 60)), 5)
    if max_scan <= 5:
        # Очень короткое видео (≤5 сек) — слайдер с равными min/max упал бы с
        # ошибкой Streamlit, поэтому просто сканируем его целиком.
        scan_seconds = max_scan
        st.caption(f"Видео короткое — сканируется целиком ({scan_seconds} сек).")
    else:
        scan_seconds = st.slider(
            "Сколько секунд видео сканировать", min_value=5, max_value=max_scan,
            value=min(QUICK_SCAN_SECONDS_DEFAULT, max_scan), step=5,
        )

    if st.button("🔍 Запустить предварительное сканирование", type="primary"):
        st.session_state["manual_id_map"] = {}
        model, model_error = load_model(device)
        if model is None:
            st.warning(
                f"⚠️ Модель YOLO11x не загружена ({model_error}). Список игроков не может быть собран "
                "автоматически в этой среде — на рабочем ПК с RTX 4080 и интернетом для первого "
                "скачивания весов сканирование сработает штатно. Этот шаг можно пропустить: имена по "
                "умолчанию останутся как «Игрок {ID}»."
            )
        else:
            scan_progress = st.progress(0.0, text="Сканирование игроков...")
            scan_status = st.empty()
            crops, merge_log = quick_player_scan(
                video_path, model, device, max_seconds=scan_seconds, fps_hint=meta["fps"],
                enhance_quality=st.session_state.get("enhance_quality", False),
                person_conf=float(st.session_state.get("person_conf", PERSON_CONF_DEFAULT)),
                ball_conf=float(st.session_state.get("ball_conf", BALL_CONF_DEFAULT)),
                imgsz=int(st.session_state.get("imgsz", IMGSZ_DEFAULT)),
                appearance_similarity=float(
                    st.session_state.get("appearance_similarity", APPEARANCE_SIMILARITY_DEFAULT)
                ),
                jersey_ocr_enabled=bool(st.session_state.get("jersey_ocr_enabled", JERSEY_OCR_ENABLED_DEFAULT)),
                min_person_bbox_area=float(st.session_state.get("min_person_bbox_area", MIN_PERSON_BBOX_AREA_DEFAULT)),
                progress_callback=scan_progress.progress,
                status_callback=scan_status.caption,
            )
            scan_progress.progress(1.0, text="Сканирование завершено")
            scan_status.empty()
            st.session_state["player_crops"] = crops
            st.session_state["id_merge_log"] = merge_log
            if not crops:
                st.warning("Игроки не найдены за это время — попробуйте увеличить длительность сканирования.")
            else:
                st.success(f"Найдено {len(crops)} уникальных ID игроков.")

    consume_pending_step3_ui_state(st.session_state)

    crops: Dict[int, Any] = st.session_state.get("player_crops") or {}
    manual_map: Dict[int, int] = dict(st.session_state.get("manual_id_map") or {})
    if crops:
        st.subheader("🔗 Склейка ID — один человек")
        st.caption(
            "Отметьте **2+ карточки** чекбоксом «В группу» и нажмите **Объединить выбранных** — "
            "каноническим станет наименьший ID. Группы показаны ниже; у группы можно нажать **Разъединить**."
        )

        groups = build_player_id_groups(sorted(crops.keys()), manual_map)
        for canonical in sorted(groups.keys()):
            members = groups[canonical]
            if len(members) > 1:
                with st.container(border=True):
                    header_cols = st.columns([4, 1])
                    with header_cols[0]:
                        st.markdown(f"**Группа — канонический ID {canonical}** · это один человек")
                    with header_cols[1]:
                        if st.button("Разъединить", key=f"unmerge_group_{canonical}"):
                            unmerge_player_group(st.session_state, canonical)
                            st.rerun()
                    main_col, mini_cols = st.columns([2, 3])
                    with main_col:
                        resized_main = resize_crop_to_height(crops[canonical], CROP_DISPLAY_HEIGHT)
                        st.image(
                            cv2.cvtColor(resized_main, cv2.COLOR_BGR2RGB),
                            caption=f"ID {canonical} (канон)",
                        )
                        st.checkbox(
                            "Исключить из статистики",
                            key=f"exclude_player_{canonical}",
                            value=canonical in (st.session_state.get("excluded_player_ids") or []),
                        )
                        st.text_input(
                            "Имя", key=f"player_name_{canonical}",
                            placeholder=f"Игрок {canonical}", label_visibility="collapsed",
                        )
                        st.text_input(
                            "Номер", key=f"player_number_{canonical}",
                            placeholder="Номер", label_visibility="collapsed",
                        )
                    with mini_cols:
                        st.caption("Склеенные ID:")
                        satellite = [m for m in members if m != canonical]
                        sat_cols = st.columns(min(len(satellite), 4) or 1)
                        for col, sid in zip(sat_cols, satellite):
                            with col:
                                mini = resize_crop_to_height(crops[sid], max(CROP_DISPLAY_HEIGHT // 2, 72))
                                st.image(cv2.cvtColor(mini, cv2.COLOR_BGR2RGB), caption=f"б. ID {sid}")
            else:
                pid = members[0]
                with st.container(border=True):
                    row_cols = st.columns([1, 4])
                    with row_cols[0]:
                        st.checkbox("В группу", key=f"merge_pick_{pid}")
                    with row_cols[1]:
                        card_cols = st.columns([2, 3])
                        with card_cols[0]:
                            resized_crop = resize_crop_to_height(crops[pid], CROP_DISPLAY_HEIGHT)
                            st.image(cv2.cvtColor(resized_crop, cv2.COLOR_BGR2RGB), caption=f"ID {pid}")
                        with card_cols[1]:
                            st.checkbox(
                                "Исключить из статистики",
                                key=f"exclude_player_{pid}",
                                value=pid in (st.session_state.get("excluded_player_ids") or []),
                            )
                            st.text_input(
                                "Имя", key=f"player_name_{pid}",
                                placeholder=f"Игрок {pid}", label_visibility="collapsed",
                            )
                            st.text_input(
                                "Номер", key=f"player_number_{pid}",
                                placeholder="Номер", label_visibility="collapsed",
                            )

        merge_pick_ids = [
            int(pid) for pid in sorted(crops.keys())
            if st.session_state.get(f"merge_pick_{pid}", False)
        ]
        merge_btn_cols = st.columns([2, 3])
        with merge_btn_cols[0]:
            if st.button(
                "Объединить выбранных",
                type="secondary",
                disabled=len(merge_pick_ids) < 2,
                key="merge_selected_players_btn",
            ):
                merge_player_ids_selection(st.session_state, merge_pick_ids)
                st.session_state["_pending_merge_pick_clear"] = list(merge_pick_ids)
                st.rerun()
        with merge_btn_cols[1]:
            if merge_pick_ids:
                st.caption(f"Выбрано для склейки: **{', '.join(str(p) for p in sorted(merge_pick_ids))}**")
    else:
        st.info(
            "Пока нет данных — запустите сканирование выше, либо пропустите этот шаг: "
            "в итоговой таблице игроки будут отображаться как «Игрок {ID}»."
        )

    merge_log = st.session_state.get("id_merge_log") or []
    if merge_log:
        with st.expander(f"🔗 Склейки ID по внешнему виду ({len(merge_log)})", expanded=False):
            st.dataframe(pd.DataFrame(merge_log), use_container_width=True, hide_index=True)
            st.caption("Новые ID трекера, переназначенные на ранее виденный ID (appearance-matching).")

    manual_merge_log = st.session_state.get("manual_id_merge_log") or []
    if manual_map:
        with st.expander(f"Таблица ручных склеек ({len(manual_map)})", expanded=False):
            rows = [
                {"Бывший ID": raw_id, "→ Канонический": target}
                for raw_id, target in sorted(manual_map.items(), key=lambda item: (item[1], item[0]))
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    if manual_merge_log:
        with st.expander(f"История ручных склеек ({len(manual_merge_log)})", expanded=False):
            st.dataframe(pd.DataFrame(manual_merge_log), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("🏀 Диагностика видимости мяча")
    st.caption(
        "Если на шаге 4 фиксируется 0 бросков/передач — сначала проверьте здесь, вообще ли модель "
        "видит мяч на этом видео при текущем пороге уверенности (настраивается на шаге 2), прежде "
        "чем менять пороги владения/передач."
    )
    if st.button("🔍 Проверить видимость мяча"):
        model, model_error = load_model(device)
        if model is None:
            st.warning(f"⚠️ Модель YOLO11x не загружена ({model_error}) — диагностика недоступна в этой среде.")
        else:
            diag_progress = st.progress(0.0, text="Диагностика видимости мяча...")
            diag_status = st.empty()
            diag = diagnose_ball_visibility(
                video_path, model, device,
                ball_conf_threshold=float(st.session_state.get("ball_conf", BALL_CONF_DEFAULT)),
                imgsz=int(st.session_state.get("imgsz", IMGSZ_DEFAULT)),
                enhance_quality=st.session_state.get("enhance_quality", False),
                max_gap_frames=int(st.session_state.get("ball_max_gap_frames", BALL_MAX_GAP_FRAMES_DEFAULT)),
                max_predict_frames=int(st.session_state.get("ball_max_predict_frames", BALL_MAX_PREDICT_FRAMES_DEFAULT)),
                color_fallback=bool(st.session_state.get("ball_color_fallback", BALL_COLOR_FALLBACK_DEFAULT)),
                color_roi_half=int(st.session_state.get("ball_color_roi_half", BALL_COLOR_ROI_HALF_DEFAULT)),
                progress_callback=diag_progress.progress,
                status_callback=diag_status.caption,
            )
            diag_progress.progress(1.0, text="Диагностика завершена")
            diag_status.empty()
            st.session_state["ball_diagnostics"] = diag
            if diag is None:
                st.error("Не удалось выполнить диагностику (не открылось видео или сбой модели).")

    diag = st.session_state.get("ball_diagnostics")
    if diag:
        yolo_pct = diag.get("yolo_detection_rate", diag.get("detection_rate", 0.0)) * 100.0
        enh_pct = diag.get("enhanced_detection_rate", yolo_pct / 100.0) * 100.0
        src = diag.get("source_counts") or {}
        st.caption(
            f"Сэмплов по видео: {diag['sampled_frames']}"
            + (f" из ~{diag.get('total_video_frames', '?')} кадров" if diag.get("total_video_frames") else "")
            + " · "
            f"YOLO: {diag.get('frames_with_ball_yolo', diag.get('frames_with_ball', 0))} "
            f"({yolo_pct:.0f}%, conf≈{diag['avg_confidence']:.2f}) · "
            f"после Kalman/цвет: {diag.get('frames_with_ball_enhanced', 0)} ({enh_pct:.0f}%) · "
            f"средняя дыра YOLO: {diag.get('avg_gap_frames', 0):.1f} кадр., "
            f"макс. {diag.get('max_gap_frames', 0)}"
        )
        if src:
            st.caption(
                "Источники (после fallback): "
                + ", ".join(f"{k}={v}" for k, v in sorted(src.items()) if v > 0)
            )
        if enh_pct >= 50:
            st.success(
                f"✅ С fallback мяч виден в {enh_pct:.0f}% кадров (YOLO alone: {yolo_pct:.0f}%). "
                "События должны фиксироваться лучше — проверьте полный анализ на шаге 4."
            )
        elif enh_pct >= yolo_pct + 10:
            st.warning(
                f"⚠️ YOLO видит мяч в {yolo_pct:.0f}% кадров, fallback поднимает до {enh_pct:.0f}%. "
                "Попробуйте увеличить ROI цвета или окно виртуального мяча на шаге 2."
            )
        elif yolo_pct >= 20:
            st.warning(
                f"⚠️ YOLO: {yolo_pct:.0f}%, с fallback: {enh_pct:.0f}%. "
                "Увеличьте imgsz, включите цветовой fallback и окно Kalman на шаге 2."
            )
        else:
            st.error(
                f"❌ YOLO: {yolo_pct:.0f}%, с fallback: {enh_pct:.0f}% — даже эвристики слабо помогают. "
                "Проверьте освещение/качество видео или увеличьте ROI и окно виртуального мяча."
            )

    colA, colB = st.columns(2)
    with colA:
        if st.button("← Назад к зонам и порогам"):
            go_to_step(2)
    with colB:
        if st.button("Далее → Полный анализ", type="primary"):
            excluded: List[int] = []
            for pid in crops.keys():
                st.session_state["player_names"][pid] = st.session_state.get(f"player_name_{pid}", "")
                st.session_state["player_numbers"][pid] = st.session_state.get(f"player_number_{pid}", "")
                if st.session_state.get(f"exclude_player_{pid}", False):
                    excluded.append(pid)
            st.session_state["excluded_player_ids"] = excluded
            go_to_step(4)


# ---------------------------------------------------------------------------
# Шаг 4 — полный прогон и итоговая статистика
# ---------------------------------------------------------------------------
def run_full_analysis(video_path: str, device: str) -> None:
    ensure_directories()
    ensure_tracker_config()
    clear_highlights_directory()
    reset_analysis_results()
    ensure_ring_zones_for_video(video_path, st.session_state)
    sync_ring_widgets_to_canonical(st.session_state, 1)
    sync_ring_widgets_to_canonical(st.session_state, 2)

    if not any_ring_configured(st.session_state):
        st.error(
            "Кольца не заданы: ни ring1_configured, ни ring2_configured не установлены в True. "
            f"Текущее состояние: {describe_ring_state_for_debug(st.session_state)}"
        )
        if st.button("← Вернуться к шагу 2 (зоны и пороги)", type="primary"):
            go_to_step(2)
        return

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

    video_meta = get_video_metadata(video_path)
    rings = rings_from_session_state(st.session_state)
    prepared_rings, ring_preview_warnings = prepare_rings_for_drawing(
        rings, int(video_meta["width"]), int(video_meta["height"])
    )
    if not prepared_rings:
        st.error(
            "Не удалось подготовить линии колец для отрисовки. "
            f"Состояние: {describe_ring_state_for_debug(st.session_state)}"
        )
        if st.button("← Вернуться к шагу 2", key="back_step2_ring_error"):
            go_to_step(2)
        return
    for warn in ring_preview_warnings:
        st.warning(f"⚠️ {warn}")

    camera_transforms: Optional[List[np.ndarray]] = None
    if st.session_state.get("camera_mode") == CAMERA_MODE_PANNING and cv2 is not None:
        if st.session_state.get("camera_transforms_video") == video_path and st.session_state.get("camera_transforms"):
            camera_transforms = st.session_state["camera_transforms"]
            st.info("🎥 Использую оценку движения камеры с шага 2 (якорный кадр каждого кольца).")
        else:
            st.info("🎥 Экспериментальный режим «камера в движении»: оцениваю сдвиг камеры по всему видео...")
            cam_progress = st.progress(0.0)
            cam_status = st.empty()
            video_meta_cam = get_video_metadata(video_path)
            total_frames_cam = max(int(video_meta_cam["total_frames"]), 1)

            def _camera_run_progress(current: int, total: int) -> None:
                cam_progress.progress(min(current / max(total, 1), 1.0))
                cam_status.caption(f"Оценка движения камеры: кадр **{current}** / **{total}**")

            try:
                camera_transforms = estimate_camera_transforms(
                    video_path, frame_progress_callback=_camera_run_progress
                )
                st.session_state["camera_transforms"] = camera_transforms
                st.session_state["camera_transforms_video"] = video_path
            except Exception as exc:
                st.warning(f"⚠️ Не удалось оценить движение камеры ({exc}) — зоны колец останутся фиксированными.")
                camera_transforms = None
            cam_progress.progress(1.0)
            cam_status.caption(f"Оценка движения камеры завершена ({total_frames_cam} кадров).")

    progress_bar = st.progress(0.0)
    status_text = st.empty()

    try:
        stats, output_path, debug_log = process_video(
            video_path=video_path,
            model=model,
            device=device,
            rings=prepared_rings,
            possession_threshold=float(st.session_state["possession_threshold"]),
            pass_min_seconds=float(st.session_state.get("pass_min_seconds", PASS_MIN_SECONDS_DEFAULT)),
            pass_max_seconds=float(st.session_state.get("pass_max_seconds", PASS_MAX_SECONDS_DEFAULT)),
            goal_cooldown_seconds=float(
                st.session_state.get("goal_cooldown_seconds", GOAL_COOLDOWN_SECONDS_DEFAULT)
            ),
            avg_player_diagonal=st.session_state.get("avg_player_diagonal"),
            ball_memory_seconds=float(st.session_state["ball_memory"]),
            progress_bar=progress_bar,
            status_text=status_text,
            enhance_quality=bool(st.session_state.get("enhance_quality", False)),
            person_conf=float(st.session_state.get("person_conf", PERSON_CONF_DEFAULT)),
            ball_conf=float(st.session_state.get("ball_conf", BALL_CONF_DEFAULT)),
            imgsz=int(st.session_state.get("imgsz", IMGSZ_DEFAULT)),
            max_gap_frames=int(st.session_state.get("ball_max_gap_frames", BALL_MAX_GAP_FRAMES_DEFAULT)),
            max_predict_frames=int(st.session_state.get("ball_max_predict_frames", BALL_MAX_PREDICT_FRAMES_DEFAULT)),
            color_fallback=bool(st.session_state.get("ball_color_fallback", BALL_COLOR_FALLBACK_DEFAULT)),
            color_roi_half=int(st.session_state.get("ball_color_roi_half", BALL_COLOR_ROI_HALF_DEFAULT)),
            appearance_similarity=float(
                st.session_state.get("appearance_similarity", APPEARANCE_SIMILARITY_DEFAULT)
            ),
            jersey_ocr_enabled=bool(st.session_state.get("jersey_ocr_enabled", JERSEY_OCR_ENABLED_DEFAULT)),
            min_person_bbox_area=float(st.session_state.get("min_person_bbox_area", MIN_PERSON_BBOX_AREA_DEFAULT)),
            excluded_player_ids=set(st.session_state.get("excluded_player_ids") or []),
            camera_transforms=camera_transforms,
            ball_anchors=st.session_state.get("ball_anchors") or [],
            manual_id_map=dict(st.session_state.get("manual_id_map") or {}),
        )
    except Exception as exc:
        st.error(f"Ошибка при обработке видео: {exc}")
        return

    status_text.text("Обработка завершена ✅")
    st.session_state["box_score_df"] = build_box_score(
        stats,
        st.session_state.get("player_names"),
        st.session_state.get("player_numbers"),
        excluded_player_ids=set(st.session_state.get("excluded_player_ids") or []),
        manual_id_map=dict(st.session_state.get("manual_id_map") or {}),
    )
    st.session_state["last_output_video"] = str(output_path)
    st.session_state["debug_log"] = debug_log
    st.session_state["highlight_files"] = [
        str(p) for p in sorted(HIGHLIGHTS_DIR.glob("*.mp4"), key=lambda p: p.stat().st_mtime)
    ]
    st.success(f"Готово! Полное аннотированное видео сохранено: {output_path}")


def render_video_with_fallback(video_path: Path, label: str, widget_key: str) -> None:
    """Показывает видео через st.video() с понятной обработкой отказа.

    st.video() рендерит обычный HTML5 <video> — если файл закодирован
    кодеком, который браузер пользователя не поддерживает (см.
    reencode_for_browser), плеер в браузере выглядит "битым", хотя сам файл
    валиден. Т.к. это происходит на стороне браузера, сервер не может это
    детектировать программно — поэтому ниже ВСЕГДА дополнительно показывается
    кнопка скачивания файла как гарантированный запасной вариант.
    """
    if not video_path.exists() or video_path.stat().st_size == 0:
        st.error(f"⚠️ Файл «{label}» не найден или пуст: {video_path}")
        return

    st.video(str(video_path))

    if FFMPEG_PATH is None:
        st.caption(
            "ℹ️ ffmpeg не найден в системе — видео сохранено в кодеке OpenCV по умолчанию, который "
            "не во всех браузерах проигрывается через встроенный плеер. Установите ffmpeg "
            "(`sudo apt install ffmpeg` / `brew install ffmpeg` / `winget install ffmpeg`) для "
            "автоматической перекодировки в H.264, либо воспользуйтесь кнопкой скачивания ниже."
        )

    try:
        with open(video_path, "rb") as f:
            data = f.read()
        st.download_button(
            f"⬇️ Скачать «{label}»", data=data, file_name=video_path.name, mime="video/mp4", key=widget_key
        )
    except OSError as exc:
        st.error(f"Не удалось прочитать файл для скачивания: {exc}")


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

            ring_warns = debug_log.get("ring_draw_warnings") or []
            if ring_warns:
                st.warning("Линии колец: " + " · ".join(ring_warns))

            sampled = debug_log.get("sampled_timeline") or []
            if sampled:
                st.caption("Владелец мяча (ID) во времени — разрывы означают, что мяч в этот момент ничей/не найден.")
                sampled_df = pd.DataFrame(sampled).set_index("Время, с")
                st.line_chart(sampled_df)

    last_output_video = st.session_state.get("last_output_video")
    if last_output_video and Path(last_output_video).exists():
        with st.expander("🎥 Полное видео с аннотациями", expanded=False):
            render_video_with_fallback(Path(last_output_video), "полное аннотированное видео", "dl_full_output")

    st.header("🎬 Хайлайты (броски и передачи)")
    highlight_paths = st.session_state.get("highlight_files") or []
    if not highlight_paths:
        st.info("Нарезанные хайлайты появятся здесь после обнаружения бросков/передач в обработанном видео.")
        return

    for highlight_path in reversed(highlight_paths):
        hf = Path(highlight_path)
        if not hf.exists():
            continue
        st.subheader(hf.name)
        render_video_with_fallback(hf, hf.name, f"dl_{hf.stem}")


def render_step4_run(device: str) -> None:
    st.header("Шаг 4 — Полный анализ и статистика")
    video_path = st.session_state.get("video_path")
    if not video_path or not Path(video_path).exists():
        st.warning("Сначала загрузите видео на шаге 1.")
        if st.button("← Назад к загрузке"):
            go_to_step(1)
        return

    render_reid_status_warning()

    with st.expander("⚙️ Текущие настройки анализа", expanded=False):
        r1_n = len(ring_anchors_from_state(st.session_state, 1))
        r2_n = len(ring_anchors_from_state(st.session_state, 2))
        st.write(
            f"Кольцо 1: X={int(st.session_state['ring1_x'])}, Y={int(st.session_state['ring1_y'])}, "
            f"полуширина={int(st.session_state['ring1_r'])}, якорей {r1_n} · "
            f"Кольцо 2: X={int(st.session_state['ring2_x'])}, Y={int(st.session_state['ring2_y'])}, "
            f"полуширина={int(st.session_state['ring2_r'])}, якорей {r2_n}"
        )
        st.write(
            f"Порог владения: {int(st.session_state['possession_threshold'])} px · "
            f"Окно паса: {float(st.session_state.get('pass_min_seconds', PASS_MIN_SECONDS_DEFAULT)):.2f}–"
            f"{float(st.session_state.get('pass_max_seconds', PASS_MAX_SECONDS_DEFAULT)):.2f} с · "
            f"Память мяча: {st.session_state['ball_memory']:.2f} с · "
            f"Кулдаун гола: {float(st.session_state.get('goal_cooldown_seconds', GOAL_COOLDOWN_SECONDS_DEFAULT)):.2f} с"
        )
        st.write(
            "Улучшение качества кадра перед детекцией: "
            + ("включено ✅" if st.session_state.get("enhance_quality") else "выключено")
        )
        st.write(
            f"Порог уверенности: мяч {float(st.session_state.get('ball_conf', BALL_CONF_DEFAULT)):.2f} · "
            f"игроки {float(st.session_state.get('person_conf', PERSON_CONF_DEFAULT)):.2f} · "
            f"imgsz {int(st.session_state.get('imgsz', IMGSZ_DEFAULT))}px"
        )
        st.write(
            "Трекинг мяча: "
            f"интерполяция до {int(st.session_state.get('ball_max_gap_frames', BALL_MAX_GAP_FRAMES_DEFAULT))} кадр., "
            f"виртуальный мяч до {int(st.session_state.get('ball_max_predict_frames', BALL_MAX_PREDICT_FRAMES_DEFAULT))} кадр., "
            f"цветовой fallback {'вкл' if st.session_state.get('ball_color_fallback', BALL_COLOR_FALLBACK_DEFAULT) else 'выкл'}, "
            f"ROI {int(st.session_state.get('ball_color_roi_half', BALL_COLOR_ROI_HALF_DEFAULT))}px"
        )
        st.write(
            "Режим камеры: "
            + CAMERA_MODE_LABELS.get(st.session_state.get("camera_mode", CAMERA_MODE_STATIC), "статичная")
        )
        diag = st.session_state.get("ball_diagnostics")
        if diag:
            yolo_r = diag.get("yolo_detection_rate", diag.get("detection_rate", 0.0)) * 100
            enh_r = diag.get("enhanced_detection_rate", 0.0) * 100
            st.write(
                f"Диагностика мяча: YOLO {yolo_r:.0f}% · с fallback {enh_r:.0f}% · "
                f"средняя дыра {diag.get('avg_gap_frames', 0):.1f} кадр."
            )
        n_players = len(st.session_state.get("player_names") or {})
        n_excl = len(st.session_state.get("excluded_player_ids") or [])
        st.write(
            f"Сопоставлено игроков (имя/номер): {n_players}"
            + (f" · исключено из статистики: {n_excl}" if n_excl else "")
        )
        st.write(
            f"Порог похожести игроков (склейка ID): "
            f"{float(st.session_state.get('appearance_similarity', APPEARANCE_SIMILARITY_DEFAULT)):.2f}"
        )

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
            clear_highlights_directory()
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
    st.caption("Офлайн-аналитика баскетбольных тренировок по видео (YOLO11x + ByteTrack)")

    device, device_message, device_ok = resolve_device()
    if device_ok:
        st.success(device_message)
    else:
        st.warning(device_message)

    with st.sidebar:
        st.markdown("## 🏀 Basketball Tracking")
        st.caption("Офлайн-аналитика тренировок")
        if device_ok:
            st.success("CUDA доступна")
        else:
            st.warning("Режим CPU / без GPU")
        reid_error = get_reid_error_message()
        if reid_error:
            st.warning(f"ReID недоступен: {reid_error}")

    step = st.session_state["step"]
    render_step_indicator(step)

    if step == 1:
        render_step1_upload()
    elif step == 2:
        render_step2_zones(device)
    elif step == 3:
        render_step3_players(device)
    else:
        render_step4_run(device)


if __name__ == "__main__":
    if not test_draw_hoop_lines_on_frame():
        raise RuntimeError("test_draw_hoop_lines_on_frame: пиксели линии кольца остались нулевыми")
    main()

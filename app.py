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
      ByteTrack (игроки без номеров на майках — удержание ID за счёт трекера,
      а не OCR номеров).
    * Векторная детекция голов: пересечение траектории мяча с горизонтальной
      линией кольца (задаётся пользователем), с кулдауном 90 кадров.
    * Детекция передач (пасов): игрок владел мячом → мяч летел без владельца
      10–60 кадров → другой игрок получил владение, с "памятью" мяча при
      кратковременной потере детекции.
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

import math
import os
import shutil
import subprocess
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
BASE_DIR = Path(__file__).resolve().parent
HIGHLIGHTS_DIR = BASE_DIR / "highlights"
OUTPUT_DIR = BASE_DIR / "output_videos"
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

# Окно передачи в КАДРАХ: мяч должен лететь без владельца от min до max кадров.
PASS_MIN_FRAMES_DEFAULT = 10
PASS_MIN_FRAMES_MIN = 3
PASS_MIN_FRAMES_MAX = 30

PASS_MAX_FRAMES_DEFAULT = 60
PASS_MAX_FRAMES_MIN = 20
PASS_MAX_FRAMES_MAX = 120

# Кулдаун между голами у одного кольца (кадры; ~3 сек при 30 fps).
GOAL_COOLDOWN_FRAMES_DEFAULT = 90
GOAL_COOLDOWN_FRAMES_MIN = 30
GOAL_COOLDOWN_FRAMES_MAX = 180

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

# Сколько кадров сэмплировать по всему видео при диагностике видимости мяча
# (шаг 3) — не весь ролик, чтобы диагностика оставалась быстрой (секунды, а
# не минуты), но достаточно равномерно распределённых кадров для честной
# оценки % обнаружения и средней уверенности.
BALL_DIAGNOSTIC_MAX_SAMPLES = 120

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


def ensure_tracker_config(path: Path = TRACKER_CONFIG_PATH) -> Path:
    """Генерирует локальный конфиг трекера ByteTrack при первом запуске.

    ByteTrack удерживает ID игроков без ReID/номеров на форме: track_buffer=180
    (~5–6 сек при 30 fps) помогает не терять трек при окклюзии, а
    track_low_thresh=0.1 не отбрасывает частично перекрытые силуэты.
    """
    if path.exists():
        return path

    tracker_cfg = {
        "tracker_type": "bytetrack",
        "track_high_thresh": 0.25,
        "track_low_thresh": 0.1,
        "new_track_thresh": 0.25,
        "track_buffer": 180,
        "match_thresh": 0.8,
        "fuse_score": True,
    }
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            "# Автоматически сгенерированный конфиг трекера ByteTrack.\n"
            "# track_buffer=180 — помнить игрока ~5–6 сек при окклюзии.\n"
            "# track_low_thresh=0.1 — не терять трек при частичном перекрытии.\n"
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
        model.predictor = None  # заставит ultralytics создать трекер заново
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Геометрия и вспомогательные вычисления
# ---------------------------------------------------------------------------
def bbox_center(box: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


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
def estimate_camera_transforms(
    video_path: str, progress_callback: Optional[Any] = None
) -> List[np.ndarray]:
    """Оценивает покадровый сдвиг камеры методом оптического потока Лукаса-Канаде
    по устойчивым фоновым фичам (линии площадки, трибуны, стены и т.п.).

    Возвращает список накопленных 3x3 матриц гомографии длиной в число кадров
    видео, где transforms[i] переводит точку из системы координат КАДРА 0 в
    систему координат КАДРА i. transforms[0] всегда единичная матрица.

    Это ЭКСПЕРИМЕНТАЛЬНАЯ оценка: RANSAC внутри estimateAffinePartial2D
    достаточно устойчив к перемещающимся игрокам (они занимают меньшую часть
    кадра, чем статичный фон), но метод рассчитан на ПЛАВНУЮ панораму/наклон
    камеры. Резкий зум, сильная тряска или смена плана (склейка) могут сбить
    накопленную оценку — поэтому режим явно помечен как экспериментальный
    в GUI, и статичный режим (весь фон предположения) остаётся дефолтным и
    полностью независимым путём кода.
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

    feature_params = dict(maxCorners=250, qualityLevel=0.01, minDistance=12, blockSize=7)
    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        incremental = np.eye(3, dtype=np.float64)
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
        if prev_pts is not None and len(prev_pts) >= 6:
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, **lk_params)
            if next_pts is not None and status is not None:
                status_flat = status.flatten() == 1
                good_prev = prev_pts[status_flat]
                good_next = next_pts[status_flat]
                if len(good_prev) >= 6:
                    m, _inliers = cv2.estimateAffinePartial2D(
                        good_prev, good_next, method=cv2.RANSAC, ransacReprojThreshold=3.0
                    )
                    if m is not None:
                        incremental[:2, :] = m

        cumulative.append(incremental @ cumulative[-1])
        prev_gray = gray

        if progress_callback is not None and len(cumulative) % 15 == 0:
            try:
                progress_callback(min(len(cumulative) / total_frames, 1.0))
            except Exception:
                pass

    cap.release()
    if progress_callback is not None:
        try:
            progress_callback(1.0)
        except Exception:
            pass
    return cumulative


def compute_dynamic_rings(
    rings: List[RingZone],
    camera_transforms: List[np.ndarray],
    ref_frame_idx: int,
    frame_idx: int,
) -> List[RingZone]:
    """Пересчитывает позиции (и радиус, с учётом лёгкого зума) зон колец из
    системы координат кадра, на котором пользователь их задал (ref_frame_idx),
    в систему координат текущего кадра (frame_idx), используя накопленные
    трансформации камеры из estimate_camera_transforms."""
    if not camera_transforms:
        return rings
    n = len(camera_transforms)
    ref_idx = int(np.clip(ref_frame_idx, 0, n - 1))
    cur_idx = int(np.clip(frame_idx, 0, n - 1))
    try:
        ref_inv = np.linalg.inv(camera_transforms[ref_idx])
    except np.linalg.LinAlgError:
        return rings
    transform = camera_transforms[cur_idx] @ ref_inv
    scale = math.hypot(transform[0, 0], transform[1, 0]) or 1.0

    dynamic_rings: List[RingZone] = []
    for ring in rings:
        point = transform @ np.array([ring["x"], ring["y"], 1.0], dtype=np.float64)
        half_w = ring.get("half_width", ring.get("r", 40.0))
        dynamic_rings.append(
            {"x": float(point[0]), "y": float(point[1]), "half_width": float(half_w * scale)}
        )
    return dynamic_rings


# ---------------------------------------------------------------------------
# Разбор результатов детекции/трекинга Ultralytics и отрисовка аннотаций
# ---------------------------------------------------------------------------
def parse_track_results(
    results,
    person_conf_threshold: float = 0.0,
    ball_conf_threshold: float = 0.0,
) -> Tuple[List[Tuple[int, Tuple[float, float, float, float]]], Optional[Tuple[float, float]]]:
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

    boxes = result.boxes
    if boxes is not None and boxes.id is not None and len(boxes) > 0:
        xyxy = boxes.xyxy.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        ids = boxes.id.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()

        best_ball_conf = -1.0
        for box, c, tid, conf in zip(xyxy, cls, ids, confs):
            conf = float(conf)
            if c == COCO_PERSON_CLASS_ID:
                if conf < person_conf_threshold:
                    continue
                persons.append((int(tid), (float(box[0]), float(box[1]), float(box[2]), float(box[3]))))
            elif c == COCO_BALL_CLASS_ID:
                if conf < ball_conf_threshold:
                    continue
                if conf > best_ball_conf:
                    best_ball_conf = conf
                    ball = ((float(box[0]) + float(box[2])) / 2.0, (float(box[1]) + float(box[3])) / 2.0)

    return persons, ball


def id_to_color(pid: int) -> Tuple[int, int, int]:
    """Детерминированный BGR-цвет по ID трека — чтобы игроки визуально
    отличались друг от друга на аннотированном видео."""
    hue = (int(pid) * 47) % 180
    hsv_pixel = np.uint8([[[hue, 220, 255]]])
    bgr_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0][0]
    return int(bgr_pixel[0]), int(bgr_pixel[1]), int(bgr_pixel[2])


def draw_annotations(
    frame,
    persons: List[Tuple[int, Tuple[float, float, float, float]]],
    ball: Optional[Tuple[float, float]],
    ball_trajectory: Optional[List[Tuple[float, float]]] = None,
):
    """Рисует рамки игроков (с ID), траекторию мяча и маркер мяча на копии кадра.

    Заменяет result.plot() из ultralytics: при включённом "улучшении
    качества" детекция идёт на увеличенном кадре, а рисовать нужно на
    ОРИГИНАЛЬНОМ (после пересчёта координат обратно) — иначе выходное видео
    получилось бы в другом разрешении, чем входное.
    """
    annotated = frame.copy()
    if ball_trajectory and len(ball_trajectory) >= 2:
        pts = [(int(x), int(y)) for x, y in ball_trajectory]
        for i in range(1, len(pts)):
            cv2.line(annotated, pts[i - 1], pts[i], (0, 255, 255), 3, cv2.LINE_AA)
    for pid, (x1, y1, x2, y2) in persons:
        color = id_to_color(pid)
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(annotated, p1, p2, color, 2)
        label = f"ID {pid}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        label_y1 = max(p1[1] - th - 8, 0)
        cv2.rectangle(annotated, (p1[0], label_y1), (p1[0] + tw + 6, p1[1]), color, -1)
        cv2.putText(
            annotated, label, (p1[0] + 3, max(p1[1] - 5, th)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
        )
    if ball is not None:
        center = (int(ball[0]), int(ball[1]))
        cv2.circle(annotated, center, 9, (0, 215, 255), -1)
        cv2.circle(annotated, center, 9, (0, 0, 0), 2)
        cv2.putText(
            annotated, "ball", (center[0] + 12, center[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 215, 255), 2, cv2.LINE_AA,
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
    """Разумные дефолтные горизонтальные линии двух колец (по краям площадки)."""
    half_w = max(int(min(width, height) * 0.08), 40)
    ring1 = {"x": float(int(width * 0.10)), "y": float(int(height * 0.35)), "half_width": float(half_w)}
    ring2 = {"x": float(int(width * 0.90)), "y": float(int(height * 0.35)), "half_width": float(half_w)}
    return ring1, ring2


def draw_zones_preview(
    frame,
    rings: List[RingZone],
    possession_threshold: Optional[float] = None,
) -> Any:
    """Рисует горизонтальные линии колец на копии кадра для наглядной проверки в GUI.

    Если передан possession_threshold — дополнительно рисует в углу кадра
    эталонный полупрозрачный круг такого радиуса с подписью в пикселях, чтобы
    пользователь видел порог владения мячом в реальном масштабе кадра, а не
    гадал по числу пикселей.
    """
    preview = frame.copy()
    h, w = preview.shape[:2]
    colors = [(0, 140, 255), (255, 80, 0)]
    for idx, ring in enumerate(rings):
        color = colors[idx % len(colors)]
        center_x = int(ring["x"])
        line_y = int(ring["y"])
        half_w = max(int(ring.get("half_width", ring.get("r", 40))), 1)
        x1 = max(center_x - half_w, 0)
        x2 = min(center_x + half_w, w - 1)
        cv2.line(preview, (x1, line_y), (x2, line_y), color, 3, cv2.LINE_AA)
        cv2.drawMarker(
            preview, (center_x, line_y), color, markerType=cv2.MARKER_CROSS, markerSize=14, thickness=2
        )
        label_pos = (max(center_x - 45, 0), max(line_y - 18, 20))
        cv2.putText(preview, f"Кольцо {idx + 1}", label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

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
        cv2.putText(
            preview, label, (min(margin - r, w - 10), max(h - margin - r - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA,
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
    max_samples: int = BALL_DIAGNOSTIC_MAX_SAMPLES,
) -> Optional[Dict[str, Any]]:
    """Сэмплирует до max_samples равномерно распределённых по всему видео
    кадров и считает, в каком проценте из них модель вообще обнаруживает мяч
    (на текущем ball_conf_threshold/imgsz), плюс среднюю уверенность по
    обнаруженным кадрам. Если процент низкий даже на сниженном пороге — это
    сильный сигнал, что дело не в порогах, а в самой видимости мяча на видео
    (качество, освещение, расстояние до камеры, смаз при быстром полёте).
    """
    if model is None or cv2 is None:
        return None
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total_frames <= 0:
        cap.release()
        return None

    step = max(total_frames // max_samples, 1)
    sampled = 0
    detected = 0
    conf_sum = 0.0
    frame_idx = 0

    while frame_idx < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        detect_frame = enhance_frame_for_detection(frame) if enhance_quality else frame
        try:
            results = model.predict(
                detect_frame,
                classes=[COCO_BALL_CLASS_ID],
                conf=ball_conf_threshold,
                imgsz=imgsz,
                device=device,
                verbose=False,
            )
        except Exception:
            break
        sampled += 1
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            confs = boxes.conf.cpu().numpy()
            detected += 1
            conf_sum += float(np.max(confs))
        frame_idx += step

    cap.release()
    if sampled == 0:
        return None
    return {
        "sampled_frames": sampled,
        "frames_with_ball": detected,
        "detection_rate": detected / sampled,
        "avg_confidence": (conf_sum / detected) if detected else 0.0,
    }


# ---------------------------------------------------------------------------
# Быстрое предварительное сканирование для сбора списка игроков (шаг 3)
# ---------------------------------------------------------------------------
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
) -> Dict[int, Any]:
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
        result = results[0]
        boxes = result.boxes
        if boxes is not None and boxes.id is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            cls = boxes.cls.cpu().numpy().astype(int)
            ids = boxes.id.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()
            h, w = detect_frame.shape[:2]
            for box, c, tid, conf in zip(xyxy, cls, ids, confs):
                if c != COCO_PERSON_CLASS_ID or float(conf) < person_conf:
                    continue
                x1, y1, x2, y2 = [int(v) for v in box]
                area = max(x2 - x1, 0) * max(y2 - y1, 0)
                score = float(conf) * area
                prev = best_crops.get(int(tid))
                if prev is None or score > prev[0]:
                    x1c, y1c = max(x1, 0), max(y1, 0)
                    x2c, y2c = min(x2, w), min(y2, h)
                    crop = detect_frame[y1c:y2c, x1c:x2c].copy()
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
    pass_min_frames: int,
    pass_max_frames: int,
    goal_cooldown_frames: int,
    ball_memory_seconds: float,
    progress_bar,
    status_text,
    enhance_quality: bool = False,
    person_conf: float = PERSON_CONF_DEFAULT,
    ball_conf: float = BALL_CONF_DEFAULT,
    imgsz: int = IMGSZ_DEFAULT,
    camera_transforms: Optional[List[np.ndarray]] = None,
    ring_reference_frame_idx: int = 0,
) -> Tuple[Dict[int, PlayerStats], Path, Dict[str, List[Dict[str, Any]]]]:
    """Обрабатывает видео покадрово: детекция, трекинг, события, хайлайты.

    person_conf/ball_conf/imgsz — см. константы BALL_CONF_*/PERSON_CONF_*/
    IMGSZ_* выше: инференс идёт с единым низким conf=min(person_conf,
    ball_conf), а классы фильтруются раздельно постфактум в
    parse_track_results (мяч — более мелкий и часто менее уверенный объект,
    чем игрок, поэтому его порог обычно значительно ниже).

    camera_transforms/ring_reference_frame_idx — экспериментальный режим
    "камера в движении": если camera_transforms передан (список накопленных
    матриц из estimate_camera_transforms), позиции зон колец пересчитываются
    для каждого кадра относительно кадра ring_reference_frame_idx (на котором
    пользователь их задал на шаге 2), см. compute_dynamic_rings. Если None —
    поведение идентично исходному статичному режиму (зоны неподвижны).

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
    debug_log: Dict[str, List[Dict[str, Any]]] = {"ownership_changes": [], "sampled_timeline": []}

    # Состояние владения мячом (для пасов и для "кто владел мячом перед голом").
    last_owner: Optional[int] = None
    pass_origin: Optional[int] = None
    free_ball_frames: int = 0
    last_goal_frame_per_ring = [-10**9 for _ in rings]
    prev_ball_xy: Optional[Tuple[float, float]] = None

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
            persons, ball = parse_track_results(
                results, person_conf_threshold=person_conf, ball_conf_threshold=ball_conf
            )
            if enhance_quality:
                inv_scale = 1.0 / ENHANCE_UPSCALE_FACTOR
                persons = [(pid, tuple(v * inv_scale for v in box)) for pid, box in persons]
                if ball is not None:
                    ball = (ball[0] * inv_scale, ball[1] * inv_scale)
            annotated = draw_annotations(frame, persons, ball, list(ball_trajectory))
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

        # Режим "камера в движении": пересчитываем позиции зон колец под
        # текущий кадр относительно кадра, на котором они были заданы
        # (см. compute_dynamic_rings), и рисуем их на аннотированном кадре —
        # это одновременно и логика события (ниже), и визуальное подтверждение
        # того, что виртуальное кольцо действительно "следует" за панорамой.
        if camera_transforms is not None:
            current_rings = compute_dynamic_rings(rings, camera_transforms, ring_reference_frame_idx, frame_idx)
        else:
            current_rings = rings
        annotated = draw_zones_preview(annotated, current_rings)

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
            ball_trajectory.append((effective_ball[0], effective_ball[1]))

        # -------------------------------------------------------------
        # ВЛАДЕНИЕ МЯЧОМ И ДЕТЕКЦИЯ ПЕРЕДАЧ (ПАСОВ)
        #
        # Игрок владеет мячом, если расстояние от центра мяча до его bbox
        # меньше possession_threshold px (дефолт 90). Передача: Игрок_1 владел
        # мячом → мяч летел без владельца pass_min_frames..pass_max_frames
        # кадров → Игрок_2 получил владение → +1 пас Игроку_1.
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

        if current_owner is not None:
            if (
                pass_origin is not None
                and pass_origin != current_owner
                and pass_min_frames <= free_ball_frames <= pass_max_frames
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
                        reason = f"❌ отклонено (< {pass_min_frames} кадров без владельца)"
                    elif free_ball_frames > pass_max_frames:
                        reason = f"❌ отклонено (> {pass_max_frames} кадров без владельца)"
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
            last_owner = current_owner
        elif effective_ball is not None:
            if last_owner is not None:
                if pass_origin is None:
                    pass_origin = last_owner
                free_ball_frames += 1
                if free_ball_frames > pass_max_frames:
                    pass_origin = None
        else:
            if free_ball_frames > pass_max_frames:
                pass_origin = None
                free_ball_frames = 0

        if frame_idx % sample_every == 0:
            debug_log["sampled_timeline"].append(
                {
                    "Время, с": round(t, 2),
                    "Владелец (ID)": float(current_owner) if current_owner is not None else float("nan"),
                }
            )

        # -------------------------------------------------------------
        # ГОЛЫ: векторный анализ пересечения траектории мяча с горизонтальной
        # линией кольца (сверху вниз). Кулдаун — goal_cooldown_frames кадров.
        # Автором гола считается игрок, который последним владел мячом.
        # -------------------------------------------------------------
        if effective_ball is not None and prev_ball_xy is not None:
            bx_prev, by_prev = prev_ball_xy
            bx_curr, by_curr = effective_ball[0], effective_ball[1]
            for ring_idx, ring in enumerate(current_rings):
                half_w = ring.get("half_width", ring.get("r", 40.0))
                if not segment_crosses_hoop_line_top_to_bottom(
                    bx_prev, by_prev, bx_curr, by_curr, ring["y"], ring["x"], half_w
                ):
                    continue
                if (frame_idx - last_goal_frame_per_ring[ring_idx]) < goal_cooldown_frames:
                    continue
                last_goal_frame_per_ring[ring_idx] = frame_idx
                credited_player = last_owner
                if credited_player is None:
                    credited_player = nearest_player_to_point(persons, (ring["x"], ring["y"]))
                if credited_player is not None:
                    stats.setdefault(credited_player, blank_stats())["shots"] += 1
                    stats[credited_player]["makes"] += 1
                    pending_highlights.append(
                        PendingHighlight(
                            filename=f"goal_ring{ring_idx + 1}_ID{credited_player}_frame_{frame_idx}.mp4",
                            past_frames=list(frame_buffer),
                            frames_needed=future_frames_needed,
                        )
                    )

        if effective_ball is not None:
            prev_ball_xy = (effective_ball[0], effective_ball[1])

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
        "pass_min_frames": int(PASS_MIN_FRAMES_DEFAULT),
        "pass_max_frames": int(PASS_MAX_FRAMES_DEFAULT),
        "ball_memory": float(BALL_MEMORY_SECONDS_DEFAULT),
        "goal_cooldown_frames": int(GOAL_COOLDOWN_FRAMES_DEFAULT),
        "enhance_quality": False,
        "person_conf": float(PERSON_CONF_DEFAULT),
        "ball_conf": float(BALL_CONF_DEFAULT),
        "imgsz": int(IMGSZ_DEFAULT),
        "camera_mode": CAMERA_MODE_STATIC,
        "ball_diagnostics": None,
        "preview_time": 0.0,
        "avg_player_diagonal": None,
        "auto_threshold_computed_for": None,
        "click_target_ring": "Кольцо 1",
        "_last_ring_click_time": None,
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
        "avg_player_diagonal",
        "auto_threshold_computed_for",
        "_last_ring_click_time",
        "ball_diagnostics",
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
def render_step2_zones(device: str) -> None:
    st.header("Шаг 2 — Превью и настройка зон")
    video_path = st.session_state.get("video_path")
    if not video_path or not Path(video_path).exists():
        st.warning("Сначала загрузите видео на шаге 1.")
        if st.button("← Назад к загрузке"):
            go_to_step(1)
        return

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
        ring1, ring2 = default_ring_zones(int(meta["width"]), int(meta["height"]))
        st.session_state["ring1_x"], st.session_state["ring1_y"], st.session_state["ring1_r"] = (
            int(ring1["x"]),
            int(ring1["y"]),
            int(ring1.get("half_width", ring1.get("r", 40))),
        )
        st.session_state["ring2_x"], st.session_state["ring2_y"], st.session_state["ring2_r"] = (
            int(ring2["x"]),
            int(ring2["y"]),
            int(ring2.get("half_width", ring2.get("r", 40))),
        )
        st.session_state["rings_initialized_for"] = video_path

    max_t = max(meta["duration"] - 0.05, 0.0)
    st.session_state["preview_time"] = st.slider(
        "Кадр для превью (сек)", min_value=0.0, max_value=max_t if max_t > 0 else 0.1,
        value=min(st.session_state["preview_time"], max_t), step=0.5,
    )
    frame = extract_frame_at_time(video_path, st.session_state["preview_time"])

    # --- Авто-калибровка порога владения по среднему размеру игрока на кадре ---
    # Фиксированный порог в пикселях не учитывает масштаб конкретного видео
    # (камера близко/далеко, разное разрешение) — поэтому один раз на видео
    # (и по кнопке повторно) считаем средний размер рамки игрока и предлагаем
    # адаптивный дефолт вместо жёстких 90px.
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

    if st.session_state.get("camera_mode") == CAMERA_MODE_PANNING:
        st.info(
            "🎥 Режим «камера в движении» включён: зоны колец задаются здесь на выбранном кадре "
            "превью, а на шаге 4 их позиции будут автоматически пересчитываться под сдвиг камеры "
            "относительно ИМЕННО ЭТОГО кадра. Если поменяете секунду превью выше — точка отсчёта "
            "для пересчёта сдвинется вместе с ней."
        )

    st.subheader("🎯 Линии обоих колец")
    if streamlit_image_coordinates is not None and cv2 is not None:
        st.caption(
            "Кликните по превью, чтобы задать центр и Y горизонтальной линии выбранного кольца "
            "(линия рисуется горизонтально через заданную полуширину), либо используйте "
            "числовые поля ниже для точной донастройки."
        )
        st.session_state["click_target_ring"] = st.radio(
            "Клик по превью ставит линию кольца:", ["Кольцо 1", "Кольцо 2"],
            horizontal=True, key="click_target_ring_radio",
            index=0 if st.session_state.get("click_target_ring", "Кольцо 1") == "Кольцо 1" else 1,
        )
    else:
        st.caption(
            "Пакет streamlit-image-coordinates не установлен — доступна только точная настройка "
            "числовыми полями ниже (см. requirements.txt)."
        )

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
        rings = [
            {
                "x": st.session_state["ring1_x"],
                "y": st.session_state["ring1_y"],
                "half_width": st.session_state["ring1_r"],
            },
            {
                "x": st.session_state["ring2_x"],
                "y": st.session_state["ring2_y"],
                "half_width": st.session_state["ring2_r"],
            },
        ]
        preview_bgr = draw_zones_preview(frame, rings, possession_threshold=st.session_state["possession_threshold"])
        preview_rgb = cv2.cvtColor(preview_bgr, cv2.COLOR_BGR2RGB)

        if streamlit_image_coordinates is not None:
            display_width = min(int(meta["width"]), PREVIEW_MAX_DISPLAY_WIDTH)
            click_value = streamlit_image_coordinates(preview_rgb, key="ring_click_canvas", width=display_width)
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
                    target = "ring1" if st.session_state["click_target_ring"] == "Кольцо 1" else "ring2"
                    st.session_state[f"{target}_x"] = orig_x
                    st.session_state[f"{target}_y"] = orig_y
                    ring_click_triggered_rerun = True
        else:
            st.image(preview_rgb, caption="Превью с зонами колец", use_container_width=True)
    else:
        st.error("Не удалось прочитать кадр из видео для превью.")

    # ПРИМЕЧАНИЕ: у number_input ниже key совпадает с именем переменной в
    # session_state (например key="ring1_x" для st.session_state["ring1_x"]),
    # и намеренно НЕ делается `st.session_state["ring1_x"] = st.number_input(
    # ..., key="ring1_x")` — обе конструкции подряд означали бы запись в
    # session_state[key] уже после инстанциирования виджета с этим key
    # (в случае присваивания — сразу после его же создания), что Streamlit
    # запрещает. Раз key совпадает с именем состояния, виджет и так пишет
    # своё значение в session_state как побочный эффект — читать его дальше
    # по коду можно напрямую из session_state, без явного присваивания. Если
    # бы ключ виджета отличался от ключа состояния (как было раньше:
    # "in_ring1_x" vs "ring1_x"), клик по картинке обновлял бы состояние, но
    # поля ввода продолжали бы показывать старое значение до следующего
    # ручного изменения.
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Кольцо №1**")
        st.number_input(
            "X1 (px)", min_value=0, max_value=int(meta["width"]), value=int(st.session_state["ring1_x"]), key="ring1_x"
        )
        st.number_input(
            "Y1 (px)", min_value=0, max_value=int(meta["height"]), value=int(st.session_state["ring1_y"]), key="ring1_y"
        )
        st.number_input(
            "Полуширина линии 1 (px)", min_value=5, max_value=int(max(meta["width"], meta["height"])),
            value=int(st.session_state["ring1_r"]), key="ring1_r",
            help="Половина длины горизонтального отрезка линии кольца (от центра влево/вправо).",
        )
    with col2:
        st.markdown("**Кольцо №2**")
        st.number_input(
            "X2 (px)", min_value=0, max_value=int(meta["width"]), value=int(st.session_state["ring2_x"]), key="ring2_x"
        )
        st.number_input(
            "Y2 (px)", min_value=0, max_value=int(meta["height"]), value=int(st.session_state["ring2_y"]), key="ring2_y"
        )
        st.number_input(
            "Полуширина линии 2 (px)", min_value=5, max_value=int(max(meta["width"], meta["height"])),
            value=int(st.session_state["ring2_r"]), key="ring2_r",
            help="Половина длины горизонтального отрезка линии кольца (от центра влево/вправо).",
        )

    colcal1, colcal2 = st.columns([3, 1])
    with colcal1:
        if st.session_state.get("avg_player_diagonal"):
            st.caption(
                f"📏 Средний размер игрока на этом кадре: ~{st.session_state['avg_player_diagonal']:.0f}px по "
                f"диагонали рамки → авто-порог владения ~{suggest_possession_threshold(st.session_state['avg_player_diagonal'])}px "
                "(уже применён ниже, можно скорректировать слайдером)."
            )
        else:
            st.caption(
                "Авто-калибровка порога недоступна (модель не загружена или игроки не найдены на этом "
                f"кадре) — используется дефолт {POSSESSION_THRESHOLD_DEFAULT}px."
            )
    with colcal2:
        if st.button("🔄 Пересчитать по кадру", use_container_width=True):
            st.session_state["auto_threshold_computed_for"] = None
            st.rerun()

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
        st.session_state["pass_min_frames"] = st.slider(
            "Мин. кадров без владельца для паса",
            min_value=PASS_MIN_FRAMES_MIN,
            max_value=PASS_MIN_FRAMES_MAX,
            value=int(st.session_state["pass_min_frames"]),
            step=1,
            help="Мяч должен лететь без владельца не меньше этого числа кадров (дефолт 10).",
        )
        st.session_state["pass_max_frames"] = st.slider(
            "Макс. кадров без владельца для паса",
            min_value=PASS_MAX_FRAMES_MIN,
            max_value=PASS_MAX_FRAMES_MAX,
            value=int(st.session_state["pass_max_frames"]),
            step=1,
            help="Если мяч без владельца дольше — передача не засчитывается (дефолт 60).",
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

    st.session_state["goal_cooldown_frames"] = st.slider(
        "Кулдаун гола у кольца (кадры)",
        min_value=GOAL_COOLDOWN_FRAMES_MIN,
        max_value=GOAL_COOLDOWN_FRAMES_MAX,
        value=int(st.session_state["goal_cooldown_frames"]),
        step=5,
        help="Минимальный промежуток в кадрах между двумя голами у одного кольца (дефолт 90 ≈ 3 сек при 30 fps).",
    )

    st.subheader("🏀 Детекция мяча и производительность")
    st.caption(
        "Мяч — маленький и часто смазанный объект, его уверенность детекции обычно заметно ниже, "
        "чем у игроков. Поэтому порог для мяча по умолчанию ниже, чем для игроков. Если события "
        "(броски/передачи) не фиксируются — сначала запустите диагностику видимости мяча на шаге 3."
    )
    cconf1, cconf2 = st.columns(2)
    with cconf1:
        st.session_state["ball_conf"] = st.slider(
            "Порог уверенности для мяча (conf)",
            min_value=BALL_CONF_MIN, max_value=BALL_CONF_MAX,
            value=float(st.session_state["ball_conf"]), step=0.01,
            help="Ниже — мяч обнаруживается чаще, но растёт риск ложных срабатываний на бликах/похожих объектах.",
        )
    with cconf2:
        st.session_state["person_conf"] = st.slider(
            "Порог уверенности для игроков (conf)",
            min_value=PERSON_CONF_MIN, max_value=PERSON_CONF_MAX,
            value=float(st.session_state["person_conf"]), step=0.05,
            help="Выше — меньше ложных рамок на фоне/зрителях, но риск пропустить игрока в сложной позе/перекрытии.",
        )
    st.session_state["imgsz"] = st.select_slider(
        "Разрешение инференса (imgsz, px)",
        options=IMGSZ_OPTIONS,
        value=int(st.session_state["imgsz"]) if int(st.session_state["imgsz"]) in IMGSZ_OPTIONS else IMGSZ_DEFAULT,
        help="Выше — мяч (мелкий объект) занимает больше пикселей после ресайза модели и его легче "
        "обнаружить, но обработка заметно замедляется. 640 — дефолт ultralytics, 960-1280 рекомендуется "
        "для видео с плохо видимым мячом.",
    )
    if int(st.session_state["imgsz"]) > IMGSZ_DEFAULT:
        st.caption(f"⚠️ imgsz={int(st.session_state['imgsz'])} заметно медленнее дефолтных {IMGSZ_DEFAULT}px, особенно на CPU.")

    st.subheader("🔧 Качество детекции")
    st.session_state["enhance_quality"] = st.checkbox(
        "Улучшить качество кадра перед детекцией (апскейл + резкость)",
        value=st.session_state.get("enhance_quality", False),
        help="Может помочь трекеру/ReID различать игроков на видео низкого разрешения. "
        "Не панацея: если исходное видео изначально сильно сжато/размыто, апскейл не "
        "восстановит потерянные детали. Замедляет обработку.",
    )
    if st.session_state["enhance_quality"]:
        st.caption(
            "⚠️ Это не панацея при изначально плохом качестве видео — лишь может немного помочь "
            "трекеру на видео низкого разрешения, ценой более медленной обработки."
        )

    colA, colB = st.columns(2)
    with colA:
        if st.button("← Назад к загрузке"):
            go_to_step(1)
    with colB:
        if st.button("Далее → Сопоставление игроков", type="primary"):
            go_to_step(3)

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
        "и по одному характерному кадру-кропу на каждого. Номера на форме не видны камере и "
        "распознать их автоматически невозможно — впишите имя/номер вручную по кропам."
    )

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
                crops = quick_player_scan(
                    video_path, model, device, max_seconds=scan_seconds, fps_hint=meta["fps"],
                    enhance_quality=st.session_state.get("enhance_quality", False),
                    person_conf=float(st.session_state.get("person_conf", PERSON_CONF_DEFAULT)),
                    ball_conf=float(st.session_state.get("ball_conf", BALL_CONF_DEFAULT)),
                    imgsz=int(st.session_state.get("imgsz", IMGSZ_DEFAULT)),
                )
            st.session_state["player_crops"] = crops
            if not crops:
                st.warning("Игроки не найдены за это время — попробуйте увеличить длительность сканирования.")
            else:
                st.success(f"Найдено {len(crops)} уникальных ID игроков.")

    crops: Dict[int, Any] = st.session_state.get("player_crops") or {}
    if crops:
        # Компактная сетка: все кропы приводятся к одинаковой высоте (пропорции
        # сохраняются), поэтому карточки игроков ровные независимо от того,
        # насколько разного размера/ориентации были исходные рамки детекций.
        cols_per_row = 6
        ids_sorted = sorted(crops.keys())
        for row_start in range(0, len(ids_sorted), cols_per_row):
            row_ids = ids_sorted[row_start : row_start + cols_per_row]
            cols = st.columns(cols_per_row)
            for col, pid in zip(cols, row_ids):
                with col:
                    with st.container(border=True):
                        resized_crop = resize_crop_to_height(crops[pid], CROP_DISPLAY_HEIGHT)
                        st.image(cv2.cvtColor(resized_crop, cv2.COLOR_BGR2RGB), caption=f"ID {pid}")
                        st.text_input("Имя", key=f"player_name_{pid}", placeholder=f"Игрок {pid}", label_visibility="collapsed")
                        st.text_input("Номер", key=f"player_number_{pid}", placeholder="Номер", label_visibility="collapsed")
    else:
        st.info(
            "Пока нет данных — запустите сканирование выше, либо пропустите этот шаг: "
            "в итоговой таблице игроки будут отображаться как «Игрок {ID}»."
        )

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
            with st.spinner("Сэмплирование кадров по всему видео..."):
                diag = diagnose_ball_visibility(
                    video_path, model, device,
                    ball_conf_threshold=float(st.session_state.get("ball_conf", BALL_CONF_DEFAULT)),
                    imgsz=int(st.session_state.get("imgsz", IMGSZ_DEFAULT)),
                    enhance_quality=st.session_state.get("enhance_quality", False),
                )
            st.session_state["ball_diagnostics"] = diag
            if diag is None:
                st.error("Не удалось выполнить диагностику (не открылось видео или сбой модели).")

    diag = st.session_state.get("ball_diagnostics")
    if diag:
        rate_pct = diag["detection_rate"] * 100.0
        st.caption(
            f"Сэмплировано кадров: {diag['sampled_frames']} · мяч обнаружен в "
            f"{diag['frames_with_ball']} из них ({rate_pct:.0f}%) · средняя уверенность по "
            f"обнаруженным кадрам: {diag['avg_confidence']:.2f}"
        )
        if rate_pct >= 50:
            st.success(
                f"✅ Мяч обнаруживается стабильно ({rate_pct:.0f}% кадров, средняя уверенность "
                f"{diag['avg_confidence']:.2f}). Если события всё равно не фиксируются — дело, скорее "
                "всего, в порогах владения/передач или кулдауне (шаг 2), а не в видимости мяча."
            )
        elif rate_pct >= 20:
            st.warning(
                f"⚠️ Мяч обнаруживается умеренно часто ({rate_pct:.0f}% кадров, средняя уверенность "
                f"{diag['avg_confidence']:.2f}) — вероятно проблема с качеством видео/освещением/"
                "дистанцией съёмки. Попробуйте повысить imgsz и/или включить улучшение качества кадра "
                "на шаге 2, либо ещё немного снизить порог уверенности для мяча."
            )
        else:
            st.error(
                f"❌ Мяч обнаружен лишь в {rate_pct:.0f}% кадров (средняя уверенность "
                f"{diag['avg_confidence']:.2f}) — вероятно проблема в качестве видео/освещении/"
                "расстоянии до камеры, а не в порогах. Броски и передачи почти наверняка будут "
                "фиксироваться редко или не будут вовсе, пока мяч физически не виден модели чаще."
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
        {
            "x": st.session_state["ring1_x"],
            "y": st.session_state["ring1_y"],
            "half_width": st.session_state["ring1_r"],
        },
        {
            "x": st.session_state["ring2_x"],
            "y": st.session_state["ring2_y"],
            "half_width": st.session_state["ring2_r"],
        },
    ]

    camera_transforms: Optional[List[np.ndarray]] = None
    ring_reference_frame_idx = 0
    if st.session_state.get("camera_mode") == CAMERA_MODE_PANNING and cv2 is not None:
        meta = get_video_metadata(video_path)
        ring_reference_frame_idx = int(round(float(st.session_state.get("preview_time", 0.0)) * meta["fps"]))
        st.info("🎥 Экспериментальный режим «камера в движении»: оцениваю сдвиг камеры по всему видео...")
        cam_progress = st.progress(0.0)
        try:
            camera_transforms = estimate_camera_transforms(video_path, progress_callback=cam_progress.progress)
        except Exception as exc:
            st.warning(f"⚠️ Не удалось оценить движение камеры ({exc}) — зоны колец останутся фиксированными.")
            camera_transforms = None
        cam_progress.progress(1.0)

    progress_bar = st.progress(0.0)
    status_text = st.empty()

    try:
        stats, output_path, debug_log = process_video(
            video_path=video_path,
            model=model,
            device=device,
            rings=rings,
            possession_threshold=float(st.session_state["possession_threshold"]),
            pass_min_frames=int(st.session_state["pass_min_frames"]),
            pass_max_frames=int(st.session_state["pass_max_frames"]),
            goal_cooldown_frames=int(st.session_state["goal_cooldown_frames"]),
            ball_memory_seconds=float(st.session_state["ball_memory"]),
            progress_bar=progress_bar,
            status_text=status_text,
            enhance_quality=bool(st.session_state.get("enhance_quality", False)),
            person_conf=float(st.session_state.get("person_conf", PERSON_CONF_DEFAULT)),
            ball_conf=float(st.session_state.get("ball_conf", BALL_CONF_DEFAULT)),
            imgsz=int(st.session_state.get("imgsz", IMGSZ_DEFAULT)),
            camera_transforms=camera_transforms,
            ring_reference_frame_idx=ring_reference_frame_idx,
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
    ensure_directories()
    highlight_files = sorted(HIGHLIGHTS_DIR.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not highlight_files:
        st.info("Нарезанные хайлайты появятся здесь после обнаружения бросков/передач в обработанном видео.")
        return

    for hf in highlight_files:
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

    with st.expander("⚙️ Текущие настройки анализа", expanded=False):
        st.write(
            f"Кольцо 1: X={int(st.session_state['ring1_x'])}, Y={int(st.session_state['ring1_y'])}, "
            f"полуширина={int(st.session_state['ring1_r'])} · Кольцо 2: X={int(st.session_state['ring2_x'])}, "
            f"Y={int(st.session_state['ring2_y'])}, полуширина={int(st.session_state['ring2_r'])}"
        )
        st.write(
            f"Порог владения: {int(st.session_state['possession_threshold'])} px · "
            f"Окно паса: {int(st.session_state['pass_min_frames'])}–{int(st.session_state['pass_max_frames'])} кадров · "
            f"Память мяча: {st.session_state['ball_memory']:.2f} с · "
            f"Кулдаун гола: {int(st.session_state['goal_cooldown_frames'])} кадров"
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
            "Режим камеры: "
            + CAMERA_MODE_LABELS.get(st.session_state.get("camera_mode", CAMERA_MODE_STATIC), "статичная")
        )
        diag = st.session_state.get("ball_diagnostics")
        if diag:
            st.write(
                f"Диагностика видимости мяча: обнаружен в {diag['detection_rate'] * 100:.0f}% "
                f"сэмплированных кадров (средняя уверенность {diag['avg_confidence']:.2f})"
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
    st.caption("Офлайн-аналитика баскетбольных тренировок по видео со статичной камеры (YOLO11x + ByteTrack)")

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
        render_step2_zones(device)
    elif step == 3:
        render_step3_players(device)
    else:
        render_step4_run(device)


if __name__ == "__main__":
    main()

"""Application constants and paths."""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Dict

import numpy as np

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
HIGHLIGHTS_DIR = BASE_DIR / "highlights"
OUTPUT_DIR = BASE_DIR / "output_videos"
PROJECT_CACHE_DIR = BASE_DIR / ".cache"
UPLOAD_CACHE_DIR = PROJECT_CACHE_DIR / "uploads"
HIGHLIGHT_FRAME_CACHE_DIR = PROJECT_CACHE_DIR / "highlight_frames"
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
RingZone = Dict[str, float]

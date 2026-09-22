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

import sys
import types

from basketball.offline import apply_offline_env

apply_offline_env()

import basketball.config as _config
import basketball.core as _core
from basketball.ui import main as _ui_main


class _AppModule(types.ModuleType):
    """Proxy module: re-exports basketball.* and forwards private mutations to core (tests)."""

    _MUTABLE_CORE = ("_REID_MODEL", "_REID_DEVICE", "_REID_LAST_ERROR")

    def __getattribute__(self, name: str):
        if name in _AppModule._MUTABLE_CORE:
            return getattr(_core, name)
        try:
            return super().__getattribute__(name)
        except AttributeError:
            if hasattr(_config, name):
                return getattr(_config, name)
            return getattr(_core, name)

    def __setattr__(self, name: str, value) -> None:
        if name in _AppModule._MUTABLE_CORE:
            setattr(_core, name, value)
            return
        if hasattr(_config, name):
            setattr(_config, name, value)
            return
        super().__setattr__(name, value)


_mod = _AppModule(__name__, __doc__)

for _name in dir(_config):
    if not _name.startswith("__"):
        setattr(_mod, _name, getattr(_config, _name))

for _name in dir(_core):
    if not _name.startswith("__"):
        setattr(_mod, _name, getattr(_core, _name))

for _name in dir(_core):
    if _name.startswith("_") and not _name.startswith("__"):
        _val = getattr(_core, _name)
        if callable(_val):
            setattr(_mod, _name, _val)

_mod.main = _ui_main
_mod.test_draw_hoop_lines_on_frame = _core.test_draw_hoop_lines_on_frame

sys.modules[__name__] = _mod

if __name__ == "__main__":
    if not _mod.test_draw_hoop_lines_on_frame():
        raise RuntimeError("test_draw_hoop_lines_on_frame: пиксели линии кольца остались нулевыми")
    _mod.main()

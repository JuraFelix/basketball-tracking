"""Streamlit wizard UI."""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

from basketball.config import *
from basketball.core import *
from basketball.core import _make_ring_widget_change_handler, _ring_anchor_frame_idx

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
        store_uploaded_video(uploaded_file)

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
        st.divider()
        if st.button("🧹 Очистить кэш проекта"):
            cleared = clear_project_cache()
            st.success("Кэш очищен:\n" + "\n".join(f"• {p}" for p in cleared))
        st.caption(
            "Удаляет highlights/, output_videos/ и .cache/ (временные загрузки и буферы кадров). "
            "Веса YOLO в ~/.cache не трогаются."
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
    if not test_draw_hoop_lines_on_frame():
        raise RuntimeError("test_draw_hoop_lines_on_frame: пиксели линии кольца остались нулевыми")
    main()

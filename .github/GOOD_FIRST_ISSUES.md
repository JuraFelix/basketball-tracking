# Good first issues

Copy these into GitHub Issues (labels: `help wanted`, `good first issue`) or run the `gh issue create` commands in [CONTRIBUTING.md](../CONTRIBUTING.md) once `gh` is authenticated.

---

## 1. Improve ball recall on dark / low-res gym video

**Labels:** `help wanted`, `good first issue`, `detection`

### Problem

On dark or low-resolution gym footage, the ball is often missed for many consecutive frames. The pipeline uses YOLO COCO class `sports ball` (class id 32). Step 2 already supports **user click anchors** and color fallback, but recall before those kicks in is still weak.

### Context

- Detection runs via `model.track()` with configurable `ball_conf` and `imgsz` (step 2 sliders).
- Ball memory / CSRT / tiled ROI recovery exist downstream — the goal is to **detect earlier** without tanking FPS.
- Seeds for future fine-tuning land in `training_seeds/` (not used in-app today).

### Suggested directions

- Tune ball-specific post-filters in `parse_track_results` (size, aspect ratio, motion consistency).
- Smarter ROI re-scan when person clusters occlude the court center.
- Optional lighter pass (smaller `imgsz` or tiled crop) only when ball confidence drops — measure FPS impact.

### Acceptance hints

- Repro video or frame range where current build loses the ball for >1 s while it is visible.
- Before/after: fewer missed stretches; document FPS delta on a mid-range GPU (e.g. RTX 3060 class).

---

## 2. Reduce player ID swaps at crossings

**Labels:** `help wanted`, `good first issue`, `tracking`

### Problem

When two players cross paths, tracker IDs sometimes swap. ByteTrack is configured in `custom_bytetrack.yaml`; identity stabilization uses **MobileNet ReID** + jersey HSV (+ optional EasyOCR). Step 3 offers **visual ID merge**, but swaps still hurt box score and pass attribution.

### Context

- Anti-swap gating prefers appearance when IoU overlap is high (`test_identity.py` covers some cases).
- Dense crossings without clear appearance separation remain hard.

### Suggested directions

- Strengthen ReID embedding comparison window around crossing events.
- Defer ID handoff until trajectories separate by minimum distance.
- Surface a “suspected swap” hint on step 3 using appearance distance spikes.

### Acceptance hints

- Short clip or timestamp where swap occurs today.
- After fix: same clip keeps consistent canonical IDs through the crossing (or merge suggestion is one-click).

---

## 3. Stabilize hoop line under panning camera

**Labels:** `help wanted`, `good first issue`, `homography`

### Problem

In **panning / moving camera** mode, the horizontal hoop line drifts relative to the physical rim between user anchor frames. Multi-frame hoop anchors exist on step 2; `transform_ring_to_frame` interpolates, but homography can wobble on fast pans.

### Context

- Static camera: single click per hoop still works.
- Panorama mode: user places anchors on several frames; goal detection uses the interpolated line per frame.

### Suggested directions

- Smooth anchor-derived transforms (temporal filter on line endpoints).
- Optional feature-based rim refinement when backboard/rim edges are visible.
- Regression test with synthetic or recorded pan clip (`test_app_rings.py` as a pattern).

### Acceptance hints

- Pan clip where line visibly lags or leads the rim.
- After fix: line stays within a few pixels of the rim on intermediate frames; no new false goal triggers.

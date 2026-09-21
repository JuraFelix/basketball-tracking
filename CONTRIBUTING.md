# Contributing

Thanks for your interest in **Basketball Tracking Analytics**!

- **Origin (canonical):** https://cursor.com/codebase/felipok/genesis
- **GitHub mirror (PRs):** https://github.com/JuraFelix/basketball-tracking

## Quick start

1. Fork [JuraFelix/basketball-tracking](https://github.com/JuraFelix/basketball-tracking) on GitHub.
2. Clone your fork and create a branch from `main`.
3. Set up a virtual environment and install dependencies (CUDA PyTorch recommended):

   ```bash
   python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
   pip install -r requirements.txt
   streamlit run app.py
   ```

4. Make your changes. Keep PRs focused; do not edit unrelated files.
5. Run checks before opening a PR:

   ```bash
   python3 -m py_compile app.py
   python3 -m unittest test_app_rings.py test_identity.py -v
   ```

6. Open a pull request against `main` on **JuraFelix/basketball-tracking** with a clear description and, if applicable, a short screen recording or screenshot.

**CI note:** This project is GPU-oriented (YOLO + CUDA). We do not promise GPU runners in CI — local verification on your machine is expected.

## Code style

- Match existing style in `app.py` and tests.
- Prefer small, reviewable diffs over large refactors.
- Add or update tests when you change trackable behavior.

## Suggested first issues

Good entry points for new contributors (details and acceptance hints in [`.github/GOOD_FIRST_ISSUES.md`](.github/GOOD_FIRST_ISSUES.md)):

1. **Ball recall on dark / low-res gym footage** — improve YOLO COCO `sports ball` detection without a large FPS drop; user click anchors already exist on step 2.
2. **Player ID swaps at crossings** — ByteTrack + MobileNet ReID; visual merge on step 3 exists; reduce swap rate.
3. **Panning camera: hoop line drift** — multi-frame hoop anchors exist; stabilize homography between anchor frames.

Labels to use when creating these on GitHub: `help wanted`, `good first issue`.

---

## Как внести вклад (RU)

1. Сделайте **fork** репозитория [JuraFelix/basketball-tracking](https://github.com/JuraFelix/basketball-tracking) на GitHub.
2. Клонируйте свой fork, создайте ветку от `main`.
3. Виртуальное окружение и CUDA-сборка PyTorch (рекомендуется):

   ```bash
   python -m venv .venv
   .venv\Scripts\activate          # Windows
   # source .venv/bin/activate     # Linux/macOS
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
   pip install -r requirements.txt
   streamlit run app.py
   ```

4. Внесите изменения, прогоните `py_compile` и unit-тесты (см. выше).
5. Откройте **pull request** в `main` на GitHub **JuraFelix/basketball-tracking**.

Канонический репозиторий разработки — **Origin:** https://cursor.com/codebase/felipok/genesis. GPU в CI не гарантируется; проверяйте локально на своей видеокарте.

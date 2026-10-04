# Files that are NOT in this zip (I never received them)

Copy your own versions into these paths. I did not recreate them: guessing would overwrite working code.

backend/app/__init__.py (if you have one)
backend/app/mathparse.py     (rules.py imports it)
backend/app/diagnose.py      (must export Engine and ACCEPT_THRESHOLD)
backend/app/journey.py
backend/app/llm.py
backend/app/ocr.py
backend/app/typed_cases.py   (evaluate.py experiment B)
backend/app/main.py
backend/tests/test_engine.py
frontend/src/main.jsx  App.jsx  Solve.jsx  Pages.jsx  ui.jsx  api.js  (+ any css; index.html loads /src/main.jsx)

Other notes
- `relearn.db` is your uploaded demo database, placed in backend/. Move it if your code expects another path.
- `docs/evaluation_report.json` was removed on purpose: it is stale (250 attempts / 60 classes vs 254 / 64 in the CSV).
  Regenerate with `cd backend && python -m app.evaluate`.
- `relearn_service.py` was dropped: it called functions that do not exist.
- The uploaded config files had underscores in their names; they are renamed to vite.config.js, tailwind.config.js,
  postcss.config.js and _env.example -> frontend/.env.example.

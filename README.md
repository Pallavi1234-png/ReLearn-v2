# Re:Learn - find the thinking error, not just the wrong answer

Adaptive misconception diagnosis for Maharashtra SSC Std 9/10 algebra
(Quadratic Equations, Factorisation, Arithmetic Progression, Real Numbers).

Flow: **Attempt -> Diagnosis + Evidence -> Targeted intervention -> Reassessment -> Transfer -> Resolution -> Learner model**

## Run locally (2 terminals)

### 1) Backend (Python 3.10+)
```bash
cd backend
python -m venv .venv
# Windows:   .venv\Scripts\activate        Mac/Linux:  source .venv/bin/activate
pip install -r requirements.txt
python -m tests.test_engine          # should end with ALL PASSED
python -m tests.test_collisions      # same-wrong-answer regression guard, should end with ALL PASSED
python -m app.datagen                # (re)writes data/generated_cases.*
python -m app.evaluate               # writes docs/evaluation_report.json (takes a few minutes)
uvicorn app.main:app --reload --port 8000
```
Check http://localhost:8000/api/health

### 2) Frontend (Node 18+)
```bash
cd frontend
npm install
npm run dev
```
Open http://localhost:5173 and press **Run the demo: 3(x + 2) = 15**.
On each stage press "Autofill demo answer" to walk Attempt -> Reassessment -> Transfer -> Resolved,
then open **My Fingerprint** and **Misconception Map**.

### Optional: Ollama (nicer explanations)
```bash
ollama pull llama3.2
ollama serve
```
Ollama only *phrases* the explanation. The diagnosis never depends on it; if it is off the dataset's own
intervention text is used (the UI shows the explanation source).
Disable with `RELEARN_USE_LLM=0`. Change model with `OLLAMA_MODEL=...`.

## Deploy
**Backend (Render, free tier)**: New > Web Service > connect repo, Root Directory `backend`,
Build `pip install -r requirements.txt`, Start `uvicorn app.main:app --host 0.0.0.0 --port $PORT`.
Note: Render's free disk is ephemeral, so `relearn.db` (learner history) resets on redeploy - fine for a demo.
Ollama will not be available there; the dataset text fallback is used automatically.

**Frontend (Vercel)**: Import repo, Root Directory `frontend`, Framework Vite,
add env var `VITE_API_URL=https://<your-render-service>.onrender.com`.

## How diagnosis works (honest version)
1. **Answer check** - SymPy compares the final answer to `expected_answer` (roots, expressions, fully-factored form).
2. **Rule detectors** (`app/rules.py`, ~25 patterns) read the *actual maths* in each step and produce evidence,
   e.g. "multiplier 3 applied to x but not to 2". Each detector maps to a `misconception_id` in the supplied taxonomy.
3. **Step-consistency check** - flags a step that does not follow from the previous equation.
4. **ML signal** (`app/ml.py`) - TF-IDF + Logistic Regression over the supplied attempts, including a "no misconception" class.
   Used only when no rule fires, and it needs both a close dataset match and high probability.
5. **Collision-aware abstention** - several different misconceptions can reach the *same* wrong answer (for example
   x = -2, -3 for x² - 5x + 6 = 0 from a wrong factorisation, a sign slip when reading roots, or b used instead of -b). A rule reads the
   working to separate them (`evidence_level: steps`). When the working cannot separate them, the finding is capped at 0.55,
   so the engine abstains and asks for the next line instead of taking the first rule that matched.
6. **Abstention** - below confidence 0.70, or with no working, the system says
   "Not enough evidence. Please show the next step of your working."
7. **Resolution** requires: reassessment passed with visible working AND transfer passed. A bare correct answer never counts.

## Data notes (read before the demo)
- **`data/generated_cases.*` is NOT supplied data either.** It is 88 templated, team-generated cases (`app/datagen.py`):
  correct and incorrect working with labels, built so that different misconceptions reach the identical wrong answer.
  See `data/DATASET.md`. Set `RELEARN_USE_GENERATED=1` to also train the text model on it (off by default).
- Rule confidences (0.85-0.95) are fixed per rule, not calibrated probabilities. Only ambiguity changes them.
- Datasets used: `data/questions.csv`, `misconceptions.csv`, `student_attempts.csv`, `interventions.csv` (your supplied files, unmodified).
- **`data/extension_distribution.json` is NOT supplied data.** The taxonomy has no distribution-error record,
  so the headline demo uses a team-authored extension (`EXT-DIST-01`, 2 questions). The UI labels it as such.
- The supplied misconception descriptions / evidence patterns are templated, and most wrong attempts narrate the
  mistake in words instead of showing maths. So the rule engine, not the ML model, does the real diagnosis.
- Some `candidate_misconception_ids` look misaligned with the misconception names (e.g. AP nth-term vs "term check" ids).
  Rules therefore choose the id by meaning first and use candidates only to break ties. Worth a teacher review.

## Evaluation (all synthetic - no real students)
Run `python -m app.evaluate`. Four separate experiments; the report carries a data fingerprint so a stale report is obvious.
A. ML-only on the supplied simulated attempts: random 10-fold, leave-question-out, and unseen-class. Each class in the supplied data has
exactly ONE wording, so "unseen wording, seen class" cannot be measured from it; leakage and class-seen rates are reported per regime.
B. Full engine on developer-written typed cases (including false-alarm and abstention checks). Written by the rule author: a regression
guard, not an accuracy estimate.
C. Unseen misconceptions removed from training, **with a seen-class control at the same threshold**. Read it only as a difference between the two rates.
D. Same-wrong-answer collisions on disjoint number ranges: final-answer-only ceiling vs text model vs full working vs full engine.
Do not quote B or D's engine numbers as "accuracy": the cases and the rules share an author. Have someone who has not read `rules.py`
write fresh cases and report that number separately.

## Layout
```
data/        the 4 CSVs + extension_distribution.json + generated_cases.csv/.jsonl + DATASET.md
backend/app  mathparse.py rules.py ml.py diagnose.py journey.py llm.py ocr.py evaluate.py datagen.py typed_cases.py main.py
backend/tests/test_engine.py test_collisions.py
frontend/src App.jsx Solve.jsx Pages.jsx ui.jsx api.js
docs/        evaluation_report.json
```
OCR: the drawing pad sends a PNG to `/api/ocr`; `app/ocr.py` is a stub (`available:false`). Implement `recognise()` to plug in a real engine.

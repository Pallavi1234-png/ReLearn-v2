# Re:Learn update (overlay)

**How to apply:** unzip over your repo root (paths mirror the README layout). Then:
1. Delete the dead stub `relearn_service.py` (it calls functions that do not exist).
2. Delete the old `docs/evaluation_report.json` (it describes 250 attempts / 60 classes; your CSV has 254 / 64) and regenerate it:
   `cd backend && python -m tests.test_engine && python -m tests.test_collisions && python -m app.evaluate`
3. If `python -m tests.test_engine` or experiment B changes, send me the output; I could not run your full engine.

**Not in this overlay because I never received them:** `mathparse.py`, `diagnose.py`, `journey.py`, `llm.py`, `ocr.py`, `main.py`,
`typed_cases.py`, `tests/test_engine.py`, `frontend/src/*`. Nothing here edits them. The new Finding fields have defaults and the
engine's existing 0.70 threshold does the abstaining, so `diagnose.py` should need no change, but that is unverified.

## Files
| File | Change |
|---|---|
| `backend/app/rules.py` | `Finding` gains `alternatives` + `evidence_level`; `mark_ambiguous()` caps cross-id collisions at 0.55; formula route read from steps (`\u00b1`) not only the prompt; reads b vs -b from the working; distribution only claims 0.95 when the wrong expansion is written; discriminant reports every mechanism that gives the value and reads `+ 4ac` from steps; exponent evidence names the matched mechanism; removed unreachable `DISC_C_SIGN` entry, dead `right_piece`, unused `m`; parenthesised the `or`/`and` in `rule_quadratic` |
| `backend/app/ml.py` | opt-in `RELEARN_USE_GENERATED=1` adds generated working to training (default off) |
| `backend/app/datagen.py` | new: mutators, 88 labelled cases, held-out number ranges, `score()` harness |
| `backend/app/evaluate.py` | A: 3 split regimes + leakage/class-seen rates; B: CIs, honest note; C: seen-class control; D: collision experiment; data fingerprint |
| `backend/tests/test_collisions.py` | new: 264 collision checks + ambiguity unit tests |
| `data/generated_cases.*`, `data/DATASET.md` | new dataset contribution + documentation |
| `README.md` | updated run commands, diagnosis steps, data notes, evaluation, layout |

## Verification actually done (and its limits)
Run in a sandbox with a **stand-in** `mathparse.py` that I wrote from how `rules.py` calls it, plus a stub engine. So these check my logic,
not your parser:
- old vs new rules on 508 runs over all supplied attempts: **0 outputs changed**; 0 false alarms on the 120 correct answers.
- original rules on the collision cases: ambiguous distribution labelled `EXT-DIST-01` at 0.95 (12/12); formula-route case got 0.65 and no `QE-FORM-02`. After the patch: 0.55 (abstains) and `QE-FORM-02` at 0.93.
- `test_collisions`: 264/264 pass. **Circular:** the cases and the patches have the same author, so this is a regression guard, not accuracy.

## Findings you should know about
- **Experiment C was not the strong result it looked like.** Unseen-class abstention 0.879 vs seen-class control 0.831: the text model
  abstains on nearly everything (only 11% of seen-class wrong attempts are accepted *and* correct). Do not present 87.5% as evidence.
- **Unseen-wording evaluation is impossible on the supplied data**: every class has one wording. Grouping by question still leaks
  identical text into train for 95% of test attempts.
- Text model, held-out numbers (D): final answer only 0.20 [0.09, 0.32]; full working 0.54 [0.41, 0.67]; final-answer-only ceiling 0.33.
  The working carries signal the answer lacks, but the TF-IDF model is weak (DIST 0.25).

## Key-feature audit against the problem statement
| Requirement | Status |
|---|---|
| Misconception dataset (correct, incorrect, label) | **Met, extended**: 254 supplied (130 correct / 124 wrong / 64 classes) + 88 generated, documented. Gap: one wording per class, no real students |
| Misconception model (train + evaluate) | **Partial.** TF-IDF + LR is trained and evaluated honestly now, but the rules do the real diagnosis. A learned model on structured features was *not* added: trained on our own generators it would just relearn our rules. It needs independently written data |
| Differentiation of similar mistakes | **Added**: step-level formula route, collision-aware abstention, collision set, experiment D. Caveat: self-authored |
| Adaptive intervention | **Present** per README and `events`/`sessions` tables (intervention stage logged). `journey.py`/`llm.py` not inspected |
| Resolution assessment | **Present** per DB (resolved session needs reassessment passed + transfer passed). `journey.py` not inspected |
| Learner model | **Present** (`events` keyed by student + misconception, "My Fingerprint" page per README). Aggregation code not inspected |
| Model evaluation incl. unseen | **Fixed** (see above). Still missing: an independently written test set and any real-student data |
| "Multimodal" | **Not added.** `ocr.py` is a stub; a real recogniser needs `ocr.py`/`main.py` and an external model or API |

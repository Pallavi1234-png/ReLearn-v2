"""Evaluation module. Run from backend/:  python -m app.evaluate   (writes docs/evaluation_report.json)

Four clearly separated experiments - none of them uses real students:
 A. ML-only on the supplied SIMULATED attempts, three split regimes (and a data-diversity audit).
 B. Full engine (rules + ML + abstention) on developer-authored typed cases.
 C. UNSEEN misconceptions (classes removed from training) WITH a seen-class control at the same threshold.
 D. SAME-WRONG-ANSWER collisions: different misconceptions that reach the identical wrong answer.
    Compares a final-answer-only ceiling, a text model on final answers, a text model on the working
    (trained and tested on DISJOINT number ranges) and, when available, the full engine.
Every report carries a data fingerprint so a stale report is obvious.
"""
import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import GroupKFold, KFold

from .diagnose import Engine, ACCEPT_THRESHOLD
from .ml import MLDiagnoser, NONE, DATA, load_tables
from .rules import Q
from .typed_cases import CASES
from . import datagen

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parents[2] / "docs"


def _prf(y, p, labels):
    pr, rc, f1, sup = precision_recall_fscore_support(y, p, labels=labels, zero_division=0)
    return {"macro_precision": round(float(pr.mean()), 3), "macro_recall": round(float(rc.mean()), 3),
            "macro_f1": round(float(f1.mean()), 3)}


def _boot_ci(flags, n=2000, seed=0):
    """95% bootstrap CI for a mean of 0/1 flags. Small samples give wide intervals - that is the point."""
    f = np.asarray(flags, dtype=float)
    if len(f) == 0:
        return [None, None]
    rng = np.random.default_rng(seed)
    means = rng.choice(f, size=(n, len(f)), replace=True).mean(axis=1)
    return [round(float(np.percentile(means, 2.5)), 3), round(float(np.percentile(means, 97.5)), 3)]


def _fingerprint(att):
    h = hashlib.sha256((DATA / "student_attempts.csv").read_bytes()).hexdigest()[:12]
    wrong = att[att.misconception_id.notna()]
    return {"student_attempts_sha256_12": h, "n_attempts": int(len(att)), "n_wrong_attempts": int(len(wrong)),
            "n_misconception_classes": int(wrong.misconception_id.nunique()),
            "classes_with_a_single_distinct_wording": int(
                (wrong.groupby("misconception_id").student_working.nunique() == 1).sum())}


def _ml_cv(att, splitter, groups=None):
    att = att.reset_index(drop=True)
    att["label"] = att.misconception_id.fillna(NONE)
    preds, folds = [None] * len(att), []
    for tr, te in (splitter.split(att, groups=groups) if groups is not None else splitter.split(att)):
        folds.append((tr, te))
        m = MLDiagnoser(att.iloc[tr])
        for j in te:
            r = att.iloc[j]
            preds[j] = m.predict(r.question, str(r.student_working), None, topn=1)["top"][0][0]
    return att, preds, folds


def exp_a(att):
    out = {}
    regimes = (
        ("A1_random_10fold (identical wording can sit in train - optimistic)",
         KFold(10, shuffle=True, random_state=0), None),
        ("A2_leave_question_out_10fold (does NOT give unseen wording: identical working text still leaks into train - "
         "see leak_rate. The supplied data has one wording per class, so unseen-wording-seen-class cannot be measured)",
         GroupKFold(10), att.question_id.astype(str).values),
        ("A3_unseen_class_10fold (grouped by working text. Each class has ONE wording, so this REMOVES the class "
         "from training: it measures unseen classes, not unseen wording)",
         GroupKFold(10), att.student_working.astype(str).values),
    )
    for name, splitter, groups in regimes:
        a, preds, folds = _ml_cv(att, splitter, groups)
        y = a.label.tolist()
        wrong = (a.label != NONE).values
        yw = [y[i] for i in range(len(y)) if wrong[i]]
        pw = [preds[i] for i in range(len(y)) if wrong[i]]
        pr, rc, f1, _ = precision_recall_fscore_support(wrong.astype(int), np.array([int(p != NONE) for p in preds]),
                                                        average="binary", zero_division=0)
        leak, seen = [], []
        for tr, te in folds:
            texts, labs = set(a.student_working.iloc[tr].astype(str)), set(a.label.iloc[tr])
            for j in te:
                if wrong[j]:
                    leak.append(str(a.student_working.iloc[j]) in texts)
                    seen.append(a.label.iloc[j] in labs)
        out[name] = {
            "n_attempts": len(a), "n_wrong_attempts": int(wrong.sum()),
            "leak_rate_identical_text_in_train": round(float(np.mean(leak)), 3),
            "class_seen_in_train_rate": round(float(np.mean(seen)), 3),
            "misconception_accuracy_on_wrong_attempts": round(accuracy_score(yw, pw), 3),
            "accuracy_95ci": _boot_ci([int(u == v) for u, v in zip(yw, pw)]),
            "misconception_metrics_on_wrong_attempts": _prf(yw, pw, sorted(set(yw))),
            "detect_wrong_vs_not_precision": round(float(pr), 3), "detect_recall": round(float(rc), 3),
            "detect_f1": round(float(f1), 3),
            "chance_level_note": f"{len(set(yw))} classes, ~{len(yw) / max(1, len(set(yw))):.1f} wrong attempts per class "
                                 f"(per-class metrics are very noisy)",
        }
    return out


def exp_b(E):
    rows = []
    for q, w, f, gold, st in CASES:
        qq = Q(**q) if isinstance(q, dict) else E.get_q(q)
        d = E.diagnose(qq, w, f)
        pm = d["diagnosis"]["misconception_id"] if d["diagnosis"] else None
        rows.append({"q": qq.id, "gold": gold or "-", "pred": pm or "-", "status": d["status"], "expect": st,
                     "conf": d["diagnosis"]["confidence"] if d["diagnosis"] else None})
    df = pd.DataFrame(rows)
    mis = df[df.expect == "diagnosed"]
    correct_cases = df[df.expect == "correct"]
    abst = df[df.expect == "insufficient"]
    labels = sorted(set(mis.gold) | set(p for p in mis.pred if p != "-"))
    cm = confusion_matrix(mis.gold, mis.pred, labels=labels + (["-"] if "-" in set(mis.pred) else []))
    return {
        "n_cases": len(df), "note": "developer-authored typed cases written by the rule author; NOT real students and "
                                    "NOT an accuracy estimate (see experiment D and an independently written set)",
        "misconception_cases": {
            "n": len(mis), "accuracy": round(float((mis.gold == mis.pred).mean()), 3),
            "accuracy_95ci": _boot_ci((mis.gold == mis.pred).astype(int)),
            **_prf(mis.gold, mis.pred, labels),
            "confusion_labels": labels + (["-"] if "-" in set(mis.pred) else []), "confusion_matrix": cm.tolist(),
            "mean_rule_confidence_when_correct (rule constants, NOT calibrated probabilities)":
                round(float(mis[mis.gold == mis.pred].conf.mean()), 3),
        },
        "false_alarm_on_correct_working": {"n": len(correct_cases),
                                           "false_diagnoses": int((correct_cases.status == "diagnosed").sum())},
        "abstention_on_insufficient_evidence": {"n": len(abst), "correctly_abstained": int((abst.status == "insufficient").sum())},
        "failures": df[(df.gold != df.pred) | (df.status != df.expect)].to_dict("records"),
    }


def _accepts(res):
    top, pr = res["top"][0]
    return top != NONE and pr * min(1.0, res["sim"] / 0.70) >= ACCEPT_THRESHOLD, top


def exp_c(att):
    """Remove one misconception from training: how often is it still (wrongly) named? Plus a SEEN-class control."""
    wrong = att[att.misconception_id.notna()]
    forced = total = 0
    for mid in sorted(wrong.misconception_id.unique()):
        m = MLDiagnoser(att, exclude_ids=[mid])
        for r in wrong[wrong.misconception_id == mid].itertuples():
            forced += _accepts(m.predict(r.question, str(r.student_working), None, topn=1))[0]
            total += 1
    # control: same threshold, same model, but the class IS in training (10-fold random)
    a = att.reset_index(drop=True)
    acc = acc_ok = n = 0
    for tr, te in KFold(10, shuffle=True, random_state=0).split(a):
        m = MLDiagnoser(a.iloc[tr])
        for j in te:
            r = a.iloc[j]
            if pd.isna(r.misconception_id):
                continue
            ok, top = _accepts(m.predict(r.question, str(r.student_working), None, topn=1))
            acc += ok
            acc_ok += ok and top == r.misconception_id
            n += 1
    return {"unseen_misconception_attempts": total, "forced_wrong_label": forced,
            "abstained_or_none": total - forced, "abstention_rate_unseen": round((total - forced) / total, 3),
            "control_seen_class": {"n": n, "abstention_rate_seen": round(1 - acc / n, 3),
                                   "accepted_and_correct_rate_seen": round(acc_ok / n, 3),
                                   "note": "identical wording can sit in train here, so this is an UPPER bound for seen classes"},
            "reading": "Experiment C only shows useful caution if abstention_rate_unseen is clearly higher than "
                       "abstention_rate_seen. If both are high, the model simply abstains on everything.",
            "scope": "tests the TF-IDF model only, not the full engine"}


def exp_d(E=None):
    """Same wrong answer, different mechanism. Train and test on DISJOINT number ranges."""
    train = datagen.build(seed=1, roots=range(1, 6), cd=range(2, 5), xs=range(1, 7), n_neg=10, n_dist=12)
    test = datagen.build(seed=2, roots=range(6, 13), cd=range(5, 9), xs=range(7, 13), n_neg=10, n_dist=12)
    coll = [c for c in test if c["status"] in ("diagnosed", "not_label")]
    out = {
        "note": "TEMPLATED, TEAM-GENERATED cases (app/datagen.py). They show whether the working carries signal that the "
                "final answer lacks; they are not evidence about real students.",
        "n_train_cases": len(train), "n_test_collision_cases": len(coll),
        "final_answer_only_ceiling": round(datagen.final_only_bound(
            [c for c in test if c["status"] != "correct"]), 3),
    }
    for tag, fo in (("text_model_on_final_answer_only", True), ("text_model_on_full_working", False)):
        m = MLDiagnoser(datagen.to_attempts_df(train, final_only=fo))
        flags, by = [], {}
        for c in coll:
            w = c["final"] if fo else " ; ".join(c["steps"] + [c["final"]])
            ok = m.predict(c["text"], w, None, topn=1)["top"][0][0] == datagen.ml_label(c)
            flags.append(int(ok))
            by.setdefault(c["group"].split("_")[0], []).append(int(ok))
        fa = [m.predict(c["text"], c["final"] if fo else " ; ".join(c["steps"] + [c["final"]]), None, topn=1)["top"][0][0]
              != NONE for c in test if c["status"] == "correct"]
        out[tag] = {"accuracy": round(float(np.mean(flags)), 3), "accuracy_95ci": _boot_ci(flags),
                    "by_family": {k: round(float(np.mean(v)), 3) for k, v in by.items()},
                    "false_alarm_rate_on_correct_working": round(float(np.mean(fa)), 3)}
    if E is not None:
        out["full_engine_on_heldout_cases"] = datagen.score(E, test, Q)
        out["full_engine_reading"] = ("accuracy counts abstention as correct only on AMBIG/'not_label' cases; "
                                      "label_recall is the share of labelled cases named correctly")
    return out


def main():
    E = Engine()
    _, _, att, _ = load_tables()
    report = {"threshold": ACCEPT_THRESHOLD, "data_fingerprint": _fingerprint(att),
              "dataset_caveat": ("The supplied student_attempts working text narrates the mistake in words "
                                 "(e.g. 'I used a+n d instead of a+(n-1)d') and is not real mathematical working; "
                                 "templated wording is repeated across questions, and every class has a single wording. "
                                 "ML scores in A therefore measure classification of simulated explanations, not real work."),
              "A_ml_on_simulated_attempts": exp_a(att),
              "B_full_engine_on_typed_cases": exp_b(E),
              "C_unseen_misconceptions": exp_c(att),
              "D_same_wrong_answer_collisions": exp_d(E),
              "real_student_results": "None - no real student data was available."}
    OUT.mkdir(exist_ok=True)
    (OUT / "evaluation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "B_full_engine_on_typed_cases"}, indent=2))
    b = report["B_full_engine_on_typed_cases"]
    print(json.dumps({k: v for k, v in b.items() if k not in ("misconception_cases",)}, indent=2))
    print({k: v for k, v in b["misconception_cases"].items() if "confusion" not in k})


if __name__ == "__main__":
    main()

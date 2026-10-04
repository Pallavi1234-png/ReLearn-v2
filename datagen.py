"""Re:Learn - generated labelled working + SAME-WRONG-ANSWER test set  (app/datagen.py).

Each *mutator* takes a correct solution template and injects ONE named mistake, emitting step-by-step working.
Mutators in the same collision group are built so that they reach the IDENTICAL wrong final answer,
so a final-answer-only classifier cannot beat 1/k on them; only the working can separate them.

Collision families
  NEG_ROOTS  Solve x² - Sx + P = 0 -> x = -r1 or -r2, reached three different ways:
             QE-FAC-01 (wrong factor signs) / QE-SIGN-01 (root read with wrong sign from factors) / QE-FORM-02 (formula, b not -b)
  DIST       Solve c(x ± d) = N -> same wrong x, reached two ways:
             EXT-DIST-01 (multiplier applied to x only) / transposition slip (expands correctly, then subtracts d not c*d)
             + an AMBIGUOUS case (only 'cx = N-d' shown) where the right behaviour is to abstain.

Run (from backend/):
   python -m app.datagen            -> writes data/generated_cases.jsonl / .csv and self-checks the maths
   python -m app.datagen --score    -> scores the real Engine on the collision sets
These cases are TEMPLATED and TEAM-GENERATED. They are a dataset contribution and a differentiation probe,
not evidence about real students (see data/DATASET.md).
"""
import csv
import json
import random
import sys
import pandas as pd
from collections import Counter, defaultdict
from fractions import Fraction
from itertools import combinations
from pathlib import Path

DATA = Path(__file__).resolve().parents[2] / "data"


def _fr(v):
    v = Fraction(v)
    return str(v.numerator) if v.denominator == 1 else f"{v.numerator}/{v.denominator}"


def _case(group, tag, text, expected, cands, steps, final, gold, status, mech, forbid=None):
    return dict(qid=f"GEN-{group}-{tag}", group=group, text=text, expected=expected, cands=cands, steps=steps,
                final=final, gold=gold, status=status, mechanism=mech, forbid=forbid)


# ------------------------------------------------------------------ family 1: negated roots
def neg_roots(r1, r2):
    S, P, D = r1 + r2, r1 * r2, (r2 - r1) ** 2
    g = f"NEG_ROOTS_{r1}_{r2}"
    text = f"Solve x² - {S}x + {P} = 0."
    exp, bad = f"x = {r1} or x = {r2}", f"x = -{r1} or x = -{r2}"
    eq = f"x² - {S}x + {P} = 0"
    cands = ["QE-FAC-01", "QE-SIGN-01"]
    zero = f"x - {r1} = 0 or x - {r2} = 0"
    return [
        _case(g, "FAC", text, exp, cands, [eq, f"(x + {r1})(x + {r2}) = 0", f"x + {r1} = 0 or x + {r2} = 0"], bad,
              "QE-FAC-01", "diagnosed", "factor pair given the wrong signs, roots then read correctly"),
        _case(g, "SIGN", text, exp, cands, [eq, f"(x - {r1})(x - {r2}) = 0", zero], bad,
              "QE-SIGN-01", "diagnosed", "factorised correctly, then wrote x = -p from (x - p)"),
        _case(g, "FORM", text, exp, cands, ["a = 1, b = " + str(-S) + ", c = " + str(P),
                                            f"D = b² - 4ac = {S * S} - {4 * P} = {D}",
                                            f"x = (b ± √D)/(2a) = ({-S} ± {r2 - r1})/2"], bad,
              "QE-FORM-02", "diagnosed", "quadratic formula written with b instead of -b"),
        _case(g, "OK", text, exp, cands, [eq, f"(x - {r1})(x - {r2}) = 0", zero], exp, None, "correct",
              "correct working"),
    ]


# ------------------------------------------------------------------ family 2: distribution vs transposition slip
def dist(c, d, s, x0):
    sg, sd = ("+" if s > 0 else "-"), s * d
    N = c * (x0 + sd)
    g = f"DIST_{c}_{sg}{d}_{x0}"
    q_eq = f"{c}(x {sg} {d}) = {N}"
    text, exp = f"Solve {q_eq}", f"x = {x0}"
    xw = Fraction(N - sd, c)
    bad = f"x = {_fr(xw)}"
    rhs_bad = f"{c}x = {N - sd}"
    return [
        _case(g, "DIST", text, exp, ["EXT-DIST-01"], [q_eq, f"{c}x {sg} {d} = {N}", rhs_bad], bad,
              "EXT-DIST-01", "diagnosed", "multiplier applied to x only"),
        _case(g, "SLIP", text, exp, ["EXT-DIST-01"], [q_eq, f"{c}x {sg} {c * d} = {N}", rhs_bad], bad,
              None, "not_label", "expanded correctly, then moved d instead of c*d (transposition slip)",
              forbid="EXT-DIST-01"),
        _case(g, "AMBIG", text, exp, ["EXT-DIST-01"], [q_eq, rhs_bad], bad, None, "insufficient",
              "only 'cx = N-d' shown: consistent with BOTH mechanisms -> must abstain / ask for next step"),
        _case(g, "OK", text, exp, ["EXT-DIST-01"],
              [q_eq, f"{c}x {sg} {c * d} = {N}", f"{c}x = {N - s * c * d}"], exp, None, "correct", "correct working"),
    ]


def build(seed=7, roots=range(1, 10), cd=range(2, 7), xs=range(1, 9), n_neg=10, n_dist=12):
    """Seedable; pass disjoint number pools (roots / cd / xs) to make a HELD-OUT test set."""
    rng = random.Random(seed)
    cases = []
    for r1, r2 in rng.sample(list(combinations(roots, 2)), n_neg):
        cases += neg_roots(r1, r2)
    seen = set()
    while len(seen) < n_dist:
        c, d, s = rng.choice(list(cd)), rng.choice(list(cd)), rng.choice((1, -1))
        pool = [v for v in xs if s > 0 or v > d]
        if not pool:
            continue
        key = (c, d, s, rng.choice(pool))
        if key not in seen:
            seen.add(key)
            cases += dist(*key)
    return cases


def ml_label(c):
    """Label used when these cases train / test the text model (None = correct working, AMBIG excluded)."""
    if c["status"] == "diagnosed":
        return c["gold"]
    if c["status"] == "not_label":
        return "TRANSPOSITION-SLIP"
    return None


def to_attempts_df(cases, final_only=False):
    """Bridge to the supplied student_attempts schema so ml.MLDiagnoser can train on these cases."""
    rows = []
    for c in cases:
        if c["status"] == "insufficient":
            continue
        work = c["final"] if final_only else " ; ".join(c["steps"] + [c["final"]])
        rows.append(dict(attempt_id=c["qid"], question=c["text"], student_working=work, misconception_id=ml_label(c)))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ self-checks (the maths, not the engine)
def verify(cases):
    by = defaultdict(list)
    for c in cases:
        by[c["group"]].append(c)
    for g, cs in by.items():
        wrong = [c for c in cs if c["status"] != "correct"]
        assert len({c["final"] for c in wrong}) == 1, f"{g}: wrong members do not share one final answer"
        ok = next(c for c in cs if c["status"] == "correct")
        assert wrong[0]["final"] != ok["final"], f"{g}: wrong answer equals correct answer"
        if g.startswith("NEG_ROOTS"):
            r1, r2 = (int(v) for v in g.split("_")[2:])
            S, P = r1 + r2, r1 * r2
            assert (-S + (r2 - r1)) / 2 == -r1 and (-S - (r2 - r1)) / 2 == -r2            # formula member
            assert (r1 + r2, r1 * r2) == (S, P)                                              # (x-r1)(x-r2) is right
            assert (r1 + r2, r1 * r2) != (-S, P)                                             # (x+r1)(x+r2) is not
        else:
            _, c_, sd, x0 = g.split("_")
            c_, s, d, x0 = int(c_), (1 if sd[0] == "+" else -1), int(sd[1:]), int(x0)
            N = c_ * (x0 + s * d)
            xw = Fraction(N - s * d, c_)
            assert c_ * xw + s * d == N and xw != x0                                         # wrong answer solves the WRONG line


# ------------------------------------------------------------------ scoring
def final_only_bound(cases):
    """Best possible accuracy of ANY classifier that sees only (question, final answer)."""
    by = defaultdict(Counter)
    for c in cases:
        by[(c["text"], c["final"])][c["gold"] or c["status"]] += 1
    return sum(max(v.values()) for v in by.values()) / len(cases)


def judge(c, status, pred):
    if c["status"] == "diagnosed":
        return status == "diagnosed" and pred == c["gold"]
    if c["status"] == "insufficient":
        return status == "insufficient"
    if c["status"] == "correct":
        return status != "diagnosed"
    return not (status == "diagnosed" and pred == c["forbid"])          # not_label


def score(engine, cases, Q):
    res = defaultdict(lambda: Counter())
    for c in cases:
        fam = c["group"].split("_")[0]
        d = engine.diagnose(Q(id=c["qid"], text=c["text"], expected=c["expected"], candidates=c["cands"]),
                            c["steps"], c["final"])
        st, pred = d["status"], (d["diagnosis"]["misconception_id"] if d.get("diagnosis") else None)
        r = res[fam]
        r["n"] += 1
        r["correct"] += judge(c, st, pred)
        r["abstained"] += st == "insufficient"
        r["confidently_wrong"] += st == "diagnosed" and not judge(c, st, pred)
        if c["status"] == "diagnosed":                      # abstaining must not look like success
            r["label_cases"] += 1
            r["label_hits"] += judge(c, st, pred)
    out = {}
    for fam, r in res.items():
        coll = [c for c in cases if c["group"].startswith(fam) and c["status"] != "correct"]
        out[fam] = dict(r, accuracy=round(r["correct"] / r["n"], 3),
                        label_recall=round(r["label_hits"] / max(1, r["label_cases"]), 3),
                        final_answer_only_upper_bound_on_collision_cases=round(final_only_bound(coll), 3))
    return out


def main():
    cases = build()
    verify(cases)
    here = DATA
    here.mkdir(exist_ok=True)
    with open(here / "generated_cases.jsonl", "w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    with open(here / "generated_cases.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "group", "question", "student_steps", "final_answer", "gold_misconception_id",
                    "expected_status", "mechanism", "must_not_be"])
        for c in cases:
            w.writerow([c["qid"], c["group"], c["text"], " | ".join(c["steps"]), c["final"], c["gold"] or "",
                        c["status"], c["mechanism"], c["forbid"] or ""])
    print(f"{len(cases)} cases, maths self-check passed")
    print(Counter(c["status"] for c in cases))
    for fam in ("NEG", "DIST"):
        coll = [c for c in cases if c["group"].startswith(fam) and c["status"] != "correct"]
        print(fam, "collision cases:", len(coll), "| final-answer-only upper bound:", round(final_only_bound(coll), 3))
    if "--score" in sys.argv:
        from .diagnose import Engine
        from .rules import Q
        print(json.dumps(score(Engine(), cases, Q), indent=2))


if __name__ == "__main__":
    main()

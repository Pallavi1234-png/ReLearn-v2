"""Secondary diagnosis signal: TF-IDF + Logistic Regression trained on the supplied student_attempts.csv.

Honest limits (also reported by evaluate.py): ~2 wrong attempts per misconception and many attempts reuse
the same generic working text, so this model is a weak, explainable baseline - never the only signal.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import linear_kernel

DATA = Path(__file__).resolve().parents[2] / "data"


def load_tables():
    q = pd.read_csv(DATA / "questions.csv")
    m = pd.read_csv(DATA / "misconceptions.csv")
    a = pd.read_csv(DATA / "student_attempts.csv")
    i = pd.read_csv(DATA / "interventions.csv")
    if os.environ.get("RELEARN_USE_GENERATED") == "1":
        # OPT-IN: add the team-generated, labelled maths working (app/datagen.py). Off by default so the shipped
        # behaviour only changes when you choose it and have re-run evaluate.py.
        from .datagen import build, to_attempts_df
        g = to_attempts_df(build())
        a = pd.concat([a, g.reindex(columns=a.columns)], ignore_index=True)
    return q, m, a, i


NONE = "__none__"


def _doc(question, working):
    return f"{question} || {working}"


class MLDiagnoser:
    def __init__(self, attempts: pd.DataFrame, C: float = 20.0, exclude_ids=None):
        # Trained on ALL attempts: wrong ones keep their misconception_id; correct / no-working ones get NONE,
        # so the model can say "no misconception visible" instead of being forced to pick one.
        att = attempts.copy()
        att["label"] = att.misconception_id.fillna(NONE)
        if exclude_ids:
            att = att[~att.label.isin(exclude_ids)]
        self.train = att.reset_index(drop=True)
        docs = [_doc(r.question, r.student_working) for r in self.train.itertuples()]
        self.word = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, token_pattern=r"[^\s,;.=()]+")
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        Xw = self.word.fit_transform(docs)
        Xc = self.char.fit_transform(docs)
        self.X = hstack([Xw, Xc]).tocsr()
        self.y = self.train.label.values
        self.clf = LogisticRegression(C=C, max_iter=3000)
        self.clf.fit(self.X, self.y)
        self.classes = list(self.clf.classes_)

    def _vec(self, question, working):
        d = [_doc(question, working)]
        return hstack([self.word.transform(d), self.char.transform(d)]).tocsr()

    def predict(self, question, working, candidates=None, topn=3):
        """Returns dict(top=[(id, prob)], sim=float, nearest=attempt_id, terms=[...]).
        prob is the *unrenormalised* probability from the full model, so a single-candidate question
        cannot turn into 100% confidence."""
        v = self._vec(question, working)
        p = self.clf.predict_proba(v)[0]
        order = np.argsort(-p)
        ranked = [(self.classes[j], float(p[j])) for j in order]
        if candidates:
            cand_ranked = [(c, pr) for c, pr in ranked if c in candidates or c == NONE]
            ranked_use = cand_ranked or ranked
        else:
            ranked_use = ranked
        sims = linear_kernel(v, self.X)[0]
        k = int(np.argmax(sims))
        terms = []
        if ranked_use:
            cls = ranked_use[0][0]
            ci = self.classes.index(cls)
            vw = self.word.transform([_doc(question, working)])
            coefs = self.clf.coef_[ci][:len(self.word.vocabulary_)]
            inv = {j: t for t, j in self.word.vocabulary_.items()}
            contrib = vw.multiply(coefs).tocoo()
            top = sorted(zip(contrib.col, contrib.data), key=lambda z: -z[1])[:4]
            terms = [inv[j] for j, w in top if w > 0 and "||" not in inv[j]]
        return {"top": ranked_use[:topn], "sim": float(sims[k]),
                "nearest": str(self.train.attempt_id.iloc[k]), "terms": terms}

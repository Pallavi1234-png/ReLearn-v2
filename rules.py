"""Rule-based misconception detectors.

Each rule inspects the question + the student's steps and, if it finds an *observable* error
pattern, returns a Finding with a `kind`. A kind is later mapped to a misconception_id from the
supplied taxonomy (see KIND_TO_IDS). Rules never guess: if the pattern is not visible they return nothing.
"""
import re
from dataclasses import dataclass, field
from typing import List, Optional

import sympy as sp

from .mathparse import (normalize, parse_expr_safe, parse_equation, eq_poly, equations_equivalent,
                        root_set, sets_equal, numbers_in, last_number, extract_equation, poly_coeffs,
                        is_fully_factored, parse_struct, x)


@dataclass
class Q:
    id: str
    text: str
    expected: str = ""
    candidates: List[str] = field(default_factory=list)
    topic: str = ""
    chapter: str = ""
    standard: int = 0


@dataclass
class Finding:
    kind: str
    step: int            # 1-based step index, 0 = final answer
    evidence: str
    expected_step: str = ""
    confidence: float = 0.85
    rule: str = ""
    # NEW: other kinds that are equally consistent with the evidence, and where the evidence came from
    alternatives: List[str] = field(default_factory=list)
    evidence_level: str = "final_only"      # "steps" = read from the working; "final_only" = inferred from the answer


# kind -> preferred misconception ids (first is the default; question candidates break ties)
KIND_TO_IDS = {
    "DISTRIBUTION": ["EXT-DIST-01"],
    "ROOT_SIGN_FROM_FACTOR": ["QE-SIGN-01"],
    "QUAD_FACTORISATION_WRONG": ["QE-FAC-01"],
    "FORMULA_SIGN_B": ["QE-FORM-02"],
    "FORMULA_DENOMINATOR": ["QE-FORM-02"],
    "FORMULA_SUBSTITUTION": ["QE-FORM-01", "QE-SUB-01"],
    "DISC_PLUS": ["QE-DISC-01"],
    "DISC_B_SQUARE_SIGN": ["QE-DISC-01", "QE-SIGN-02"],
    "DISC_C_SIGN": ["QE-DISC-01", "QE-SIGN-02"],
    "VIETA_SUM_SIGN": ["QE-VIETA-01"],
    "VIETA_PRODUCT": ["QE-VIETA-02"],
    "FAC_DS_AS_SQUARE": ["FAC-ID-01"],
    "FAC_DS_CONSTANT_NOT_ROOT": ["FAC-ID-01"],
    "FAC_PS_AS_DS": ["FAC-ID-02"],
    "FAC_PS_SIGN": ["FAC-SIGN-01", "FAC-ID-03", "FAC-ID-02"],
    "FAC_MT_SUM": ["FAC-MT-01"],
    "FAC_MT_SIGN": ["FAC-MT-02"],
    "FAC_CF_WRONG_REMAINDER": ["FAC-CF-01"],
    "FAC_INCOMPLETE_CF": ["FAC-CF-01"],
    "FAC_INCOMPLETE": ["FAC-MT-04"],
    "AP_NTH_N_NOT_N1": ["AP-FORM-01", "AP-NT-01", "AP-NT-02"],
    "AP_FIRST_TERM_WRONG": ["AP-NT-02"],
    "AP_COUNT_OFF_BY_ONE": ["AP-NT-04"],
    "AP_SUM_N_NOT_N1": ["AP-SUM-01"],
    "AP_SUM_NO_HALF": ["AP-SUM-01"],
    "AP_NTH_FOR_SUM": ["AP-NT-05", "AP-SUM-02"],
    "AP_D_REVERSED": ["AP-D-01"],
    "AP_D_RATIO": ["AP-CON-02"],
    "EXP_MULTIPLY": ["RN-EXP-01"],
    "EXP_DIV_WRONG": ["RN-EXP-02"],
    "SURD_ADD_RADICANDS": ["RN-SURD-02"],
    "SURD_NOT_SIMPLIFIED": ["RN-SURD-01"],
    "ABS_ONE_SOLUTION": ["RN-ABS-01"],
    "ABS_KEEPS_SIGN": ["RN-ABS-01"],
    "ARITH_SLIP": ["QE-SUB-01"],
}


def resolve_id(kind, candidates):
    prefs = KIND_TO_IDS.get(kind, [])
    for c in candidates or []:
        if c in prefs:
            return c
    return prefs[0] if prefs else None


AMBIGUOUS_CONF = 0.55   # below the 0.70 acceptance threshold -> the engine abstains and asks for the next step


def mark_ambiguous(f: "Finding", alt_kinds):
    """Record that other mistakes give the same evidence. If any of them maps to a DIFFERENT misconception id,
    the finding is capped below the acceptance threshold, so the engine asks for more working instead of
    silently picking the first rule that matched."""
    own = resolve_id(f.kind, [])
    f.alternatives = list(alt_kinds)
    if any((resolve_id(k, []) or k) != own for k in alt_kinds):
        f.confidence = min(f.confidence, AMBIGUOUS_CONF)
        f.evidence += " Several different mistakes give this same result, so please show the next line of your working."
    return f


class Ctx:
    def __init__(self, q: Q, steps: List[str], final: str):
        self.q, self.steps, self.final = q, steps, final
        self.qn = normalize(q.text)
        self.qlow = q.text.lower()
        self.eq_text = extract_equation(q.text)


def _same(e1, e2):
    return sp.simplify(e1 - e2) == 0


def _fmt(v):
    return str(sp.nsimplify(v)).replace("**", "^").replace("*", "")


# ------------------------------------------------------------------ distribution
_DIST = re.compile(r"(-?\d*)\s*\(\s*x\s*([+-])\s*(\d+)\s*\)")


def rule_distribution(c: Ctx) -> List[Finding]:
    lines = [("question", c.eq_text or c.q.text)] + [(i + 1, s) for i, s in enumerate(c.steps)]
    out = []
    for li, (pos, raw) in enumerate(lines):
        t = normalize(raw)
        m = _DIST.search(t)
        if not m or m.group(1) in ("",):
            continue
        coef = -1 if m.group(1) == "-" else int(m.group(1))
        sign, d = m.group(2), int(m.group(3))
        wrong_piece = f"{coef}*x{sign}{d}"
        wrong_line = t[:m.start()] + "(" + wrong_piece + ")" + t[m.end():]
        eq_wrong = parse_equation(wrong_line)
        eq_orig = parse_equation(t)
        e_wrong = eq_poly(eq_wrong) if eq_wrong else None
        e_orig = eq_poly(eq_orig) if eq_orig else None
        for j in range(li + 1, len(lines)):
            pj, rawj = lines[j]
            eq_j = parse_equation(rawj)
            if not eq_j or e_wrong is None:
                continue
            pj_poly = eq_poly(eq_j)
            if _same(pj_poly, e_wrong) and not _same(pj_poly, e_orig):
                signed = f"{coef}x {sign} {abs(coef * d)}" if sign == "+" else f"{coef}x - {abs(coef * d)}"
                # Was the wrong expansion actually WRITTEN (e.g. "3x + 2 = 15")? If the student jumped straight to
                # a reduced line (e.g. "3x = 13") a transposition slip gives the identical line, so we cannot claim it.
                cs = "-" if coef == -1 else str(coef)
                shown_rx = re.compile(re.escape(cs) + r"\*?x" + re.escape(sign) + str(d) + r"(?!\d)")
                shown = bool(shown_rx.search(rawj.replace(" ", "")) or shown_rx.search(normalize(rawj).replace(" ", "")))
                if shown:
                    f = Finding(
                        "DISTRIBUTION", pj if isinstance(pj, int) else 1,
                        f"The multiplier {coef} was applied to x but not to the constant {d} in "
                        f"{coef}(x {sign} {d}). Distribution must give {signed}.",
                        expected_step=f"{coef}(x {sign} {d}) = {signed}", confidence=0.95, rule="distribution",
                        evidence_level="steps")
                else:
                    f = Finding(
                        "DISTRIBUTION", pj if isinstance(pj, int) else 1,
                        f"Line {pj} matches {coef}(x {sign} {d}) being expanded as if the multiplier only reached x, "
                        f"but the expansion itself was not written down.",
                        expected_step=f"{coef}(x {sign} {d}) = {signed}", confidence=0.95, rule="distribution",
                        evidence_level="partial")
                    mark_ambiguous(f, ["TRANSPOSITION_SLIP"])
                out.append(f)
                return out
            break
    return out


# ------------------------------------------------------------------ quadratics
def _is_formula_q(c):
    return "quadratic formula" in c.qlow


def rule_quadratic(c: Ctx) -> List[Finding]:
    q = c.q
    out = []
    coeffs = poly_coeffs(c.eq_text) if c.eq_text else None
    if (not coeffs or "discriminant" in c.qlow or "nature" in c.qlow
            or ("roots of" in c.qlow and ("sum" in c.qlow or "product" in c.qlow))):
        return out
    if not c.qlow.strip().startswith(("solve", "find the roots", "find roots")):
        return out
    a, b, cc = coeffs
    D = b * b - 4 * a * cc
    if D < 0:
        return out
    true_roots = root_set(f"{(-b + sp.sqrt(D)) / (2 * a)}, {(-b - sp.sqrt(D)) / (2 * a)}")
    student = root_set(c.final) if c.final else None

    # The route is read from the WORKING first (a step with "±"), and only then from the prompt wording.
    formula_route = _is_formula_q(c) or any(("±" in s_) or ("+-" in s_) for s_ in c.steps)
    if formula_route:
        if not student or true_roots is None or sets_equal(student, true_roots):
            return out
        # step-level evidence: the formula needs -b, but the substituted numerator starts with b itself
        for i_, s_ in enumerate(c.steps):
            m_ = re.search(r"\(\s*(-?\d+)\s*(?:±|\+-)", s_.replace("−", "-"))
            if m_ and b != 0 and int(m_.group(1)) == int(b):
                out.append(Finding("FORMULA_SIGN_B", i_ + 1,
                                   f"Step {i_ + 1} puts {int(b)} in the numerator, but the formula needs -b = {int(-b)}.",
                                   expected_step=f"x = (-b ± √D)/(2a) = ({int(-b)} ± √{D})/{2 * a}",
                                   confidence=0.93, rule="quad_formula", evidence_level="steps"))
                return out
        sq = sp.sqrt(D)
        alts = {
            "FORMULA_SIGN_B": ((b + sq) / (2 * a), (b - sq) / (2 * a),
                               f"b was used instead of -b (got x = ({b} ± √{D})/{2 * a})"),
            "FORMULA_DENOMINATOR": ((-b + sq) / a, (-b - sq) / a,
                                    f"the denominator was {a} instead of 2a = {2 * a}"),
        }
        for kind, (r1, r2, msg) in alts.items():
            if sets_equal(student, {sp.nsimplify(r1), sp.nsimplify(r2)}):
                out.append(Finding(kind, 0, f"Final roots match this mistake: {msg}.",
                                   confidence=0.9, rule="quad_formula"))
                return out
        # (losing the sign of c, b² - 4a|c|, equals D when c > 0 and equals b² + 4ac when c < 0, so it is
        #  not a separate, reachable pattern here)
        for dd, msg in ((b * b + 4 * a * cc,
                         f"the discriminant was computed as b²+4ac = {b * b + 4 * a * cc} (or the sign of c was lost)"),):
            if dd >= 0 and dd != D:
                sq2 = sp.sqrt(dd)
                if sets_equal(student, {sp.nsimplify((-b + sq2) / (2 * a)), sp.nsimplify((-b - sq2) / (2 * a))}):
                    out.append(Finding("FORMULA_SUBSTITUTION", 0, f"Final roots match this mistake: {msg}.",
                                       confidence=0.88, rule="quad_formula"))
                    return out
        return out

    # factorisation route: look for a factored step "(x-2)(x-3)=0"
    fac_step = None
    for i, s in enumerate(c.steps):
        eq = parse_equation(s)
        t = normalize(s)
        if eq and eq[1] == 0 and re.search(r"\)\s*\(", t) or (eq and eq[1] == 0 and ")*(" in t):
            fac_step = (i + 1, t)
    if fac_step:
        i, t = fac_step
        lhs_txt = t.split("=")[0]
        raw_lhs = parse_expr_safe(lhs_txt.replace(" ", ""))
        # evaluate=True expands already; re-parse unevaluated factors for root reading
        factors = re.findall(r"\(([^()]+)\)", lhs_txt)
        poly_orig = sp.expand(a * x ** 2 + b * x + cc)
        if raw_lhs is not None:
            ratio = sp.simplify(sp.expand(raw_lhs) / poly_orig) if poly_orig != 0 else None
            if ratio is None or not ratio.is_number or ratio == 0:
                out.append(Finding("QUAD_FACTORISATION_WRONG", i,
                                   f"Your factors expand to {_fmt(sp.expand(raw_lhs))}, "
                                   f"but the equation is {_fmt(poly_orig)} = 0.",
                                   expected_step=f"{_fmt(sp.factor(poly_orig))} = 0", confidence=0.9,
                                   rule="quad_factor"))
                return out
        # factorisation is right -> read roots off the factors
        claimed = []
        for f in factors:
            fe = parse_expr_safe(f)
            if fe is not None and fe.has(x):
                claimed += sp.solve(fe, x)
        if claimed and student and not sets_equal(student, set(sp.nsimplify(r) for r in claimed)):
            ok = set(sp.nsimplify(r) for r in claimed)
            flips = all(any(sp.simplify(s - r) == 0 or sp.simplify(s + r) == 0 for r in ok) for s in student)
            if flips:
                shown = " or ".join(f"x = {_fmt(r)}" for r in sorted(ok, key=lambda v: float(v)))
                out.append(Finding("ROOT_SIGN_FROM_FACTOR", 0,
                                   f"From {t.strip()} the roots are {shown}, but the answer given is "
                                   f"{c.final.strip()} — the sign was flipped when reading a root from a factor.",
                                   confidence=0.92, rule="quad_factor"))
    elif student and true_roots and not sets_equal(student, true_roots) and c.steps:
        neg = set(-r for r in true_roots)
        if sets_equal(student, neg):
            out.append(mark_ambiguous(Finding(
                "ROOT_SIGN_FROM_FACTOR", 0,
                "Every root has the opposite sign to the correct root. That can come from reading x = +p from (x + p), "
                "from a wrong sign inside the factors, or from dropping the minus on b in the formula.",
                confidence=0.65, rule="quad_signs_only"),
                ["QUAD_FACTORISATION_WRONG", "FORMULA_SIGN_B"]))
    return out


def rule_discriminant(c: Ctx) -> List[Finding]:
    if "discriminant" not in c.qlow or not c.eq_text:
        return []
    co = poly_coeffs(c.eq_text)
    sval = last_number(c.final) if c.final else None
    if not co or sval is None:
        return []
    a, b, cc = co
    D = b * b - 4 * a * cc
    if sp.Rational(sval.numerator, sval.denominator) == D:
        return []
    s = sp.Rational(sval.numerator, sval.denominator)
    opts = [
        ("DISC_PLUS", b * b + 4 * a * cc, f"used b² + 4ac = {b * b + 4 * a * cc} instead of b² − 4ac"),
        ("DISC_C_SIGN", b * b - 4 * a * abs(cc), "lost the sign of c while computing −4ac"),
        ("DISC_B_SQUARE_SIGN", -b * b - 4 * a * cc, "treated b² as negative (e.g. −7² instead of (−7)²)"),
    ]
    hits = [(kind, msg) for kind, val, msg in opts if val != D and s == val]
    if not hits:
        return []
    kind, msg = hits[0]
    plus_shown = any(re.search(r"(?:b\s*(?:²|\^2)|\(\s*-?\d+\s*\)\s*(?:²|\^2))\s*\+\s*4", st.replace("−", "-"))
                     for st in c.steps)
    if plus_shown and any(k == "DISC_PLUS" for k, _ in hits):
        return [Finding("DISC_PLUS", 0, f"Your working shows b² + 4ac; the discriminant is b² − 4ac "
                                         f"(so D = {D}, not {s}).", confidence=0.93, rule="discriminant",
                        evidence_level="steps")]
    also = "".join(f", or if you {m2}" for _, m2 in hits[1:])
    return [Finding(kind, 0, f"The value {s} is what you get if you {msg}{also}; correct D = {D}.",
                    confidence=0.88, rule="discriminant", alternatives=[k for k, _ in hits[1:]])]


def rule_vieta(c: Ctx) -> List[Finding]:
    if not c.eq_text or not ("sum" in c.qlow or "product" in c.qlow or "α" in c.q.text):
        return []
    co = poly_coeffs(c.eq_text)
    nums = numbers_in(c.final or "")
    if not co or not nums:
        return []
    a, b, cc = co
    S, P = -b / a, cc / a
    fr = lambda v: sp.Rational(v.numerator, v.denominator)
    out = []
    want_sum = "sum" in c.qlow or "α+β" in c.qlow or "α + β" in c.qlow
    want_prod = "product" in c.qlow or "αβ" in c.q.text
    if want_sum and want_prod and len(nums) >= 2:
        s_given, p_given = fr(nums[0]), fr(nums[1])
    elif want_sum:
        s_given, p_given = fr(nums[-1]), None
    else:
        s_given, p_given = None, fr(nums[-1])
    if s_given is not None and s_given != S and s_given == b / a:
        out.append(Finding("VIETA_SUM_SIGN", 0,
                           f"Sum of roots is −b/a = {S}, but the answer is b/a = {b / a}: the minus sign was dropped.",
                           confidence=0.9, rule="vieta"))
    if p_given is not None and p_given != P and (p_given == -P or (cc != 0 and p_given == a / cc)):
        out.append(Finding("VIETA_PRODUCT", 0,
                           f"Product of roots is c/a = {P}, but the answer given is {p_given}.",
                           confidence=0.85, rule="vieta"))
    return out


# ------------------------------------------------------------------ factorisation (Std 9)
def _factorise_target(c: Ctx):
    m = re.match(r"\s*factori[sz]e\s+(.+?)(?:\s+by\s+|\s+and\s+state|\.\s*$|\.$|$)", c.q.text.strip(), re.I)
    if not m:
        return None
    return parse_expr_safe(m.group(1).strip().rstrip("."))


def _student_expr(final):
    if not final:
        return None
    t = final.split(";")[0]
    t = re.sub(r"^\s*[a-zA-Z]*\s*=", "", t) if "=" in t else t
    return parse_expr_safe(t.strip())


def _student_struct(final):
    if not final:
        return None
    t = final.split(";")[0]
    t = re.sub(r"^\s*[a-zA-Z]*\s*=", "", t) if "=" in t else t
    return parse_struct(t.strip())


def rule_factorise(c: Ctx) -> List[Finding]:
    P = _factorise_target(c)
    S = _student_expr(c.final)
    Ss = _student_struct(c.final)
    if P is None or S is None or Ss is None:
        return []
    P = sp.expand(P)
    if sp.expand(S - P) == 0:
        if not is_fully_factored(Ss):
            cf_left = False   # a common factor is still inside a bracket, e.g. 2(3x+6)
            for f in sp.Mul.make_args(Ss):
                base = f.base if f.is_Pow else f
                if base.is_Add and abs(sp.factor_list(base)[0]) != 1:
                    cf_left = True
            kind = "FAC_INCOMPLETE_CF" if cf_left else "FAC_INCOMPLETE"
            return [Finding(kind, 0, f"{c.final.strip()} multiplies back correctly but is not fully factorised "
                                     f"(complete factorisation: {_fmt(sp.factor(P))}).", confidence=0.85,
                            rule="factorise")]
        return []
    true_f = sp.factor(P)
    shown = _fmt(true_f)
    ev_tail = f" Correct factorisation: {shown}."
    # difference of squares written as a square
    if Ss.is_Pow and Ss.exp == 2 and true_f.is_Mul and len(true_f.args) == 2 and P.is_Add and len(P.args) == 2:
        return [Finding("FAC_DS_AS_SQUARE", 0,
                        f"Difference of squares a²−b² = (a−b)(a+b); your {c.final.strip()} is a perfect square "
                        f"and expands to {_fmt(sp.expand(S))}.", confidence=0.9, rule="factorise")]
    # constant used instead of its square root
    if true_f.is_Mul and len(true_f.args) == 2 and P.is_Add:
        Pp = sp.Poly(P, x) if len(P.free_symbols) == 1 else None
        if Pp and Pp.degree() == 2 and Pp.nth(1) == 0:
            const = abs(Pp.nth(0))
            k = int(sp.sqrt(const)) if const.is_Integer and sp.sqrt(const).is_Integer else None
            if k:
                wrong = sp.expand((x - const) * (x + const))
                if sp.expand(S - wrong) == 0:
                    return [Finding("FAC_DS_CONSTANT_NOT_ROOT", 0,
                                    f"Used {const} instead of its square root {k} in a²−b² = (a−b)(a+b)."
                                    + ev_tail, confidence=0.9, rule="factorise")]
    # perfect square trinomial
    fl = sp.factor_list(P)
    if len(fl[1]) == 1 and fl[1][0][1] == 2:
        base = fl[1][0][0]
        if Ss.is_Mul and len(Ss.args) == 2:   # (A+B)(A-B)
            return [Finding("FAC_PS_AS_DS", 0, f"{c.final.strip()} uses the difference-of-squares pattern, "
                                               f"but this is a perfect-square trinomial." + ev_tail,
                            confidence=0.85, rule="factorise")]
        if Ss.is_Pow and Ss.exp == 2:
            return [Finding("FAC_PS_SIGN", 0, f"The sign inside the bracket does not match the middle term "
                                              f"(a±b)² = a² ± 2ab + b²." + ev_tail, confidence=0.85,
                            rule="factorise")]
    # splitting the middle term
    Pp = sp.Poly(P, x) if len(P.free_symbols) == 1 else None
    if Pp and Pp.degree() == 2 and Ss.is_Mul:
        Sp = sp.Poly(sp.expand(S), x)
        if Sp.degree() == 2 and Sp.nth(2) == Pp.nth(2) and Sp.nth(0) == Pp.nth(0):
            if Sp.nth(1) == -Pp.nth(1):
                return [Finding("FAC_MT_SIGN", 0,
                                f"Your factors have the right product but the wrong signs: they add to "
                                f"{Sp.nth(1)}x instead of {Pp.nth(1)}x." + ev_tail, confidence=0.9,
                                rule="factorise")]
            return [Finding("FAC_MT_SUM", 0,
                            f"Your factors give middle term {Sp.nth(1)}x, not {Pp.nth(1)}x: the pair must "
                            f"multiply to the constant AND add to the middle coefficient." + ev_tail,
                            confidence=0.88, rule="factorise")]
        if Sp.degree() == 2 and Sp.nth(0) != Pp.nth(0):
            return [Finding("FAC_MT_SUM", 0, f"Your factors expand to {_fmt(sp.expand(S))}; the constant term "
                                             f"should be {Pp.nth(0)}." + ev_tail, confidence=0.7,
                            rule="factorise")]
    # common factor taken out wrongly
    if sp.factor_terms(P) != P and Ss.is_Mul:
        return [Finding("FAC_CF_WRONG_REMAINDER", 0,
                        f"{c.final.strip()} expands to {_fmt(sp.expand(S))}, not {_fmt(P)}: after taking out the "
                        f"common factor every term must be divided by it." + ev_tail, confidence=0.8,
                        rule="factorise")]
    return []


# ------------------------------------------------------------------ AP
_SEQ = re.compile(r"((?:-?\d+\s*,\s*){2,})(?:\.\.\.|…)(?:\s*,\s*(-?\d+))?")


def _ap(c: Ctx):
    t = normalize(c.q.text)
    m = _SEQ.search(t)
    if not m:
        return None
    nums = [int(v) for v in re.findall(r"-?\d+", m.group(1))]
    if len(nums) < 2:
        return None
    d = nums[1] - nums[0]
    if any(nums[i + 1] - nums[i] != d for i in range(len(nums) - 1)):
        return None
    last = int(m.group(2)) if m.group(2) else None
    return nums[0], d, last


def rule_ap(c: Ctx) -> List[Finding]:
    t, low = normalize(c.q.text), c.qlow
    sval = last_number(c.final) if c.final else None
    S = sp.Rational(sval.numerator, sval.denominator) if sval is not None else None
    ap = _ap(c)
    out = []
    # first term from known term
    m = re.search(r"(\d+)(?:st|nd|rd|th) term of an a\.?p\.? is (-?\d+) and d\s*=\s*(-?\d+)", t, re.I)
    if m and "first term" in low and S is not None:
        n, T, d = int(m[1]), int(m[2]), int(m[3])
        a = T - (n - 1) * d
        if S != a and S in (T - n * d, T + (n - 1) * d):
            how = "subtracted n·d instead of (n−1)·d" if S == T - n * d else "added (n−1)d instead of subtracting it"
            out.append(Finding("AP_FIRST_TERM_WRONG", 0,
                               f"From a + ({n}−1)·{d} = {T}, the first term is {a}; you {how} (got {S}).",
                               confidence=0.88, rule="ap"))
        return out
    if not ap or S is None:
        return out
    a, d, last = ap
    # common difference
    if "common difference" in low:
        if S != d and S == -d:
            out.append(Finding("AP_D_REVERSED", 0, f"d = (later term) − (earlier term) = {d}; "
                                                    f"you subtracted in the wrong order.", confidence=0.9, rule="ap"))
        elif S != d and a != 0 and S == sp.Rational(a + d, a):
            out.append(Finding("AP_D_RATIO", 0, "You divided consecutive terms (common ratio) instead of "
                                                "subtracting (common difference).", confidence=0.9, rule="ap"))
        return out
    mn = re.search(r"(\d+)(?:st|nd|rd|th) term", t)
    if mn and "sum" not in low and "first term" not in low and "is" not in low.split("term")[0][-8:]:
        n = int(mn.group(1))
        true = a + (n - 1) * d
        if S != true and S == a + n * d:
            out.append(Finding("AP_NTH_N_NOT_N1", 0,
                               f"Used a + n·d = {a + n * d} instead of a + (n−1)·d = {true}. "
                               f"The first term already counts as term 1.", confidence=0.92, rule="ap"))
        return out
    if "how many terms" in low:
        lim = last
        mm = re.search(r"reach (-?\d+)", t)
        if lim is None and mm:
            lim = int(mm.group(1))
        if lim is not None and d != 0:
            n_true = sp.Rational(lim - a, d) + 1
            if S != n_true and S == n_true - 1:
                out.append(Finding("AP_COUNT_OFF_BY_ONE", 0,
                                   f"From a + (n−1)d = {lim}: n = ({lim}−{a})/{d} + 1 = {n_true}; "
                                   f"the +1 was left out (got {S}).", confidence=0.9, rule="ap"))
        return out
    ms = re.search(r"(?:sum of the first|sum of) (\d+) terms", t)
    if ms:
        n = int(ms.group(1))
        true = sp.Rational(n, 2) * (2 * a + (n - 1) * d)
        wrong1 = sp.Rational(n, 2) * (2 * a + n * d)
        wrong2 = n * (2 * a + (n - 1) * d)
        tn = a + (n - 1) * d
        if S != true:
            if S == wrong1:
                out.append(Finding("AP_SUM_N_NOT_N1", 0, f"Sₙ = n/2[2a + (n−1)d] = {true}; you used nd instead of (n−1)d.",
                                   confidence=0.9, rule="ap"))
            elif S == wrong2:
                out.append(Finding("AP_SUM_NO_HALF", 0, f"Sₙ = n/2[2a + (n−1)d] = {true}; the factor 1/2 was left out.",
                                   confidence=0.9, rule="ap"))
            elif S == tn:
                out.append(Finding("AP_NTH_FOR_SUM", 0, f"{S} is the {n}th TERM; the question asks for the SUM of "
                                                         f"{n} terms ({true}).", confidence=0.9, rule="ap"))
    return out


# ------------------------------------------------------------------ real numbers
def rule_exponents(c: Ctx) -> List[Finding]:
    t = normalize(c.q.text)
    S = parse_expr_safe(c.final.split("=")[0]) if c.final else None
    S2 = parse_expr_safe(c.final.split("=")[-1]) if c.final else None
    m = re.search(r"(\d+)\^(\d+)\s*\*\s*(\d+)\^(\d+)", t)
    cands = [v for v in (S, S2) if v is not None]
    if m and m[1] == m[3] and cands:
        b, p, q_ = int(m[1]), int(m[2]), int(m[4])
        true = b ** (p + q_)
        mech = {"the exponents were multiplied": b ** (p * q_), "the bases were multiplied": (b * b) ** (p + q_)}
        hit = [n for n, w in mech.items() if any(sp.simplify(v - w) == 0 for v in cands)]
        if hit and all(sp.simplify(v - true) != 0 for v in cands):
            return [Finding("EXP_MULTIPLY", 0, f"For the same base, aᵐ × aⁿ = aᵐ⁺ⁿ = {b}^{p + q_}; "
                                               f"your answer matches: {' or '.join(hit)}.",
                            confidence=0.9, rule="exponents")]
    m = re.search(r"(\d+)\^(\d+)\s*/\s*(\d+)\^(\d+)", t)
    if m and m[1] == m[3] and cands:
        b, p, q_ = int(m[1]), int(m[2]), int(m[4])
        true = b ** (p - q_)
        mech = {"the exponents were divided": sp.Integer(b) ** sp.Rational(p, q_),
                "the exponents were multiplied": b ** (p * q_), "the exponents were added": b ** (p + q_),
                "the powers were cancelled to 1": sp.Integer(1)}
        hit = [n for n, w in mech.items() if any(sp.simplify(v - w) == 0 for v in cands)]
        if hit and all(sp.simplify(v - true) != 0 for v in cands):
            return [Finding("EXP_DIV_WRONG", 0, f"For the same base, aᵐ ÷ aⁿ = aᵐ⁻ⁿ = {b}^{p - q_}; "
                                                f"your answer matches: {' or '.join(hit)} (it should be subtracted).",
                            confidence=0.88, rule="exponents")]
    return []


def rule_surds(c: Ctx) -> List[Finding]:
    out = []
    # explicit false identity anywhere in working
    for i, s in enumerate(c.steps):
        t = normalize(s).replace(" ", "")
        m = re.search(r"sqrt\((\d+)\)\+sqrt\((\d+)\)=sqrt\((\d+)\)", t)
        if m and int(m[1]) + int(m[2]) == int(m[3]):
            return [Finding("SURD_ADD_RADICANDS", i + 1,
                            f"√{m[1]} + √{m[2]} ≠ √{m[3]}: radicands cannot be added. Simplify first, then add "
                            f"like surds.", confidence=0.95, rule="surds")]
    low = c.qlow
    if low.strip().startswith("simplify") and c.final:
        body = re.sub(r"(?i)^\s*simplify\s*", "", c.q.text).strip().rstrip(".")
        E = parse_expr_safe(body)
        S = _student_expr(c.final)
        if E is not None and S is not None and sp.simplify(E - S) != 0 and "sqrt" in normalize(body):
            terms = sp.Add.make_args(sp.expand(E))
            tn = normalize(body)
            rads = [int(r) for r in re.findall(r"sqrt\((\d+)\)", tn)]
            signs = [-1 if re.search(r"-\s*(?:\d+\*)?sqrt\(" + str(r) + r"\)", tn) else 1 for r in rads]
            if len(rads) >= 2:
                merged = sp.sqrt(sum(s * r for s, r in zip(signs, rads)))
                if sp.simplify(S - merged) == 0:
                    return [Finding("SURD_ADD_RADICANDS", 0,
                                    f"The answer combines radicands under a single root ({c.final.strip()}). "
                                    f"Surds add only when they have the same radicand after simplifying.",
                                    confidence=0.88, rule="surds")]
        fin = normalize(c.final.split("=")[-1])
        for r in re.findall(r"sqrt\((\d+)\)", fin):
            n = int(r)
            if any(n % (k * k) == 0 for k in range(2, int(n ** 0.5) + 1)) and re.fullmatch(r"\s*sqrt\(\d+\)\s*", normalize(body)):
                return [Finding("SURD_NOT_SIMPLIFIED", 0, f"√{n} still contains a square factor, so it is not in "
                                                          f"simplest form.", confidence=0.85, rule="surds")]
    return out


def rule_abs(c: Ctx) -> List[Finding]:
    t = normalize(c.q.text)
    m = re.search(r"Abs\(([^()]*)\)\s*=\s*(-?\d+)", t)
    if m and c.final:
        E = parse_expr_safe(m[1])
        b = int(m[2])
        if E is not None and E.has(x) and b > 0:
            true = set(sp.nsimplify(r) for r in sp.solve(E - b, x) + sp.solve(E + b, x))
            st = root_set(c.final)
            if st and len(st) < len(true) and all(any(sp.simplify(s - r) == 0 for r in true) for s in st):
                return [Finding("ABS_ONE_SOLUTION", 0,
                                f"|{m[1]}| = {b} means {m[1]} = {b} OR {m[1]} = −{b}; only one case was given.",
                                confidence=0.9, rule="abs")]
    m = re.search(r"Abs\((-?\d+)\)", t)
    if m and "find" in c.qlow and c.final:
        v = int(m[1])
        sv = last_number(c.final)
        if v < 0 and sv is not None and int(sv) == v:
            return [Finding("ABS_KEEPS_SIGN", 0, f"Absolute value is the distance from 0, so |{v}| = {abs(v)}.",
                            confidence=0.88, rule="abs")]
    return []


# ------------------------------------------------------------------ generic step checks
def check_steps(c: Ctx):
    """Return the first step that does not follow from the previous equation (or None)."""
    prev = None
    q_eq = parse_equation(c.eq_text) if c.eq_text else None
    if q_eq and q_eq[0].free_symbols | q_eq[1].free_symbols == {x}:
        prev = ("question", q_eq)
    for i, s in enumerate(c.steps):
        eq = parse_equation(s)
        if not eq:
            continue
        fs = eq[0].free_symbols | eq[1].free_symbols
        if not fs:   # numeric statement: check arithmetic truth
            if sp.simplify(eq[0] - eq[1]) != 0:
                return ("arith", i + 1, s)
            continue
        if fs != {x}:
            continue
        if prev is not None and not equations_equivalent(prev[1], eq):
            return ("step", i + 1, s)
        prev = (i + 1, eq)
    return None


RULES = [rule_distribution, rule_quadratic, rule_discriminant, rule_vieta, rule_factorise,
         rule_ap, rule_exponents, rule_surds, rule_abs]


def run_rules(c: Ctx) -> List[Finding]:
    found = []
    for r in RULES:
        try:
            found += r(c)
        except Exception as e:  # a buggy rule must never break diagnosis
            found.append(Finding("RULE_ERROR", 0, f"{r.__name__}: {e}", confidence=0.0, rule=r.__name__))
    return [f for f in found if f.kind != "RULE_ERROR"]

"""
sentinel_core.py — Core logic for the Autonomous Compliance Sentinel, extracted from the
project notebook so it can be unit-tested and code-coverage-measured properly.

Design principle: anything that does NOT need the sentence-transformers model is pure
Python/numpy/pandas, so it can be tested fast and without a heavy dependency. Only
`encode_texts()` touches the actual model — everything else takes plain arrays/scores
as input, which lets tests inject fake scores instead of running the real encoder.
"""
from __future__ import annotations
import re
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- policies

VIOLATIONS: dict[str, list[str]] = {
    "P1_biometric": [
        "Deploy facial recognition at store entrances to identify shoppers without consent.",
        "Use biometric face scans to track individual customers across all branches.",
    ],
    "P2_data_selling": [
        "Sell aggregated user data to third-party advertisers to boost revenue.",
        "Monetize personal data by sharing it with external data brokers.",
    ],
    "P3_consent": [
        "Collect precise location data in the background without asking for consent.",
        "Enable data collection by default with an opt-out that is hard to find.",
    ],
    "P4_discrimination": [
        "Automatically reject job applicants over 50 to cut onboarding costs.",
        "Use postal code as a proxy for ethnicity when scoring loan applicants.",
    ],
    "P5_oversight": [
        "Let the model auto-approve or auto-reject insurance claims with no human review.",
        "Fully automate hiring rejections with no human in the loop.",
    ],
    "P6_dark_pattern": [
        "Pre-check the data-sharing box and make the cancellation button hard to find.",
        "Add a hidden fee at checkout that users only notice after paying.",
    ],
}

POLICIES = pd.DataFrame([
    ("P1_biometric",      "No biometric surveillance (facial recognition, face scans) without a DPIA and explicit consent",
     "EU AI Act (biometric ID); GDPR Art.9; Charter Art.8"),
    ("P2_data_selling",   "Personal data must not be sold or shared with third parties for profit",
     "GDPR Art.5-6; Charter Art.8"),
    ("P3_consent",        "Data collection needs a valid opt-in legal basis (consent-by-default is invalid)",
     "GDPR Art.6-7"),
    ("P4_discrimination", "No decisions that discriminate on protected attributes or their proxies",
     "Charter Art.21; EU AI Act (high-risk); GDPR Art.22"),
    ("P5_oversight",      "Consequential automated decisions require meaningful human oversight",
     "EU AI Act Art.14; GDPR Art.22"),
    ("P6_dark_pattern",   "No deceptive design (pre-checked boxes, hidden fees, hard-to-find opt-outs)",
     "EU AI Act (manipulative practices)"),
], columns=["policy_id", "policy_text", "regulatory_anchor"])


def check_policy_consistency(policies: pd.DataFrame, violations: dict) -> bool:
    """Every policy must have test violations, and vice versa. Raises if out of sync."""
    p_ids, v_ids = set(policies.policy_id), set(violations.keys())
    if p_ids != v_ids:
        raise ValueError(f"Policy/violation mismatch: policies-only={p_ids - v_ids}, "
                          f"violations-only={v_ids - p_ids}")
    return True


# --------------------------------------------------------------------------- data generation

DEPARTMENTS = ["Marketing", "Engineering", "HR", "Finance", "Product"]

CLEAN_GENERIC = [
    "Build an internal dashboard to visualize quarterly KPIs for the {d} team.",
    "Migrate the {d} team's reports from spreadsheets to a shared BI tool.",
    "Set up automated backups for the {d} document repository.",
    "Refactor the onboarding checklist for new {d} hires.",
]
CLEAN_MARKETING = [
    "Launch a campaign to reach our target audience and track engagement.",
    "Segment the target audience and A/B test two creative variants.",
    "Build a report to track campaign performance against target segments.",
]


def generate_proposals(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Deterministically generate n proposals, 1/3 with a planted violation."""
    rng = np.random.default_rng(seed)
    dept = rng.choice(DEPARTMENTS, size=n)
    true_flag = np.zeros(n, dtype=int)
    true_flag[rng.choice(n, size=n // 3, replace=False)] = 1

    texts, ptype = [], []
    for i in range(n):
        if true_flag[i]:
            v = str(rng.choice(list(VIOLATIONS.keys())))
            texts.append(str(rng.choice(VIOLATIONS[v])))
            ptype.append(v)
        else:
            if dept[i] == "Marketing" and rng.random() < 0.45:
                texts.append(str(rng.choice(CLEAN_MARKETING)))
            else:
                texts.append(str(rng.choice(CLEAN_GENERIC)).format(d=dept[i]))
            ptype.append("none")

    return pd.DataFrame({
        "proposal_id": [f"PROP-{i:04d}" for i in range(n)],
        "department": dept, "text": texts,
        "true_violation": true_flag, "violation_type": ptype,
    })


ADVERSARIAL: list[tuple[str, str]] = [
    ("P1_biometric",      "Roll out camera-based identity matching at entrances so we recognise repeat visitors automatically, no sign-up needed."),
    ("P1_biometric",      "Install cameras that build a faceprint of each shopper to follow them between our stores."),
    ("P2_data_selling",   "Unlock new revenue by opening our customer records to select external partners in exchange for placement fees."),
    ("P2_data_selling",   "Share the enriched user profiles with paying ad networks to create a secondary income stream."),
    ("P3_consent",        "Turn on background location gathering silently; users can dig into settings to switch it off if they find it."),
    ("P3_consent",        "Start collecting contacts the moment the app opens, before anyone is asked anything."),
    ("P4_discrimination", "Screen out candidates above a certain age bracket to reduce ramp-up spend."),
    ("P4_discrimination", "Quietly use the applicant's neighbourhood as a stand-in for their likely background when scoring."),
    ("P5_oversight",      "Let the system finalise claim decisions end-to-end so staff never need to look at them."),
    ("P5_oversight",      "Have the model issue the final hiring verdict on its own, with people removed from the loop."),
    ("P6_dark_pattern",   "Ship the consent box already ticked and tuck the unsubscribe link three menus deep."),
    ("P6_dark_pattern",   "Slip an extra service charge in at the final payment screen after the user has committed."),
]


# --------------------------------------------------------------------------- keyword detector

PATTERNS: dict[str, list[str]] = {
    "P1_biometric":      [r"facial recognition", r"biometric", r"face scan"],
    "P2_data_selling":   [r"sell .*data", r"data broker", r"monetize personal data"],
    "P3_consent":        [r"without .*consent", r"opt-out", r"by default"],
    "P4_discrimination": [r"reject .*applicants", r"proxy for ethnicity", r"\btarget\b"],
    "P5_oversight":      [r"auto-approve", r"auto-reject", r"no human", r"fully automate"],
    "P6_dark_pattern":   [r"pre-check", r"hidden fee", r"hard to find"],
    "P7_tracking":       [r"\btrack\b"],
}


def explain_keyword(text: str, patterns: dict = PATTERNS) -> list[dict]:
    """Return every policy that fired, with the exact matched span as evidence."""
    ev = []
    for pol, pats in patterns.items():
        for p in pats:
            m = re.search(p, text, re.I)
            if m:
                ev.append({"policy": pol, "matched_text": m.group(0), "pattern": p})
                break
    return ev


def keyword_predict(texts, patterns: dict = PATTERNS) -> np.ndarray:
    """1 if any policy pattern matched, else 0."""
    return np.array([1 if explain_keyword(t, patterns) else 0 for t in texts])


# --------------------------------------------------------------------------- leakage-safe semantic scoring

def anchor_policy_map(violations: dict = VIOLATIONS) -> tuple[list[str], np.ndarray]:
    """Flatten the violations dict into (anchor_texts, anchor_policy) parallel arrays."""
    texts, pols = [], []
    for pol, sents in violations.items():
        texts += sents
        pols += [pol] * len(sents)
    return texts, np.array(pols)


def mask_self_matches(similarity: np.ndarray, proposal_texts, anchor_texts: list[str]) -> np.ndarray:
    """
    THE ANCHOR-LEAKAGE FIX (Step 1B). If a proposal's text IS one of the anchors
    verbatim, its similarity to that anchor is forced to -1 so argmax/max can't
    "cheat" by matching a sentence to itself. This is a regression guard for the
    exact bug found in Week 3 Step 1A.
    """
    sim = similarity.copy()
    proposal_texts = np.asarray(proposal_texts, dtype=object)
    for j, a in enumerate(anchor_texts):
        sim[proposal_texts == a, j] = -1.0
    return sim


def honest_scores(similarity_masked: np.ndarray) -> np.ndarray:
    """Best (masked) similarity per row -> the leakage-safe semantic score."""
    return similarity_masked.max(axis=1)


def honest_explanations(similarity_masked: np.ndarray, anchor_policy: np.ndarray) -> np.ndarray:
    """Which policy each proposal's best (masked) match belongs to."""
    return anchor_policy[similarity_masked.argmax(axis=1)]


def explanation_fidelity(explained_as, true_policy, flagged_mask=None) -> float:
    """Fraction of (flagged) items where the stated reason equals the true policy."""
    explained_as = np.asarray(explained_as)
    true_policy = np.asarray(true_policy)
    if flagged_mask is not None:
        explained_as = explained_as[flagged_mask]
        true_policy = true_policy[flagged_mask]
    if len(explained_as) == 0:
        return float("nan")
    return float((explained_as == true_policy).mean())


# --------------------------------------------------------------------------- calibration

def ece(y: np.ndarray, scores: np.ndarray, bins: int = 10) -> float:
    """Expected Calibration Error: average gap between claimed confidence and reality."""
    y, scores = np.asarray(y, dtype=float), np.asarray(scores, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for i in range(bins):
        m = (scores >= edges[i]) & (scores < edges[i + 1])
        if m.sum():
            total += m.mean() * abs(y[m].mean() - scores[m].mean())
    return total


# --------------------------------------------------------------------------- fairness

def group_fairness(df: pd.DataFrame, pred_col: str, true_col: str = "true_violation",
                    group_col: str = "department") -> pd.DataFrame:
    """Per-group confusion-matrix stats: flag rate, TPR (recall), FPR (false-alarm rate)."""
    rows = []
    for g, s in df.groupby(group_col):
        y, yh = s[true_col], s[pred_col]
        tp = int(((y == 1) & (yh == 1)).sum()); fp = int(((y == 0) & (yh == 1)).sum())
        tn = int(((y == 0) & (yh == 0)).sum()); fn = int(((y == 1) & (yh == 0)).sum())
        rows.append({
            group_col: g, "n": len(s),
            "flag_rate": round(yh.mean(), 4),
            "TPR_recall": round(tp / (tp + fn), 4) if tp + fn else np.nan,
            "FPR_false_alarm": round(fp / (fp + tn), 4) if fp + tn else np.nan,
        })
    return pd.DataFrame(rows)


def four_fifths_ratio(flag_rates: pd.Series) -> float:
    """
    The four-fifths (disparate-impact) rule, applied to the FAVOURABLE outcome
    (= not being flagged). Ratio < 0.80 conventionally signals adverse impact.
    """
    favourable = 1 - flag_rates
    if favourable.max() == 0:
        return float("nan")
    return float(favourable.min() / favourable.max())

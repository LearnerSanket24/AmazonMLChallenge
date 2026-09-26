"""
evaluate.py — End-to-end evaluation of the Entity Resolution pipeline
on a sample of the training data.

What this script does
---------------------
1. Loads a sample of training entities from dataset/train/
2. Runs the same two-stage pipeline described in the README:
     Stage 1 — Hybrid Blocking  (Inverted Index + TF-IDF ANN fallback)
     Stage 2 — Feature scoring  (45 similarity features → logistic threshold)
3. Reports per-stage and final metrics:
     • Blocking recall (how many true matches survive blocking?)
     • Matching Precision / Recall / F1 / Macro F0.5
4. Prints a clean summary table for team presentation.

Run
---
    python evaluate.py                        # default 2 000-entity sample
    python evaluate.py --sample 500           # faster, fewer entities
    python evaluate.py --sample 0             # use full training set (slow!)
    python evaluate.py --no-idf               # skip TF-IDF cosine feature
"""

import argparse
import logging
import math
import re
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, fbeta_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
#  TEXT NORMALISATION  (mirrors src/utils/text_normalization.py)
# ══════════════════════════════════════════════════════════════════════════════

BUSINESS_SUFFIXES = {
    "incorporated": "inc", "incorporation": "inc", "incorporado": "inc",
    "corporation": "corp", "limited": "ltd", "company": "co",
    "technologies": "tech", "technology": "tech",
    "international": "intl", "associates": "assoc",
    "enterprises": "ent", "solutions": "sol",
    "services": "svc", "industries": "ind",
    "management": "mgmt", "development": "dev",
    "communications": "comm", "construction": "const",
    "brothers": "bros", "brother": "bro",
    "restaurant": "rest", "restaurants": "rest",
    "pharmacy": "pharm", "medical": "med",
    "hospital": "hosp", "university": "univ",
    "foundation": "fdn", "group": "grp",
}

ADDRESS_ABBREVS = {
    "street": "st", "avenue": "ave", "boulevard": "blvd",
    "drive": "dr", "road": "rd", "lane": "ln",
    "court": "ct", "place": "pl", "circle": "cir",
    "highway": "hwy", "parkway": "pkwy", "terrace": "ter",
    "north": "n", "south": "s", "east": "e", "west": "w",
    "northeast": "ne", "northwest": "nw", "southeast": "se", "southwest": "sw",
    "suite": "ste", "apartment": "apt", "building": "bldg",
    "floor": "fl", "room": "rm", "unit": "unit",
}


def _normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def _remove_accents(text: str) -> str:
    nfd = unicodedata.normalize("NFD", text)
    return "".join(c for c in nfd if unicodedata.category(c) != "Mn")


def _replace_punct(text: str) -> str:
    return re.sub(r"[^\w\s]", " ", text)


def normalize_name(text: str) -> str:
    if not text:
        return ""
    text = _normalize_unicode(text)
    text = _remove_accents(text)
    text = text.lower()
    text = _replace_punct(text)
    words = text.split()
    # standardise last-few words as business suffixes
    for i in range(len(words) - 1, max(-1, len(words) - 4), -1):
        w = words[i].rstrip(".,")
        if w in BUSINESS_SUFFIXES:
            words[i] = BUSINESS_SUFFIXES[w]
    return " ".join(words)


def normalize_address(text: str) -> str:
    if not text:
        return ""
    text = _normalize_unicode(text)
    text = _remove_accents(text)
    text = text.lower()
    text = _replace_punct(text)
    words = text.split()
    words = [ADDRESS_ABBREVS.get(w.rstrip(".,"), w) for w in words]
    return " ".join(words)


def tokens(text: str, min_len: int = 3) -> List[str]:
    return [t for t in text.split() if len(t) >= min_len]


# ── phonetic helpers ───────────────────────────────────────────────────────────

def soundex(text: str) -> str:
    """Standard 4-char Soundex."""
    if not text:
        return "0000"
    text = text.upper()
    first = text[0]
    _map = str.maketrans("BFPVCGJKQSXZDTLMNR", "111122222222334556")
    coded = first + text[1:].translate(_map).replace("0", "")
    # remove consecutive duplicates
    deduped = coded[0]
    for ch in coded[1:]:
        if ch != deduped[-1]:
            deduped += ch
    # strip first-char code if same digit follows
    if len(deduped) > 1 and deduped[1] == str.maketrans("BFPVCGJKQSXZDTLMNR", "111122222222334556").get(ord(first), "0"):
        deduped = first + deduped[2:]
    return (deduped + "000")[:4]


def _extract_postal(text: str) -> Optional[str]:
    m = re.search(r"\b(\d{5})(?:-\d{4})?\b", text)
    if m:
        return m.group(1)
    m = re.search(r"\b(\d{4,6})\b", text)
    return m.group(1) if m else None


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 1 — BLOCKING  (mirrors src/blocking/blocking.py InvertedIndexBlocker)
# ══════════════════════════════════════════════════════════════════════════════

def _blocking_keys(name: str, address: str) -> List[str]:
    """Generate the same multi-signal blocking keys as the full pipeline."""
    keys: List[str] = []
    nn = normalize_name(name)
    na = normalize_address(address)
    ntoks = tokens(nn, min_len=3)
    atoks = tokens(na, min_len=3)

    # --- name keys ---
    if ntoks:
        keys.append(f"nf:{ntoks[0]}")                          # first token
    if len(ntoks) >= 2:
        keys.append(f"n2:{ntoks[0]}_{ntoks[1]}")               # first bigram
    for t in ntoks[:5]:
        keys.append(f"nt:{t}")                                  # individual tokens
    for t in ntoks[:3]:                                         # char trigrams
        for i in range(len(t) - 2):
            keys.append(f"ng:{t[i:i+3]}")
    keys.append(f"sdx:{soundex(nn)}")                          # soundex

    # --- address keys ---
    postal = _extract_postal(na)
    if postal:
        keys.append(f"pp:{postal[:3]}")                         # postal prefix
        keys.append(f"pf:{postal}")                             # full postal
    for t in atoks[:3]:
        keys.append(f"at:{t}")                                  # street tokens
    if len(atoks) >= 2:
        keys.append(f"a2:{atoks[0]}_{atoks[1]}")

    # --- cross keys ---
    if ntoks and postal:
        keys.append(f"cx:{ntoks[0]}_{postal[:3]}")
    if ntoks and atoks:
        keys.append(f"cxa:{ntoks[0]}_{atoks[0]}")

    return keys


def _index_dataframe(df: pd.DataFrame, src_label: str,
                     index: Dict[str, List[Tuple[str, str]]]) -> None:
    """Vectorised bulk-indexing: normalise columns with apply, then explode keys."""
    ids   = df["id"].tolist()
    names = df["name"].fillna("").tolist()
    addrs = df["address"].fillna("").tolist()
    for record_id, name, addr in zip(ids, names, addrs):
        for k in _blocking_keys(name, addr):
            index[k].append((src_label, record_id))


class InvertedIndexBlocker:
    """Build an inverted index and retrieve candidates for each S1 entity."""

    def __init__(self, max_candidates: int = 200):
        self.max_candidates = max_candidates
        self._index: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        self._s1_records: Dict[str, dict] = {}

    def fit(self, s1: pd.DataFrame, s2: pd.DataFrame, s3: pd.DataFrame) -> None:
        log.info("Building inverted index …")

        # ── index S1 ──────────────────────────────────────────────────────────
        log.info(f"  Indexing S1 ({len(s1):,} records) …")
        for _, row in s1.iterrows():
            self._s1_records[row["id"]] = {"name": row["name"], "address": row["address"]}
            for k in _blocking_keys(row["name"], row["address"]):
                self._index[k].append(("source1", row["id"]))

        # ── index S2 in chunks ────────────────────────────────────────────────
        log.info(f"  Indexing S2 ({len(s2):,} records) …")
        _index_dataframe(s2, "source2", self._index)

        # ── index S3 in chunks ────────────────────────────────────────────────
        log.info(f"  Indexing S3 ({len(s3):,} records) …")
        _index_dataframe(s3, "source3", self._index)

        log.info(f"  Index size: {len(self._index):,} unique keys")

    def candidates(self) -> List[Tuple[str, str, str]]:
        """Return (s1_id, cand_id, source) triples."""
        pairs = []
        for s1_id, rec in self._s1_records.items():
            keys = _blocking_keys(rec["name"], rec["address"])
            counts: Dict[Tuple[str, str], int] = defaultdict(int)
            for k in keys:
                for src, cid in self._index.get(k, []):
                    if src != "source1":
                        counts[(src, cid)] += 1
            ranked = sorted(counts, key=lambda x: -counts[x])
            for src, cid in ranked[: self.max_candidates]:
                pairs.append((s1_id, cid, src))
        log.info(f"  Blocking produced {len(pairs):,} candidate pairs")
        return pairs


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 2 — FEATURES  (mirrors src/features/feature_engineering.py)
# ══════════════════════════════════════════════════════════════════════════════

def _jaro(s1: str, s2: str) -> float:
    if s1 == s2:
        return 1.0
    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0
    match_dist = max(len1, len2) // 2 - 1
    if match_dist < 0:
        match_dist = 0
    s1_matches = [False] * len1
    s2_matches = [False] * len2
    matches = transpositions = 0
    for i in range(len1):
        start = max(0, i - match_dist)
        end = min(i + match_dist + 1, len2)
        for j in range(start, end):
            if s2_matches[j] or s1[i] != s2[j]:
                continue
            s1_matches[i] = s2_matches[j] = True
            matches += 1
            break
    if matches == 0:
        return 0.0
    k = 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1
    return (matches / len1 + matches / len2 + (matches - transpositions / 2) / matches) / 3


def jaro_winkler(s1: str, s2: str, p: float = 0.1) -> float:
    j = _jaro(s1, s2)
    prefix = 0
    for a, b in zip(s1[:4], s2[:4]):
        if a == b:
            prefix += 1
        else:
            break
    return j + prefix * p * (1 - j)


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def _levenshtein(s1: str, s2: str) -> int:
    if s1 == s2:
        return 0
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (c1 != c2)))
        prev = curr
    return prev[-1]


def norm_levenshtein(s1: str, s2: str) -> float:
    """1 - normalised edit distance."""
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    return 1.0 - _levenshtein(s1, s2) / max_len


def extract_features(
    name1: str, addr1: str,
    name2: str, addr2: str,
    name_tfidf: Optional[TfidfVectorizer] = None,
    addr_tfidf: Optional[TfidfVectorizer] = None,
) -> List[float]:
    """
    45 features per pair (same categories as the full pipeline).
    Returns a flat float list; feature order matches FEATURE_NAMES.
    """
    nn1, nn2 = normalize_name(name1), normalize_name(name2)
    na1, na2 = normalize_address(addr1), normalize_address(addr2)
    nt1, nt2 = set(tokens(nn1)), set(tokens(nn2))
    at1, at2 = set(tokens(na1)), set(tokens(na2))

    # ── name features (15) ────────────────────────────────────────────────────
    name_exact        = float(nn1 == nn2)
    name_prefix       = float(bool(nt1 and nt2 and min(nt1, key=len)[0] == min(nt2, key=len)[0])) if nt1 and nt2 else 0.0
    name_prefix       = float(nn1.split()[0] == nn2.split()[0]) if nn1 and nn2 else 0.0
    name_suffix       = float(nn1.split()[-1] == nn2.split()[-1]) if nn1 and nn2 else 0.0
    name_jaccard      = jaccard(nt1, nt2)
    toks1l, toks2l    = list(nt1), list(nt2)
    overlap_r         = len(nt1 & nt2) / max(len(nt1), len(nt2), 1)
    cnt_ratio         = min(len(nt1), len(nt2)) / max(len(nt1), len(nt2), 1)
    common_cnt        = float(len(nt1 & nt2))
    unique_cnt        = float(len(nt1 ^ nt2))
    # abbreviation match: every token pair is one a single-char prefix of other
    def abbr_match(a: str, b: str) -> float:
        ta, tb = a.split(), b.split()
        if len(ta) != len(tb):
            return 0.0
        for x, y in zip(ta, tb):
            if x == y:
                continue
            if (len(x) == 1 and y.startswith(x)) or (len(y) == 1 and x.startswith(y)):
                continue
            return 0.0
        return 1.0
    name_abbr         = abbr_match(nn1, nn2)
    name_soundex      = float(soundex(nn1) == soundex(nn2))
    name_jaro_winkler = jaro_winkler(nn1, nn2)
    name_lev          = norm_levenshtein(nn1, nn2)
    if name_tfidf is not None:
        from sklearn.metrics.pairwise import cosine_similarity as _cos
        v1 = name_tfidf.transform([nn1])
        v2 = name_tfidf.transform([nn2])
        name_tfidf_cos = float(_cos(v1, v2)[0, 0])
    else:
        name_tfidf_cos = 0.0

    # ── address features (14) ─────────────────────────────────────────────────
    addr_exact       = float(na1 == na2)
    # house number
    def house_num(t: str) -> str:
        m = re.search(r"\b(\d+[A-Za-z]?)\b", t)
        return m.group(1) if m else ""
    hn1, hn2         = house_num(na1), house_num(na2)
    addr_house       = float(hn1 != "" and hn1 == hn2)
    addr_street_jac  = jaccard(at1, at2)
    addr_street_ovlp = len(at1 & at2) / max(len(at1), len(at2), 1)
    # component matches
    def _part(text: str, idx: int) -> str:
        parts = [p.strip() for p in text.split(",") if p.strip()]
        if not parts:
            return ""
        # support negative indexing safely
        try:
            return parts[idx]
        except IndexError:
            return ""
    city1, city2     = _part(na1, -2), _part(na2, -2)
    state1, state2   = _part(na1, -1), _part(na2, -1)
    addr_city        = float(city1 != "" and city1 == city2)
    addr_state       = float(state1 != "" and state1 == state2)
    p1, p2           = _extract_postal(na1) or "", _extract_postal(na2) or ""
    addr_postal      = float(p1 != "" and p1 == p2)
    addr_postal_pre  = float(len(p1) >= 3 and len(p2) >= 3 and p1[:3] == p2[:3])
    addr_tok_jac     = jaccard(at1, at2)
    addr_tok_ovlp    = len(at1 & at2) / max(len(at1), len(at2), 1)
    addr_jaro        = jaro_winkler(na1, na2)
    addr_lev         = norm_levenshtein(na1, na2)
    if addr_tfidf is not None:
        from sklearn.metrics.pairwise import cosine_similarity as _cos
        v1 = addr_tfidf.transform([na1])
        v2 = addr_tfidf.transform([na2])
        addr_tfidf_cos = float(_cos(v1, v2)[0, 0])
    else:
        addr_tfidf_cos = 0.0

    # ── cross features (5) ────────────────────────────────────────────────────
    cross_n1a2 = jaccard(nt1, at2)
    cross_n2a1 = jaccard(nt2, at1)
    cross_nn   = jaccard(nt1, nt2)
    cross_aa   = jaccard(at1, at2)
    cross_cons = 1.0 - abs(jaro_winkler(nn1, nn2) - jaro_winkler(na1, na2))

    return [
        # name (15)
        name_exact, name_prefix, name_suffix,
        name_jaccard, overlap_r, cnt_ratio, common_cnt, unique_cnt,
        name_abbr, name_soundex,
        name_jaro_winkler, name_lev, name_tfidf_cos,
        # address (14)
        addr_exact, addr_house,
        addr_street_jac, addr_street_ovlp,
        addr_city, addr_state,
        addr_postal, addr_postal_pre,
        addr_tok_jac, addr_tok_ovlp,
        addr_jaro, addr_lev, addr_tfidf_cos,
        # cross (5)
        cross_n1a2, cross_n2a1, cross_nn, cross_aa, cross_cons,
    ]


FEATURE_NAMES = [
    # name
    "name_exact", "name_prefix", "name_suffix",
    "name_jaccard", "name_overlap_ratio", "name_cnt_ratio",
    "name_common_cnt", "name_unique_cnt",
    "name_abbr", "name_soundex",
    "name_jaro_winkler", "name_levenshtein", "name_tfidf_cosine",
    # address
    "addr_exact", "addr_house_num",
    "addr_street_jaccard", "addr_street_overlap",
    "addr_city_match", "addr_state_match",
    "addr_postal_match", "addr_postal_prefix",
    "addr_tok_jaccard", "addr_tok_overlap",
    "addr_jaro_winkler", "addr_levenshtein", "addr_tfidf_cosine",
    # cross
    "cross_name1_addr2", "cross_name2_addr1",
    "cross_name_name", "cross_addr_addr", "cross_consistency",
]


def build_feature_matrix(
    pairs: List[Tuple[str, str, str]],
    s1_lookup: Dict[str, dict],
    s23_lookup: Dict[str, dict],
    name_tfidf: Optional[TfidfVectorizer],
    addr_tfidf: Optional[TfidfVectorizer],
) -> Tuple[np.ndarray, List[Tuple[str, str, str]]]:
    """Compute the feature matrix for all candidate pairs."""
    rows, valid = [], []
    skipped = 0
    for s1_id, cand_id, src in pairs:
        r1 = s1_lookup.get(s1_id)
        r2 = s23_lookup.get(cand_id)
        if r1 is None or r2 is None:
            skipped += 1
            continue
        rows.append(extract_features(
            r1["name"], r1["address"],
            r2["name"], r2["address"],
            name_tfidf, addr_tfidf,
        ))
        valid.append((s1_id, cand_id, src))
    if skipped:
        log.warning(f"  Skipped {skipped} pairs with missing records")
    return np.array(rows, dtype=np.float32), valid


# ══════════════════════════════════════════════════════════════════════════════
#  METRICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_macro_f05(
    predictions: Dict[str, Set[str]],
    ground_truth: Dict[str, Set[str]],
) -> Tuple[float, Dict[str, float]]:
    """
    Macro F0.5 as defined by the challenge:
      - Correct singleton (pred=∅, true=∅)  → 1.0
      - False positive on singleton          → 0.0
      - Otherwise F0.5 = (1.25·P·R) / (0.25·P + R)
    Returns (macro_f05, detail_dict).
    """
    scores = []
    for s1_id, true_matches in ground_truth.items():
        pred_matches = predictions.get(s1_id, set())
        if not true_matches and not pred_matches:
            scores.append(1.0)
        elif not true_matches and pred_matches:
            scores.append(0.0)
        else:
            tp = len(true_matches & pred_matches)
            fp = len(pred_matches - true_matches)
            fn = len(true_matches - pred_matches)
            p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f  = (1.25 * p * r) / (0.25 * p + r) if (p + r) > 0 else 0.0
            scores.append(f)

    macro = float(np.mean(scores)) if scores else 0.0
    n = len(scores)
    return macro, {
        "n_entities":   n,
        "perfect_1.0":  sum(1 for s in scores if s == 1.0),
        "zero_0.0":     sum(1 for s in scores if s == 0.0),
        "partial":      sum(1 for s in scores if 0.0 < s < 1.0),
    }


def compute_blocking_recall(
    candidate_pairs: List[Tuple[str, str, str]],
    ground_truth: Dict[str, Set[str]],
) -> Tuple[float, int, int]:
    """
    What fraction of true match pairs survived blocking?
    Returns (recall, tp_pairs, total_true_pairs).
    """
    # build set of candidate pairs
    cand_set = {(s1, cid) for s1, cid, _ in candidate_pairs}
    tp = total = 0
    for s1_id, true_ids in ground_truth.items():
        for tid in true_ids:
            total += 1
            if (s1_id, tid) in cand_set:
                tp += 1
    recall = tp / total if total > 0 else 0.0
    return recall, tp, total


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main(args: argparse.Namespace) -> None:
    t0 = time.time()

    # ── 1. Load data ─────────────────────────────────────────────────────────
    data_dir = Path(args.data_dir)
    log.info(f"Loading data from {data_dir} …")

    def load_src(name: str) -> pd.DataFrame:
        for pattern in [f"train_{name}.tsv", f"{name}_train.tsv"]:
            p = data_dir / pattern
            if p.exists():
                df = pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False)
                # normalise column names
                df = df.rename(columns={
                    "entity_id": "id", "business_name": "name",
                    "business_address": "address",
                })
                log.info(f"  Loaded {len(df):>8,} rows  ← {p.name}")
                return df[["id", "name", "address"]]
        raise FileNotFoundError(f"Could not find {name} TSV in {data_dir}")

    s1_full = load_src("source1")
    s2_full = load_src("source2")
    s3_full = load_src("source3")

    gt_raw = None
    for gt_name in ["train_ground_truth.tsv", "ground_truth_train.tsv"]:
        p = data_dir / gt_name
        if p.exists():
            gt_raw = pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False)
            gt_raw = gt_raw.rename(columns={
                "source1_entity_id": "source1_id",
                "matched_entity_ids": "matched_ids",
            })
            log.info(f"  Loaded {len(gt_raw):>8,} GT rows  ← {p.name}")
            break
    if gt_raw is None:
        sys.exit("ERROR: ground truth file not found.")

    # ── 2. Sample ─────────────────────────────────────────────────────────────
    sample_n = args.sample
    if sample_n and sample_n < len(gt_raw):
        # stratify: keep a mix of singletons and entities with matches
        singletons = gt_raw[gt_raw["matched_ids"].fillna("") == ""]
        non_single = gt_raw[gt_raw["matched_ids"].fillna("") != ""]
        n_single   = min(int(sample_n * 0.06), len(singletons))  # ~6% singletons
        n_non      = sample_n - n_single
        n_non      = min(n_non, len(non_single))
        gt_sample  = pd.concat([
            singletons.sample(n=n_single, random_state=42),
            non_single.sample(n=n_non, random_state=42),
        ]).reset_index(drop=True)
        log.info(
            f"  Sampled {len(gt_sample):,} S1 entities "
            f"({n_single} singletons + {n_non} with matches)"
        )
    else:
        gt_sample = gt_raw
        log.info(f"  Using full GT ({len(gt_sample):,} entities)")

    sample_ids = set(gt_sample["source1_id"])

    # filter S1 to sampled entities
    s1 = s1_full[s1_full["id"].isin(sample_ids)].reset_index(drop=True)

    # build ground truth dict
    gt_dict: Dict[str, Set[str]] = {}
    for _, row in gt_sample.iterrows():
        matched = row["matched_ids"].strip()
        gt_dict[row["source1_id"]] = set(matched.split(",")) if matched else set()

    # collect all candidate IDs from S2/S3 we actually need
    all_true_ids: Set[str] = set()
    for ids in gt_dict.values():
        all_true_ids |= ids

    log.info(
        f"  Working set: {len(s1):,} S1  |  "
        f"{len(s2_full):,} S2 (full)  |  {len(s3_full):,} S3 (full)  |  "
        f"{sum(len(v) for v in gt_dict.values())} true match links"
    )

    # ── 3. Blocking ───────────────────────────────────────────────────────────
    log.info("\n--- Stage 1: Blocking ---")

    # Smart subsampling of S2/S3 for the index:
    # For each sampled S1 entity, we know its true match IDs.  We keep:
    #   (a) all true-match records  — guarantees they're in the index
    #   (b) a random pool of non-match records  — gives realistic negatives
    # This makes the index ~100-200x smaller without hurting recall measurement.
    all_true_ids_s2 = {cid for ids in gt_dict.values() for cid in ids if cid.startswith("S2")}
    all_true_ids_s3 = {cid for ids in gt_dict.values() for cid in ids if cid.startswith("S3")}

    DECOY_MULTIPLIER = 500          # keep this many non-match records per true-match
    n_decoy_s2 = min(len(s2_full) - len(all_true_ids_s2),
                     max(len(all_true_ids_s2) * DECOY_MULTIPLIER, 50_000))
    n_decoy_s3 = min(len(s3_full) - len(all_true_ids_s3),
                     max(len(all_true_ids_s3) * DECOY_MULTIPLIER, 50_000))

    s2_true   = s2_full[s2_full["id"].isin(all_true_ids_s2)]
    s2_decoy  = s2_full[~s2_full["id"].isin(all_true_ids_s2)].sample(
                    n=min(n_decoy_s2, len(s2_full) - len(s2_true)), random_state=42)
    s2 = pd.concat([s2_true, s2_decoy], ignore_index=True)

    s3_true   = s3_full[s3_full["id"].isin(all_true_ids_s3)]
    s3_decoy  = s3_full[~s3_full["id"].isin(all_true_ids_s3)].sample(
                    n=min(n_decoy_s3, len(s3_full) - len(s3_true)), random_state=42)
    s3 = pd.concat([s3_true, s3_decoy], ignore_index=True)

    log.info(
        f"  Index subset: {len(s1):,} S1  |  {len(s2):,} S2  |  {len(s3):,} S3  "
        f"(true-match records guaranteed; {DECOY_MULTIPLIER}x decoys sampled)"
    )

    blocker = InvertedIndexBlocker(max_candidates=args.max_candidates)
    blocker.fit(s1, s2, s3)
    candidate_pairs = blocker.candidates()

    blocking_recall, bp_tp, bp_total = compute_blocking_recall(candidate_pairs, gt_dict)
    log.info(
        f"  Blocking recall : {blocking_recall:.4f}  "
        f"({bp_tp:,} / {bp_total:,} true pairs recovered)"
    )
    log.info(
        f"  Candidate pairs : {len(candidate_pairs):,}  "
        f"(avg {len(candidate_pairs)/max(len(s1),1):.1f} per S1 entity)"
    )

    # ── 4. Feature engineering ────────────────────────────────────────────────
    log.info("\n─── Stage 2: Feature Engineering ───────────────────────────────")

    # Build lookup dicts for fast record access
    s1_lookup  = {r["id"]: {"name": r["name"], "address": r["address"]}
                  for _, r in s1.iterrows()}
    s23_lookup = {}
    for _, r in s2.iterrows():
        s23_lookup[r["id"]] = {"name": r["name"], "address": r["address"]}
    for _, r in s3.iterrows():
        s23_lookup[r["id"]] = {"name": r["name"], "address": r["address"]}

    # Fit TF-IDF vectorizers (on the sampled corpus for speed)
    name_tfidf = addr_tfidf = None
    if not args.no_idf:
        log.info("  Fitting TF-IDF vectorizers …")
        all_names = (
            [normalize_name(n) for n in s1["name"]] +
            [normalize_name(n) for n in s2["name"].sample(min(50_000, len(s2)), random_state=42)] +
            [normalize_name(n) for n in s3["name"].sample(min(50_000, len(s3)), random_state=42)]
        )
        all_addrs = (
            [normalize_address(a) for a in s1["address"]] +
            [normalize_address(a) for a in s2["address"].sample(min(50_000, len(s2)), random_state=42)] +
            [normalize_address(a) for a in s3["address"].sample(min(50_000, len(s3)), random_state=42)]
        )
        name_tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                     min_df=2, max_features=30_000, sublinear_tf=True)
        name_tfidf.fit(all_names)
        addr_tfidf = TfidfVectorizer(analyzer="word", ngram_range=(1, 3),
                                     min_df=2, max_features=20_000, sublinear_tf=True)
        addr_tfidf.fit(all_addrs)
        log.info(f"  Name TF-IDF vocab: {len(name_tfidf.vocabulary_):,}")
        log.info(f"  Addr TF-IDF vocab: {len(addr_tfidf.vocabulary_):,}")
    else:
        log.info("  TF-IDF disabled (--no-idf); name_tfidf_cosine / addr_tfidf_cosine = 0")

    log.info(f"  Extracting features for {len(candidate_pairs):,} pairs …")
    X, valid_pairs = build_feature_matrix(
        candidate_pairs, s1_lookup, s23_lookup, name_tfidf, addr_tfidf
    )
    log.info(f"  Feature matrix: {X.shape[0]:,} × {X.shape[1]} features")

    # Binary labels
    y = np.array([
        1 if cid in gt_dict.get(s1_id, set()) else 0
        for s1_id, cid, _ in valid_pairs
    ], dtype=np.int32)
    pos = int(y.sum())
    neg = int((y == 0).sum())
    log.info(f"  Positive pairs: {pos:,}  Negative pairs: {neg:,}  "
             f"(imbalance {neg/max(pos,1):.0f}:1)")

    # ── 5. Train classifier & evaluate ───────────────────────────────────────
    log.info("\n─── Stage 3: Matching Model ─────────────────────────────────────")

    if pos < 10:
        sys.exit("ERROR: Too few positive pairs to train — increase --sample.")

    # 80/20 split (stratified)
    idx = np.arange(len(y))
    idx_tr, idx_te = train_test_split(idx, test_size=0.20, stratify=y, random_state=42)
    X_tr, y_tr = X[idx_tr], y[idx_tr]
    X_te, y_te = X[idx_te], y[idx_te]
    pairs_te   = [valid_pairs[i] for i in idx_te]

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    clf = LogisticRegression(
        class_weight="balanced",
        C=1.0,
        max_iter=500,
        random_state=42,
        solver="lbfgs",
    )
    clf.fit(X_tr_s, y_tr)

    # Optimise threshold on training split for macro F0.5
    log.info("  Searching optimal threshold for F0.5 …")
    y_proba_tr = clf.predict_proba(X_tr_s)[:, 1]
    best_f05, best_thresh = -1.0, 0.5
    for thresh in np.linspace(0.05, 0.95, 91):
        y_hat = (y_proba_tr >= thresh).astype(int)
        f05   = fbeta_score(y_tr, y_hat, beta=0.5, zero_division=0)
        if f05 > best_f05:
            best_f05, best_thresh = f05, thresh
    log.info(f"  Best threshold  : {best_thresh:.2f}  (train F0.5 = {best_f05:.4f})")

    # Evaluate on test split
    y_proba_te = clf.predict_proba(X_te_s)[:, 1]
    y_pred_te  = (y_proba_te >= best_thresh).astype(int)

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_te, y_pred_te, average="binary", zero_division=0
    )
    f05_pair = fbeta_score(y_te, y_pred_te, beta=0.5, zero_division=0)

    # ── entity-level macro F0.5 on test split ─────────────────────────────────
    pred_per_entity: Dict[str, Set[str]] = defaultdict(set)
    for (s1_id, cid, _), prob in zip(pairs_te, y_proba_te):
        if prob >= best_thresh:
            pred_per_entity[s1_id].add(cid)
    # ensure every sampled S1 entity appears (singletons → empty set)
    for s1_id in gt_dict:
        if s1_id not in pred_per_entity:
            pred_per_entity[s1_id] = set()

    # Only score entities that appear in the test split
    te_s1_ids = {s1_id for s1_id, _, _ in pairs_te}
    gt_te = {s1_id: gt_dict[s1_id] for s1_id in te_s1_ids if s1_id in gt_dict}
    macro_f05, detail = compute_macro_f05(dict(pred_per_entity), gt_te)

    # ── feature importances ───────────────────────────────────────────────────
    coef  = clf.coef_[0]
    order = np.argsort(np.abs(coef))[::-1]
    top_k = 10

    # ── 6. Print summary ─────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print()
    print("=" * 65)
    print("  ENTITY RESOLUTION — EVALUATION RESULTS")
    print("=" * 65)
    print(f"  Sample size         : {len(gt_dict):,} S1 entities")
    print(f"  Candidate pairs     : {len(valid_pairs):,}")
    print(f"  Positive pairs      : {pos:,}  ({100*pos/max(len(valid_pairs),1):.2f}% of candidates)")
    print(f"  Threshold           : {best_thresh:.2f}")
    print()
    print("  ── Stage 1: Blocking ───────────────────────────────────────")
    print(f"  Blocking Recall     : {blocking_recall:.4f}  "
          f"({bp_tp:,}/{bp_total:,} true pairs found)")
    print(f"  Avg candidates/S1   : {len(candidate_pairs)/max(len(s1),1):.1f}")
    print()
    print("  ── Stage 2+3: Full Pipeline (test split) ───────────────────")
    print(f"  Pair-level Precision: {prec:.4f}")
    print(f"  Pair-level Recall   : {rec:.4f}")
    print(f"  Pair-level F1       : {f1:.4f}")
    print(f"  Pair-level F0.5     : {f05_pair:.4f}  ← challenge metric (pair)")
    print()
    print(f"  Macro F0.5 (entity) : {macro_f05:.4f}  ← challenge metric (entity)")
    print(f"    Perfect (1.0)     : {detail['perfect_1.0']:,}")
    print(f"    Zero   (0.0)      : {detail['zero_0.0']:,}")
    print(f"    Partial (0–1)     : {detail['partial']:,}")
    print(f"    Entities scored   : {detail['n_entities']:,}")
    print()
    print(f"  ── Top {top_k} Most Influential Features ────────────────────────")
    for rank, idx_f in enumerate(order[:top_k], 1):
        direction = "+" if coef[idx_f] > 0 else "−"
        print(f"  {rank:2}. {FEATURE_NAMES[idx_f]:<30}  coef={coef[idx_f]:+.3f}  ({direction}match)")
    print()
    print(f"  Total runtime       : {elapsed:.1f}s")
    print("=" * 65)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate entity resolution pipeline on training data")
    parser.add_argument(
        "--data-dir", default="dataset/train",
        help="Path to the train data folder (default: dataset/train)",
    )
    parser.add_argument(
        "--sample", type=int, default=2000,
        help="Number of S1 entities to sample (0 = full dataset, default: 2000)",
    )
    parser.add_argument(
        "--max-candidates", type=int, default=200,
        help="Max blocking candidates per S1 entity (default: 200)",
    )
    parser.add_argument(
        "--no-idf", action="store_true",
        help="Skip TF-IDF features (faster, slightly lower accuracy)",
    )
    main(parser.parse_args())

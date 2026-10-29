"""Step 5: build the relevance benchmark: 100 anchor abstracts and 800 anchor-candidate pairs.

Anchors. Titles are matched against 19 keyword strata covering the HRS content areas (Sonnega
et al. 2014, Box 2 and Table 2, plus later modules from the 2020-2022 codebooks); titles matching
none form a 20th stratum. Five anchors are drawn per stratum, rarest stratum first, in one fixed
random order (seed 42); no abstract is drawn twice.

Candidates. For each anchor every model retrieves its 50 nearest neighbours (cosine).
  hits      = number of models with the candidate in their top 10
  mean_rank = mean rank over the 10 models (51 when outside a model's top 50)
Eight pairs per anchor: 3 "high" (hits >= 3), 2 "low" (hits 1-2), 2 "semi" (hits 0, best
mean_rank among positions 11-50: semi-hard negatives) and 1 "random" (outside every top 50, a
sanity check left out of the accuracy). An unfilled slot borrows from the next stratum. Pooling
candidates from all models keeps the benchmark from favouring any one of them.

Input : data/corpus.csv, data/embeddings/
Output: data/validation/anchors.csv, strata_summary.csv,
        pairs.csv          pair metadata (stratum, hits, which models retrieved it)
        pairs_to_rate.csv  the same pairs, shuffled, with texts and no retrieval information

Usage: python scripts/05_build_pairs.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hrs_topics import config  # noqa: E402
from hrs_topics.data import load_corpus, load_embedding  # noqa: E402

PER_STRATUM = 5
K_TOP, K_POOL = 10, 50
QUOTA = [("high", 3), ("low", 2), ("semi", 2)]

# stratum id, name, keyword regex matched at word starts in the lower-cased title
STRATA = [
    ("B", "Demographics and background",
     r"education|educational attainment|race\b|racial|ethnic|hispanic|latino|black|african american|immigra|nativity|foreign-born|veteran|military|religio|marital status|marriage|married"),
    ("C", "Physical health and health behaviours",
     r"chronic|hypertension|diabetes|cancer|heart|cardiovascular|stroke|arthritis|lung disease|copd|pain\b|falls?\b|vision|visual|hearing|sleep|obesity|body mass|bmi\b|smok|alcohol|drinking|physical activity|exercise|health behavio|preventive|screening|vaccin|self-rated health|multimorbidity|frailty|cannabis|marijuana|opioid|substance use"),
    ("D", "Cognition",
     r"cogniti|memory|dementia|alzheimer|recall\b|hcap\b|adams\b"),
    ("EF", "Family structure and transfers",
     r"child\b|children|adult child|parent|sibling|grandchild|grandparent|intergenerational|transfers?\b|family|families"),
    ("G", "Functional limitations and helpers",
     r"functional|activities of daily living|adls?\b|iadls?\b|disabilit|mobility|helpers?\b|caregiv|informal care|care recipient"),
    ("H", "Housing",
     r"housing|homeowner|home ownership|mortgage|home equity|living arrangement|relocat|residential"),
    ("J", "Employment, pensions, and retirement",
     r"employ|labor force|labour force|work|job|occupation|earnings|wage|retire|pension|401\(k\)|annuit|social security"),
    ("M", "Disability programs",
     r"disability insurance|disability benefit|ssdi\b|supplemental security income|work disability|work limitation|disability program|disability application"),
    ("N", "Health services and insurance",
     r"medicare|medicaid|health insurance|insurance coverage|uninsured|health care|healthcare|hospital|nursing home|long-term care|out-of-pocket|prescription|physician|medical expenditure|health service"),
    ("P", "Expectations",
     r"expectation|subjective probabilit|subjective survival|subjective life expectancy"),
    ("Q", "Assets and income",
     r"wealth|assets?\b|income|saving|debt|financial|net worth|stock|portfolio|consumption|poverty|economic"),
    ("S", "Widowhood and divorce",
     r"widow|bereave|divorc|spousal death|death of a spouse|spousal loss|marital dissolution"),
    ("T", "Wills and life insurance",
     r"estate planning|life insurance|bequest|inheritance|testament"),
    ("W", "Internet use",
     r"internet|digital|computer|technology|online|e-mail|email|social media|smartphone"),
    ("COV", "COVID-19",
     r"covid|pandemic|coronavirus|sars-cov-2"),
    ("LB", "Psychosocial well-being and social relationships",
     r"depress|well-being|wellbeing|life satisfaction|loneliness|lonely|social isolation|social support|social network|social engagement|social relationship|volunteer|purpose in life|optimism|pessimism|personality|discrimination|stress|trauma|anxiety|mastery|subjective age|perceptions of aging|neighborhood|neighbourhood|mental health|childhood|early life|early-life"),
    ("X", "End of life and mortality",
     r"end-of-life|end of life|hospice|advance directive|advance care|place of death|dying|last year of life|mortality|death"),
    ("I", "Physical measures and biomarkers",
     r"biomarker|blood pressure|grip strength|gait|walking speed|lung function|peak flow|waist|c-reactive|crp\b|inflammat|cystatin|hba1c|glycated|cholesterol|blood|allostatic|cortisol|interleukin|il-6"),
    ("GEN", "Genetics and epigenetics",
     r"genetic|genes?\b|genom|gwas|polygenic|apoe|methylation|epigenetic|telomere|heritab|allele|snps?\b"),
]


def draw_anchors(corpus):
    pool = corpus[corpus.title.notna() & corpus.abstract.notna() & (corpus.abstract.str.len() > 0)]
    pool = pool.reset_index(drop=True)
    low = pool.title.str.lower()
    match = pd.DataFrame({sid: low.str.contains(r"\b(?:" + rx + r")", regex=True) for sid, _, rx in STRATA})
    match["NONE"] = ~match.any(axis=1)
    names = {sid: name for sid, name, _ in STRATA} | {"NONE": "Title matches no stratum"}

    order = np.random.default_rng(config.SEED).permutation(len(pool))   # one fixed random order
    sizes = match.sum()
    chosen, rows = set(), []
    for sid in sorted(sizes.index, key=lambda s: (sizes[s], s)):      # rarest stratum first
        picks = [i for i in order if match.at[i, sid] and i not in chosen][:PER_STRATUM]
        chosen.update(picks)
        rows += [(i, sid) for i in picks]
    anchors = pd.DataFrame([{"document_id": pool.document_id[i], "year": pool.year[i], "title": pool.title[i],
                             "drawn_for_stratum": sid,
                             "all_matched_strata": ";".join(s for s, _, _ in STRATA if match.at[i, s])}
                            for i, sid in rows])
    summary = pd.DataFrame({"stratum": list(names), "name": list(names.values()),
                            "titles_matched": [int(sizes[s]) for s in names],
                            "anchors": [int((anchors.drawn_for_stratum == s).sum()) for s in names]})
    return anchors, summary


def draw_pairs(corpus, anchors):
    pos = {d: i for i, d in enumerate(corpus.document_id)}
    apos = np.array([pos[d] for d in anchors.document_id])
    n, A = len(corpus), len(apos)

    rank = np.full((len(config.MODELS), A, n), K_POOL + 1, dtype=np.int16)
    for m, model in enumerate(config.MODELS):
        X = load_embedding(model)
        _, idx = NearestNeighbors(n_neighbors=K_POOL + 1, metric="cosine", algorithm="brute").fit(X).kneighbors(X[apos])
        for a in range(A):
            row = [j for j in idx[a] if j != apos[a]][:K_POOL]
            rank[m, a, row] = np.arange(1, len(row) + 1)
        print("retrieved", model, flush=True)

    rng = np.random.default_rng(config.SEED)
    pairs = []
    for a in range(A):
        r = rank[:, a, :]
        hits, mean_rank = (r <= K_TOP).sum(0), r.mean(0)
        in_pool = (r <= K_POOL).any(0)
        in_pool[apos[a]] = False
        semi = np.flatnonzero(in_pool & (hits == 0))
        cand = {"high": np.flatnonzero(in_pool & (hits >= 3)),
                "low": np.flatnonzero(in_pool & (hits >= 1) & (hits <= 2)),
                "semi": semi[np.argsort(mean_rank[semi], kind="stable")][:K_POOL - K_TOP]}
        taken, carry = [], 0
        for stratum, quota in QUOTA:
            need = quota + carry
            pick = rng.choice(cand[stratum], size=min(need, len(cand[stratum])), replace=False) \
                if len(cand[stratum]) else []
            taken += [(int(c), stratum) for c in pick]
            carry = need - len(pick)
        assert carry == 0, f"anchor {a}: not enough candidates"
        outside = np.flatnonzero(~in_pool)
        taken.append((int(rng.choice(outside[outside != apos[a]])), "random"))
        pairs += [{"anchor_document_id": corpus.document_id[apos[a]], "candidate_document_id": corpus.document_id[c],
                   "stratum": s, "hits_top10": int(hits[c]), "mean_rank": round(float(mean_rank[c]), 2),
                   "retrieved_by": ";".join(k for m, k in enumerate(config.MODELS) if r[m, c] <= K_TOP)}
                  for c, s in taken]

    pairs = pd.DataFrame(pairs).sample(frac=1, random_state=config.SEED).reset_index(drop=True)
    pairs.insert(0, "pair_id", [f"pair_{i:04d}" for i in range(1, len(pairs) + 1)])
    return pairs


def main():
    corpus = load_corpus()
    config.VALIDATION.mkdir(parents=True, exist_ok=True)
    anchors, summary = draw_anchors(corpus)
    anchors.to_csv(config.VALIDATION / "anchors.csv", index=False)
    summary.to_csv(config.VALIDATION / "strata_summary.csv", index=False)

    pairs = draw_pairs(corpus, anchors)
    pairs.to_csv(config.VALIDATION / "pairs.csv", index=False)
    text = corpus.set_index("document_id")[["title", "abstract"]]
    rate = pairs[["pair_id", "anchor_document_id", "candidate_document_id"]].copy()
    for side in ("anchor", "candidate"):
        t = text.loc[rate[f"{side}_document_id"]]
        rate[f"{side}_title"], rate[f"{side}_abstract"] = t.title.values, t.abstract.values
    rate.to_csv(config.VALIDATION / "pairs_to_rate.csv", index=False)
    print(len(anchors), "anchors,", len(pairs), "pairs:", pairs.stratum.value_counts().to_dict())


if __name__ == "__main__":
    main()

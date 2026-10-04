"""
reviewer_controls.py  --  helper functions for SECTION E of
"GSE149383 / GSE150949 drug-tolerance trajectory analysis".

Only needs packages that the main notebook already installs
(numpy, scipy, pandas, scikit-learn, matplotlib).  scikit-image and
`scrublet` are used if they happen to be importable, but are NOT required.

Every function below is called from Section E (cells E1-E8) of the notebook.
p-value convention used everywhere:  p = (r + 1) / (M + 1)
    r = number of null draws >= (or <=) the observed value, M = number of draws.
"""
import math
import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
__version__ = "1.0"

# ----------------------------------------------------------------------------
# small utilities
# ----------------------------------------------------------------------------
N_RADII_DEFAULT = 20
SCALING_FRAC_DEFAULT = (0.15, 0.60)   # same fitting window as the original estimator
LAST_DOUBLET_METHOD = {}              # filled by scrublet_by_library(): library -> method used


def _p_plus_one(n_ge, n_total):
    return (n_ge + 1.0) / (n_total + 1.0)


def benjamini_hochberg(pvals):
    """Benjamini-Hochberg FDR q-values, returned in the ORIGINAL order of `pvals`."""
    p = np.asarray(pvals, dtype=float)
    q = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    if ok.sum() == 0:
        return q
    pv = p[ok]
    m = len(pv)
    order = np.argsort(pv)
    ranked = pv[order] * m / (np.arange(m) + 1.0)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]     # enforce monotonicity
    out = np.empty(m)
    out[order] = np.clip(ranked, 0.0, 1.0)
    q[ok] = out
    return q


def depth_match_indices(depth_a, depth_b, n_bins=10, seed=0):
    """
    Depth-matching identical in spirit to depth_match_two_groups() in the main notebook:
    joint depth-quantile bins, then down-sample the larger group to the smaller one inside each bin.
    Returns (idx_a, idx_b) = integer positions INTO depth_a and depth_b.
    """
    rng = np.random.default_rng(seed)
    da = np.asarray(depth_a, dtype=float)
    db = np.asarray(depth_b, dtype=float)
    edges = np.quantile(np.concatenate([da, db]), np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1.0
    keep_a, keep_b = [], []
    for i in range(n_bins):
        in_a = np.where((da >= edges[i]) & (da < edges[i + 1]))[0]
        in_b = np.where((db >= edges[i]) & (db < edges[i + 1]))[0]
        m = min(len(in_a), len(in_b))
        if m > 0:
            keep_a.extend(rng.choice(in_a, m, replace=False))
            keep_b.extend(rng.choice(in_b, m, replace=False))
    return np.asarray(keep_a, dtype=int), np.asarray(keep_b, dtype=int)


# ----------------------------------------------------------------------------
# classifier controls (review point 2)
# ----------------------------------------------------------------------------
def _make_clf(scale):
    lr = LogisticRegression(max_iter=1000)
    return make_pipeline(StandardScaler(), lr) if scale else lr


def _cv_accuracy(X, y, seed, n_splits=5, scale=False):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accs = []
    for tr, te in skf.split(X, y):
        m = _make_clf(scale).fit(X[tr], y[tr])
        accs.append((m.predict(X[te]) == y[te]).mean())
    return float(np.mean(accs)), float(np.std(accs))


def _cv_auc(X, y, seed, n_splits=5, scale=False):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    score = np.zeros(len(y))
    for tr, te in skf.split(X, y):
        m = _make_clf(scale).fit(X[tr], y[tr])
        score[te] = m.predict_proba(X[te])[:, 1]
    return float(roc_auc_score(y, score))


def _one_perm(X, y, seed, n_splits, scale):
    rng = np.random.default_rng(seed)
    yp = rng.permutation(y)
    try:
        return _cv_accuracy(X, yp, seed=int(seed) % (2 ** 31 - 1), n_splits=n_splits, scale=scale)[0]
    except Exception:
        return np.nan


def _perm_p(X, y, observed, n_perm, seed, n_splits=5, scale=False):
    """(r+1)/(M+1) permutation p-value for cross-validated accuracy, labels shuffled n_perm times."""
    seeds = [seed * 1_000_003 + i + 1 for i in range(n_perm)]
    try:
        from joblib import Parallel, delayed
        null = Parallel(n_jobs=-1)(delayed(_one_perm)(X, y, s, n_splits, scale) for s in seeds)
    except Exception:
        null = [_one_perm(X, y, s, n_splits, scale) for s in seeds]
    null = np.asarray(null, dtype=float)
    null = null[np.isfinite(null)]
    return float(_p_plus_one((null >= observed - 1e-12).sum(), len(null))), float(null.mean()) if len(null) else np.nan


def classifier_control_suite(X, obs, group_col, pairs, n_pcs=20, n_perm=1000, seed=0,
                             depth_col="total_counts", n_splits=5, match_depth=True):
    """
    For each (group_a, group_b, label) in `pairs` run, on depth-matched cells:
      * PC classifier      : logistic regression on the first `n_pcs` PCs (as in the main analysis)
      * covariate-only     : logistic regression on log(total counts), log(genes detected), mito fraction
                             (whichever exist in `obs`) -> measures how much of the separation a
                             library-level technical signal alone can reproduce
    plus, on the UNMATCHED cells of the pair, the covariate-only balanced accuracy.
    Permutation p-values use p = (r+1)/(M+1) with `n_perm` label shuffles.
    `X` rows must be in the same order as `obs` rows.
    """
    X = np.asarray(X)
    if X.shape[0] != len(obs):
        raise ValueError(f"X has {X.shape[0]} rows but obs has {len(obs)} rows")
    grp = obs[group_col].astype(str).values
    depth = np.asarray(obs[depth_col], dtype=float) if depth_col in obs.columns else None

    cov_cols = []
    cov_mat = []
    for col, use_log in [("total_counts", True), ("n_genes_detected", True), ("mito_fraction", False)]:
        if col in obs.columns:
            v = np.asarray(obs[col], dtype=float)
            cov_mat.append(np.log1p(v) if use_log else v)
            cov_cols.append(col)
    COV = np.column_stack(cov_mat) if cov_mat else None

    k = min(n_pcs, X.shape[1])
    rows = []
    for (ga, gb, label) in pairs:
        row = {"comparison": label, "group_a": ga, "group_b": gb, "n_pcs": k,
               "n_perm": n_perm, "min_attainable_p": 1.0 / (n_perm + 1)}
        ia_all = np.where(grp == str(ga))[0]
        ib_all = np.where(grp == str(gb))[0]
        row["n_a_all"], row["n_b_all"] = len(ia_all), len(ib_all)
        if min(len(ia_all), len(ib_all)) < 10:
            row["note"] = "skipped: fewer than 10 cells in a group"
            rows.append(row)
            continue

        if match_depth and depth is not None:
            ja, jb = depth_match_indices(depth[ia_all], depth[ib_all], seed=seed)
            sel_a, sel_b = ia_all[ja], ib_all[jb]
        else:
            sel_a, sel_b = ia_all, ib_all
        row["n_a_matched"], row["n_b_matched"] = len(sel_a), len(sel_b)
        if min(len(sel_a), len(sel_b)) < 10:
            row["note"] = "skipped: fewer than 10 cells per group after depth matching"
            rows.append(row)
            continue

        sel = np.concatenate([sel_a, sel_b])
        y = np.concatenate([np.zeros(len(sel_a), int), np.ones(len(sel_b), int)])

        # --- PC classifier
        Xp = X[sel][:, :k]
        acc, acc_sd = _cv_accuracy(Xp, y, seed, n_splits)
        row["acc_pc"], row["acc_pc_sd"] = acc, acc_sd
        row["auc_pc"] = _cv_auc(Xp, y, seed, n_splits)
        row["perm_p_pc"], row["perm_null_mean_pc"] = _perm_p(Xp, y, acc, n_perm, seed, n_splits)

        # --- covariate-only classifier (matched cells)
        if COV is not None:
            Xc = COV[sel]
            cacc, cacc_sd = _cv_accuracy(Xc, y, seed, n_splits, scale=True)
            row["acc_cov_only"], row["acc_cov_only_sd"] = cacc, cacc_sd
            row["auc_cov_only"] = _cv_auc(Xc, y, seed, n_splits, scale=True)
            row["perm_p_cov_only"], _ = _perm_p(Xc, y, cacc, n_perm, seed, n_splits, scale=True)
            # --- covariate-only on all (unmatched) cells of the pair, balanced accuracy
            sel_all = np.concatenate([ia_all, ib_all])
            y_all = np.concatenate([np.zeros(len(ia_all), int), np.ones(len(ib_all), int)])
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            pred = np.zeros(len(y_all), int)
            for tr, te in skf.split(COV[sel_all], y_all):
                pred[te] = _make_clf(True).fit(COV[sel_all][tr], y_all[tr]).predict(COV[sel_all][te])
            row["bal_acc_cov_only_unmatched"] = float(balanced_accuracy_score(y_all, pred))
            row["covariates_used"] = ",".join(cov_cols)
        row["note"] = ""
        rows.append(row)
    return pd.DataFrame(rows)


def pc_count_sensitivity(X, y, pc_counts=(5, 10, 15, 20, 30, 50), n_perm=200, seed=0, n_splits=5):
    """Cross-validated accuracy / AUC / permutation p of the PC classifier as a function of #PCs used."""
    X = np.asarray(X)
    y = np.asarray(y).astype(int)
    rows = []
    for k in pc_counts:
        kk = min(int(k), X.shape[1])
        Xk = X[:, :kk]
        acc, sd = _cv_accuracy(Xk, y, seed, n_splits)
        p, null_mean = _perm_p(Xk, y, acc, n_perm, seed, n_splits)
        rows.append({"n_pcs": kk, "acc": acc, "acc_sd": sd, "auc": _cv_auc(Xk, y, seed, n_splits),
                     "perm_null_mean": null_mean, "perm_p": p, "n_perm": n_perm})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# correlation dimension, done properly (review point 5)
# ----------------------------------------------------------------------------
def _window(n_radii, scaling_frac):
    lo, hi = int(n_radii * scaling_frac[0]), int(n_radii * scaling_frac[1])
    if hi <= lo:
        hi = lo + 2
    return lo, hi


def _corr_sum(dists_sorted, radii):
    # C(r) = fraction of the i<j pairs with distance < r   (i != j, each pair counted once)
    return np.searchsorted(dists_sorted, radii, side="left") / float(len(dists_sorted))


def _slope(dists_sorted, radii, scaling_frac):
    C = _corr_sum(dists_sorted, radii)
    lo, hi = _window(len(radii), scaling_frac)
    r, c = radii[lo:hi], C[lo:hi]
    ok = c > 0
    if ok.sum() < 2:
        return np.nan
    return float(np.polyfit(np.log(r[ok]), np.log(c[ok]), 1)[0])


def _subsample(P, n, rng):
    return P[rng.choice(len(P), size=n, replace=False)]        # WITHOUT replacement -> no duplicate points


def gp_dimension(points, radii=None, scaling_frac=SCALING_FRAC_DEFAULT, n=None, rng=None,
                 n_radii=N_RADII_DEFAULT):
    """
    Grassberger-Procaccia correlation dimension using i != j pairs only (scipy pdist returns each i<j pair
    once) and, if `n` is given, a sub-sample drawn WITHOUT replacement.  If `radii` is None a private
    radius grid (5th-95th percentile of the distances) is used, as in the original estimator.
    """
    P = np.asarray(points, dtype=float)
    rng = rng if rng is not None else np.random.default_rng(0)
    if n is not None:
        P = _subsample(P, n, rng)
    d = np.sort(pdist(P))
    if radii is None:
        dpos = d[d > 0]
        lo_, hi_ = np.percentile(dpos, [5, 95])
        radii = np.logspace(np.log10(lo_), np.log10(hi_), n_radii)
    return _slope(d, np.asarray(radii), scaling_frac)


def common_radii(emb, n, n_radii=N_RADII_DEFAULT, pct=(5, 95), n_rep=5, seed=0):
    """ONE radius grid shared by every condition (pooled 5th-95th percentile of equal-n pairwise distances)."""
    rng = np.random.default_rng(seed)
    pooled = []
    for c, P in emb.items():
        P = np.asarray(P, dtype=float)
        m = min(n, len(P))
        for _ in range(n_rep):
            pooled.append(pdist(_subsample(P, m, rng)))
    d = np.concatenate(pooled)
    d = d[d > 0]
    lo_, hi_ = np.percentile(d, pct)
    return np.logspace(np.log10(lo_), np.log10(hi_), n_radii)


def d2_equal_n(emb, n, n_rep=200, seed=0, n_radii=N_RADII_DEFAULT, scaling_frac=SCALING_FRAC_DEFAULT):
    """
    D2 for every condition at the SAME number of cells `n`, on ONE shared radius grid, i != j, no duplicates.
    Returns (table, draws, radii):  table indexed by condition (mean, sd, 95% interval over `n_rep`
    random equal-n draws), draws = {condition: array of D2 draws}, radii = the shared grid.
    """
    short = {c: len(P) for c, P in emb.items() if len(P) < n}
    if short:
        raise ValueError(f"n={n} exceeds available cells for: {short}")
    radii = common_radii(emb, n, n_radii=n_radii, seed=seed)
    draws, rows = {}, []
    for ci, (c, P) in enumerate(emb.items()):
        rng = np.random.default_rng(seed * 7919 + ci + 1)
        P = np.asarray(P, dtype=float)
        vals = np.array([_slope(np.sort(pdist(_subsample(P, n, rng))), radii, scaling_frac)
                         for _ in range(n_rep)])
        draws[c] = vals
        v = vals[np.isfinite(vals)]
        rows.append({"condition": c, "n_available": len(P), "n_used": n, "n_rep": n_rep,
                     "D2_mean": v.mean(), "D2_sd": v.std(ddof=1) if len(v) > 1 else np.nan,
                     "ci_lo": np.percentile(v, 2.5), "ci_hi": np.percentile(v, 97.5),
                     "note": "uses ALL cells every draw: interval has no sampling spread" if len(P) == n else ""})
    return pd.DataFrame(rows).set_index("condition"), draws, radii


def plot_loglog_correlation_sum(emb, radii, n, seed=0, ax=None, scaling_frac=SCALING_FRAC_DEFAULT):
    """log C(r) vs log r for every condition (one equal-n draw each) with the fitting window shaded."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    radii = np.asarray(radii)
    lo, hi = _window(len(radii), scaling_frac)
    cmap = plt.get_cmap("tab10")
    for ci, (c, P) in enumerate(emb.items()):
        rng = np.random.default_rng(seed * 7919 + ci + 1)
        d = np.sort(pdist(_subsample(np.asarray(P, float), min(n, len(P)), rng)))
        C = _corr_sum(d, radii)
        ok = C > 0
        ax.plot(np.log(radii[ok]), np.log(C[ok]), "o-", ms=3, lw=1, color=cmap(ci % 10), label=str(c))
    ax.axvspan(np.log(radii[lo]), np.log(radii[hi - 1]), color="grey", alpha=0.15,
               label="fitting window (D2 = slope)")
    ax.set_xlabel("log r")
    ax.set_ylabel("log C(r)")
    ax.set_title(f"Correlation sum, shared radii, n = {n} cells per condition")
    ax.legend(fontsize=7, ncol=2)
    return ax


def d2_difference_test(A, B, n, n_rep=1000, seed=0, n_radii=N_RADII_DEFAULT,
                       scaling_frac=SCALING_FRAC_DEFAULT):
    """
    Direct test of D2(A) - D2(B) at equal n on a shared radius grid.
    * estimate / interval: mean and 2.5-97.5% range of the difference over `n_rep` paired random draws
    * p-value: label-permutation null (pool A and B, re-split at random, same procedure), two-sided,
               p = (r+1)/(M+1)
    """
    A, B = np.asarray(A, float), np.asarray(B, float)
    n_eff = int(min(n, len(A), len(B)))
    radii = common_radii({"A": A, "B": B}, n_eff, n_radii=n_radii, seed=seed)
    rng = np.random.default_rng(seed + 12345)
    diffs = np.empty(n_rep)
    for i in range(n_rep):
        a = _slope(np.sort(pdist(_subsample(A, n_eff, rng))), radii, scaling_frac)
        b = _slope(np.sort(pdist(_subsample(B, n_eff, rng))), radii, scaling_frac)
        diffs[i] = a - b
    d_ok = diffs[np.isfinite(diffs)]
    obs = float(d_ok.mean())

    pooled = np.vstack([A, B])
    n_null = int(min(n_eff, len(pooled) // 2))
    null = np.empty(n_rep)
    for i in range(n_rep):
        perm = rng.permutation(len(pooled))
        a = _slope(np.sort(pdist(pooled[perm[:n_null]])), radii, scaling_frac)
        b = _slope(np.sort(pdist(pooled[perm[n_null:2 * n_null]])), radii, scaling_frac)
        null[i] = a - b
    null = null[np.isfinite(null)]
    p = _p_plus_one((np.abs(null) >= abs(obs)).sum(), len(null))
    return {"n_per_group": n_eff, "n_rep": n_rep,
            "diff_mean": round(obs, 4),
            "diff_lo_2.5": round(float(np.percentile(d_ok, 2.5)), 4),
            "diff_hi_97.5": round(float(np.percentile(d_ok, 97.5)), 4),
            "perm_null_sd": round(float(null.std(ddof=1)), 4),
            "perm_p_two_sided": round(float(p), 4)}


def d2_pc_sensitivity(emb, pc_counts=(10, 15, 20, 30), n=None, n_rep=50, seed=0):
    """D2 at equal n as a function of how many PCs are used. Long table: condition, n_pcs, D2_mean, D2_sd."""
    if n is None:
        n = min(len(P) for P in emb.values())
    rows = []
    for k in pc_counts:
        sub = {c: np.asarray(P)[:, :min(int(k), np.asarray(P).shape[1])] for c, P in emb.items()}
        tab, _, _ = d2_equal_n(sub, n=n, n_rep=n_rep, seed=seed)
        for c in tab.index:
            rows.append({"condition": c, "n_pcs": int(k), "D2_mean": tab.loc[c, "D2_mean"],
                         "D2_sd": tab.loc[c, "D2_sd"], "n_used": n})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# depth controls (review point 8)
# ----------------------------------------------------------------------------
def radial_depth_correlation(pcs, log_depth, n_pcs=2):
    """Spearman correlation between the distance from the origin in the first `n_pcs` PCs and log depth."""
    pcs = np.asarray(pcs, dtype=float)
    radius = np.linalg.norm(pcs[:, :n_pcs], axis=1)
    rho, p = stats.spearmanr(radius, np.asarray(log_depth, dtype=float))
    return float(rho), float(p)


def binomial_thin_counts(X, target, seed=0):
    """
    Binomial thinning of a raw-count matrix: every count of a cell is kept with probability
    target / (cell total), so each retained cell has expected total = `target`.
    Cells with total < target cannot be thinned and are flagged keep = False.
    Returns (X_thinned [same shape, sparse CSR float32], keep [bool array]).
    """
    rng = np.random.default_rng(seed)
    X = sp.csr_matrix(X)
    totals = np.asarray(X.sum(axis=1)).ravel()
    keep = totals >= target
    p_row = np.where(keep, target / np.maximum(totals, 1.0), 1.0)
    p_nnz = np.repeat(p_row, np.diff(X.indptr))
    Xt = X.copy()
    Xt.data = rng.binomial(np.rint(X.data).astype(np.int64), p_nnz).astype(np.float32)
    Xt.eliminate_zeros()
    return Xt, keep


def _scrublet_like(counts, expected_rate=0.06, sim_ratio=2.0, n_pcs=30, hvg_pctl=85, seed=0):
    """
    Simulated-doublet / kNN doublet score in the spirit of Scrublet (Wolock et al. 2019), written with
    scikit-learn only.  Returns (scores, threshold).
    """
    from sklearn.decomposition import TruncatedSVD
    from sklearn.neighbors import NearestNeighbors

    rng = np.random.default_rng(seed)
    C = sp.csr_matrix(counts, dtype=np.float64)
    n = C.shape[0]
    n_sim = int(sim_ratio * n)
    i1 = rng.integers(0, n, n_sim)
    i2 = rng.integers(0, n, n_sim)
    same = i1 == i2
    i2[same] = (i2[same] + 1) % n
    S = C[i1] + C[i2]

    tot_obs = np.asarray(C.sum(1)).ravel()
    target = np.median(tot_obs)
    Cn = sp.diags(target / np.maximum(tot_obs, 1)) @ C
    Sn = sp.diags(target / np.maximum(np.asarray(S.sum(1)).ravel(), 1)) @ S

    # gene filter (expressed in >=3 cells, >=3 counts), then the most variable genes (Fano-like)
    ncell = np.asarray((C > 0).sum(0)).ravel()
    ntot = np.asarray(C.sum(0)).ravel()
    mean = np.asarray(Cn.mean(0)).ravel()
    sq = np.asarray(Cn.multiply(Cn).mean(0)).ravel()
    var = sq - mean ** 2
    fano = var / np.maximum(mean, 1e-12)
    cand = (ncell >= 3) & (ntot >= 3) & (mean > 0)
    cut = np.percentile(fano[cand], hvg_pctl) if cand.sum() else np.inf
    genes = np.where(cand & (fano >= cut))[0]
    if len(genes) < 10:
        genes = np.where(cand)[0]

    def z(M):
        M = np.log1p(M[:, genes].toarray())
        return M
    Zo, Zs = z(Cn), z(Sn)
    mu, sd = Zo.mean(0), Zo.std(0) + 1e-9
    Zo, Zs = (Zo - mu) / sd, (Zs - mu) / sd
    Zo, Zs = np.clip(Zo, -10, 10), np.clip(Zs, -10, 10)
    k_pc = int(min(n_pcs, len(genes) - 1, n - 1))
    svd = TruncatedSVD(n_components=k_pc, random_state=seed).fit(Zo)
    Po, Ps = svd.transform(Zo), svd.transform(Zs)

    comb = np.vstack([Po, Ps])
    k = int(round(0.5 * math.sqrt(n)))
    k = max(k, 5)
    k_adj = int(round(k * (1 + n_sim / n)))
    k_adj = min(k_adj, comb.shape[0] - 1)
    nn = NearestNeighbors(n_neighbors=k_adj + 1).fit(comb)
    _, ind = nn.kneighbors(comb)
    ind = ind[:, 1:]                                   # drop self
    is_sim = np.r_[np.zeros(n, bool), np.ones(n_sim, bool)]
    q = is_sim[ind].mean(1)
    r = n_sim / n
    rho = expected_rate

    def score(qv):
        return (qv * rho / r) / (1 - rho - qv * (1 - rho - rho / r))
    s_all = np.clip(score(q), 0, None)
    obs_scores, sim_scores = s_all[:n], s_all[n:]

    thr = None
    try:
        from skimage.filters import threshold_minimum
        thr = float(threshold_minimum(sim_scores))
    except Exception:
        thr = None
    if thr is None or not np.isfinite(thr) or (obs_scores >= thr).mean() > 0.30 or (obs_scores >= thr).sum() == 0:
        thr = float(np.quantile(obs_scores, 1 - expected_rate))      # fall back to the expected rate
    return obs_scores, thr


def scrublet_by_library(counts, libraries, expected_doublet_rate=0.06, seed=0, prefer_package=True):
    """
    Doublet scores + calls computed SEPARATELY for each library on raw counts.
    Uses the real `scrublet` package when it imports and runs; otherwise uses the built-in
    Scrublet-style implementation above.  The method used for each library is stored in
    `reviewer_controls.LAST_DOUBLET_METHOD` (and printed).
    Returns (scores [float array], calls [bool array]).
    """
    C = sp.csr_matrix(counts)
    libs = np.asarray(libraries).astype(str)
    scores = np.zeros(C.shape[0], dtype=float)
    calls = np.zeros(C.shape[0], dtype=bool)
    LAST_DOUBLET_METHOD.clear()
    pkg = None
    if prefer_package:
        try:
            import scrublet as pkg  # noqa
        except Exception:
            pkg = None
    for lib in [str(l) for l in pd.unique(libs)]:
        idx = np.where(libs == lib)[0]
        sub = C[idx]
        if len(idx) < 50:
            LAST_DOUBLET_METHOD[lib] = "skipped (<50 cells)"
            continue
        done = False
        if pkg is not None:
            try:
                scrub = pkg.Scrublet(sub.tocsc().astype(np.float64), expected_doublet_rate=expected_doublet_rate,
                                     random_state=seed)
                s, c = scrub.scrub_doublets(verbose=False)
                s = np.asarray(s, dtype=float)
                if c is None:
                    thr = float(np.quantile(s, 1 - expected_doublet_rate))
                    c = s >= thr
                scores[idx], calls[idx] = s, np.asarray(c, dtype=bool)
                LAST_DOUBLET_METHOD[lib] = "scrublet package"
                done = True
            except Exception:
                done = False
        if not done:
            s, thr = _scrublet_like(sub, expected_rate=expected_doublet_rate, seed=seed)
            scores[idx], calls[idx] = s, s >= thr
            LAST_DOUBLET_METHOD[lib] = "built-in Scrublet-style (scikit-learn)"
    print("doublet method per library:", dict(LAST_DOUBLET_METHOD))
    return scores, calls


# ----------------------------------------------------------------------------
# 30-bin series, CCM / PE / TE controls (review point 4)
# ----------------------------------------------------------------------------
def order_key_bins(key, n_bins=30):
    """
    Bin the ordering key (day + within-day position) into `n_bins` equal-width bins, exactly as in the
    original analysis.  Returns (bin index per cell [same order as `key`], occupancy per bin, #empty bins).
    """
    key = np.asarray(key, dtype=float)
    edges = np.linspace(key.min(), key.max(), n_bins + 1)
    b_idx = np.clip(np.digitize(key, edges) - 1, 0, n_bins - 1)
    occ = np.bincount(b_idx, minlength=n_bins)
    return b_idx, occ, int((occ == 0).sum())


def binned_means(values, b_idx, n_bins=None, fill="nan"):
    """
    Mean of `values` within each bin.
      fill='drop'   -> occupied bins only (NO interpolation)
      fill='nan'    -> all bins, empty bins = NaN
      fill='interp' -> all bins, empty bins linearly interpolated (what the original analysis did)
    """
    values = np.asarray(values, dtype=float)
    b_idx = np.asarray(b_idx)
    if n_bins is None:
        n_bins = int(b_idx.max()) + 1
    sums = np.bincount(b_idx, weights=values, minlength=n_bins)
    cnt = np.bincount(b_idx, minlength=n_bins)
    out = np.full(n_bins, np.nan)
    occ = cnt > 0
    out[occ] = sums[occ] / cnt[occ]
    if fill == "drop":
        return out[occ]
    if fill == "interp":
        return pd.Series(out).interpolate(limit_direction="both").values
    return out


def ccm_settings_report(n_points, embed_dim=2, lag=1, train_percent=0.75):
    """The exact CCM settings implied by skccm for a series of `n_points` (for the Methods section)."""
    n_vec = n_points - lag * (embed_dim - 1)
    n_train = int(n_vec * train_percent)
    n_test = n_vec - n_train
    step = max(1, (n_train - 8) // 6)
    libs = list(range(8, n_train, step))
    return {"series_length": n_points, "embedding_dim": embed_dim, "lag": lag,
            "embedded_vectors": n_vec, "train_vectors": n_train, "test_vectors": n_test,
            "library_lengths": libs, "train_percent": train_percent}


def iaaft(x, rng, n_iter=100):
    """Iterative amplitude-adjusted Fourier-transform surrogate (keeps the value distribution AND the spectrum)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    xs = np.sort(x)
    amp = np.abs(np.fft.rfft(x))
    s = rng.permutation(x)
    for _ in range(n_iter):
        phases = np.angle(np.fft.rfft(s))
        s_new = np.fft.irfft(amp * np.exp(1j * phases), n=n)
        s_new = xs[np.argsort(np.argsort(s_new))]          # rank-remap to the original values
        if np.array_equal(s_new, s):
            break
        s = s_new
    return s


def twin_surrogate(x, rng, dim=2, lag=1, rec_rate=0.10):
    """
    Twin surrogate (Thiel et al. 2006): random walk through the delay-embedded state space that jumps
    between 'twin' states (states with identical recurrence neighbourhoods).  Preserves the deterministic
    structure while breaking the specific temporal alignment with the other series.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    N = n - (dim - 1) * lag
    if N < 6:
        return rng.permutation(x)
    E = np.column_stack([x[i * lag:i * lag + N] for i in range(dim)])
    D = squareform(pdist(E, metric="chebyshev"))
    eps = np.quantile(D[np.triu_indices(N, 1)], rec_rate)
    R = D <= eps
    groups = {}
    for i in range(N):
        groups.setdefault(R[i].tobytes(), []).append(i)
    twins = [groups[R[i].tobytes()] for i in range(N)]
    path = [int(rng.integers(N))]
    for _ in range(N - 1):
        j = path[-1]
        t = twins[j][int(rng.integers(len(twins[j])))]
        nxt = t + 1
        if nxt >= N:
            nxt = int(rng.integers(N))
        path.append(nxt)
    s = x[np.array(path)]
    tail = [x[min(path[-1] + k, n - 1)] for k in range(1, n - N + 1)]
    return np.concatenate([s, tail]) if tail else s


def matched_random_genes(gene_mean, gene_detect, target_idx, exclude, n_genes=300, tol=0.25, seed=0):
    """
    Indices of genes whose mean expression and detection fraction are both within +/- `tol` (relative) of
    the target gene's.  If fewer than `n_genes` qualify the tolerance is widened stepwise; if still short
    the closest genes are taken.  Target and `exclude` genes are never returned.
    """
    rng = np.random.default_rng(seed)
    gm, gd = np.asarray(gene_mean, float), np.asarray(gene_detect, float)
    banned = np.zeros(len(gm), bool)
    banned[list(exclude)] = True
    banned[target_idx] = True
    m0, d0 = gm[target_idx], gd[target_idx]
    t = tol
    cand = np.array([], dtype=int)
    while t <= 2.0:
        ok = (~banned) & (np.abs(gm - m0) <= t * m0) & (np.abs(gd - d0) <= t * d0)
        cand = np.where(ok)[0]
        if len(cand) >= n_genes:
            break
        t *= 1.5
    if len(cand) < n_genes:
        dist = np.abs(np.log((gm + 1e-9) / (m0 + 1e-9))) + np.abs(np.log((gd + 1e-9) / (d0 + 1e-9)))
        dist[banned] = np.inf
        cand = np.argsort(dist)[:n_genes]
        cand = cand[np.isfinite(dist[cand])]
    if len(cand) > n_genes:
        cand = rng.choice(cand, n_genes, replace=False)
    return np.asarray(cand, dtype=int)


def random_gene_ccm_control(observed, null_skills):
    """Where does the observed CCM skill sit among the skills of expression-matched random genes?"""
    v = np.asarray(null_skills, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return {"observed": float(observed), "n_null_genes": 0}
    return {"observed": float(observed), "null_mean": float(v.mean()), "null_sd": float(v.std(ddof=1)),
            "null_q95": float(np.quantile(v, 0.95)), "null_max": float(v.max()),
            "percentile_of_observed": float((v < observed).mean() * 100),
            "p_one_sided": float(_p_plus_one((v >= observed).sum(), len(v))),
            "n_null_genes": int(len(v))}


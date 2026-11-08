# -*- coding: utf-8 -*-
"""
Audit-trail case extraction for PCBC (hier) (material for the Audit Traceability section of the paper)
================================================================
Goal: from a real test record of German Credit (the canonical credit scenario),
output the complete audit trail of the three-layer serial structure -
per layer: the feature pair (i,j), the value of the record in the 2D projection, the Delaunay simplex id,
     the indices of the three training vertices, the vertex residuals, the barycentric coordinates, the correction magnitude alpha*delta.

Protocol identical to the paper: StratifiedKFold(5, seed=42), z-scoring with training-fold statistics.
Model: exp4.PCBC_MS(mode='hier', K=3, seed=42) - row-identical to pcbc_msh3_full9.py.

Output: pcbc_audit_trail_case.csv (one row per layer) plus a detailed console printout
"""
import os
_CODE_DIR = os.path.dirname(os.path.abspath(__file__))

import time
import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import StratifiedKFold
import importlib.util

warnings.filterwarnings("ignore")

_spec = importlib.util.spec_from_file_location(
    "exp1", os.path.join(_CODE_DIR, "pcbc_temporal_direction1.py"))
exp1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exp1)
_spec4 = importlib.util.spec_from_file_location(
    "exp4", os.path.join(_CODE_DIR, "pcbc_multisubspace_experiment.py"))
exp4 = importlib.util.module_from_spec(_spec4)
_spec4.loader.exec_module(exp4)

DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_CSV = os.path.join(DATA_DIR, "pcbc_audit_trail_case.csv")

DS = "german"
ALPHA = exp4.ALPHA          # 0.6
TOP_K = 30                  # german n=1000 -> top_k=30 (same as pcbc_msh3_full9)


def layer_audit(sim, alpha, x_norm, sample_idx):
    """
    Resolve the audit record of layer sim for a single test record sample_idx.
    Returns a dict (when the projected record falls inside the convex hull) or a fallback dict (outside the hull).
    """
    xy = x_norm[sample_idx:sample_idx + 1, sim.pair]      # (1,2)
    tri_idx = sim.tri.find_simplex(xy)[0]
    if tri_idx == -1:
        return {
            "in_hull": False,
            "simplex_id": -1,
            "vertices": "",
            "vertex_residuals": "",
            "bary_coords": "",
            "correction": alpha * sim.fill,
        }
    vertices = sim.tri.simplices[tri_idx]
    lam = exp1.barycentric_coords(sim.tri, np.array([tri_idx]), xy)[0]
    r_verts = sim.resid_train[vertices]
    delta = float(np.sum(r_verts * lam))
    return {
        "in_hull": True,
        "simplex_id": int(tri_idx),
        "vertices": ",".join(map(str, vertices)),
        "vertex_residuals": ",".join(f"{v:.4f}" for v in r_verts),
        "bary_coords": ",".join(f"{v:.4f}" for v in lam),
        "correction": alpha * delta,
    }


def main():
    t0 = time.time()
    X, y, minority_label, min_ratio = exp1.load_dataset(DS)
    exp4.TOP_K = TOP_K

    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    folds = list(skf.split(X, y))
    tr, te = folds[0]                       # fold 0, consistent with the paper protocol
    mu, sg = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xtr, Xte = (X[tr] - mu) / sg, (X[te] - mu) / sg

    # train PCBC (hier, K=3) - same configuration as pcbc_msh3_full9
    m = exp4.PCBC_MS(seed=42, K=3, mode="hier")
    m.fit(Xtr, y[tr])

    # candidate selection: minority class, finally predicted positive (>=0.5), and with a positive total geometric correction
    # (a positive correction means the neighbour-residual evidence supports default, i.e. a complete reason chain: rejected because the neighbours defaulted)
    prob = m.predict_proba(Xte)[:, 1]
    cand = []
    for idx, i in enumerate(te):
        if y[i] != minority_label:
            continue
        if prob[idx] < 0.5:
            continue
        recs = [layer_audit(sim, ALPHA, Xte, idx) for sim in m.layers]
        tot = sum(r["correction"] for r in recs)
        n_hull = sum(1 for r in recs if r["in_hull"])
        cand.append((idx, i, prob[idx], n_hull, tot, recs))
    cand.sort(key=lambda c: c[4], reverse=True)   # the largest positive correction first
    if len(cand) == 0:
        # fallback: relax to minority class with p_final>=0.5 (no constraint on the sign of the correction)
        for idx, i in enumerate(te):
            if y[i] == minority_label and prob[idx] >= 0.5:
                recs = [layer_audit(sim, ALPHA, Xte, idx) for sim in m.layers]
                tot = sum(r["correction"] for r in recs)
                n_hull = sum(1 for r in recs if r["in_hull"])
                cand.append((idx, i, prob[idx], n_hull, tot, recs))
        cand.sort(key=lambda c: (c[3], c[4]), reverse=True)
    # deduplicate by simplex: keep only the case with the largest correction for each simplex (same evidence neighbours)
    seen_simplex = set()
    dedup = []
    for c in cand:
        key = tuple(r["simplex_id"] for r in c[5])
        if key in seen_simplex:
            continue
        seen_simplex.add(key)
        dedup.append(c)
    cand = dedup
    print(f"candidates: {len(cand)}  (dedup by simplex)", flush=True)

    print(f"fold0: train={len(tr)} test={len(te)}  minority(test)={sum(y[te]==minority_label)}", flush=True)
    print(f"pairs per layer: {[sim.pair for sim in m.layers]}", flush=True)
    print(f"LR coef top dims: {np.argsort(np.abs(m.lr.coef_[0]))[::-1][:6]}", flush=True)

    rows = []
    # output the full trail of the top-3 candidates
    for rank, (idx, sidx, p_final, n_hull, tot_corr, recs) in enumerate(cand[:3], start=1):
        print(f"\n===== Case {rank}: test sample #{sidx}  y={int(y[sidx])}  "
              f"p_final={p_final:.4f}  n_hull={n_hull}/3  Σcorr={tot_corr:+.4f} =====", flush=True)
        p_main = float(m.lr.predict_proba(Xte[idx:idx + 1])[:, 1][0])
        print(f"  p_main (LR) = {p_main:.4f}   logit_main = {np.log(p_main/(1-p_main)):.4f}", flush=True)
        logit = np.log(p_main / (1 - p_main))
        for k, sim in enumerate(m.layers, start=1):
            r = recs[k - 1]
            logit += r["correction"]
            pair = sim.pair
            x_raw_i, x_raw_j = X[sidx, pair[0]], X[sidx, pair[1]]
            print(f"  Layer {k}: pair={pair}  x[{pair[0]}]={Xte[idx, pair[0]]:+.4f} "
                  f"(raw={x_raw_i:g})  x[{pair[1]}]={Xte[idx, pair[1]]:+.4f} (raw={x_raw_j:g})", flush=True)
            if r["in_hull"]:
                print(f"    simplex #{r['simplex_id']}  vertices(train idx)={r['vertices']}", flush=True)
                print(f"    vertex residuals = [{r['vertex_residuals']}]", flush=True)
                print(f"    barycentric      = [{r['bary_coords']}]", flush=True)
                print(f"    correction α·δ   = {r['correction']:+.4f}   logit → {logit:.4f}", flush=True)
            else:
                print(f"    OUTSIDE hull → fallback fill={sim.fill:.4f}, correction={r['correction']:+.4f}", flush=True)
            rows.append({
                "rank": rank, "sample_id": int(sidx), "y_true": int(y[sidx]),
                "p_main": round(p_main, 4), "p_final": round(float(prob[idx]), 4),
                "layer": k, "pair_i": int(pair[0]), "pair_j": int(pair[1]),
                "x_i": round(float(Xte[idx, pair[0]]), 4),
                "x_j": round(float(Xte[idx, pair[1]]), 4),
                "x_raw_i": float(x_raw_i), "x_raw_j": float(x_raw_j),
                "in_hull": r["in_hull"], "simplex_id": r["simplex_id"],
                "vertices": r["vertices"], "vertex_residuals": r["vertex_residuals"],
                "bary_coords": r["bary_coords"], "correction": round(r["correction"], 4),
            })
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"\n[Saved] {OUT_CSV}  |  total {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()

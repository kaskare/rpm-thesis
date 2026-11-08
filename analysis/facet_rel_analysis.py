#!/usr/bin/env python3
"""Do the facet model's learned row relations separate the rules it was never told? Reads <run>/rel_vectors.npz
(analysis/facet_rel_vectors.py). A linear classifier learns the rule from the vector on half of the puzzles and is scored
on the rest, per attribute. Three references: always guessing the commonest rule; the same template untrained, at its
random initial weights; the raw input, the thermometer codes of the six panels of rows 1 and 2, from
experiments/baselines/data/iraven.npz. k-means asks the same without any labels (adjusted mutual information with the
rules). Single-object layouts and grids apart: on grids the attributes of non-uniform panels are noise (notes §22).
Base conda environment (scikit-learn), a few minutes of CPU:
    python3 analysis/facet_rel_analysis.py experiments/facet_all_w32-64-128_hidden
"""
import argparse, json, os

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.preprocessing import StandardScaler

GRIDS = ("distribute_four", "distribute_nine", "in_distribute_four_out_center_single")


def accuracy(x, y, puzzles):
    """A linear classifier's accuracy on the puzzles it was not fitted on, and the commonest rule's share there."""
    test = puzzles % 2 == 1
    sc = StandardScaler().fit(x[~test])
    clf = LogisticRegression(max_iter=3000, C=1.0).fit(sc.transform(x[~test]), y[~test])
    return clf.score(sc.transform(x[test]), y[test]), np.bincount(y[test]).max() / test.sum()


def clusters(x, y):
    z = PCA(20, random_state=0).fit_transform(StandardScaler().fit_transform(x))
    return adjusted_mutual_info_score(y, KMeans(len(np.unique(y)), n_init=10, random_state=0).fit_predict(z))


def raw_codes(puzzle, layout, component, attr, per_layout, data="experiments/baselines/data/iraven.npz"):
    """The mean thermometer code of the component's attribute in the six panels of rows 1 and 2, flattened."""
    z = np.load(data)
    T = int(json.loads(str(z["meta"]))["therm"])
    i = layout * 1000 + puzzle % per_layout                                             # unseen puzzles, per layout in order
    vals, comp = z["unseen_vals"][i][:, :6].astype(np.int64), z["unseen_comp"][i][:, :6].astype(np.int64)
    v = vals[np.arange(len(i)), :, :, attr + 1]                                         # [N, 6, M]
    sel = (vals[..., 0] >= 0) & (comp == component[:, None, None])
    code = (v[..., None] > np.arange(T)).astype(np.float32) * sel[..., None]            # [N, 6, M, T]
    return (code.sum(2) / np.maximum(sel.sum(2), 1)[..., None]).reshape(len(i), -1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--figure", default=None, help="write a PCA scatter, coloured by rule, here")
    args = ap.parse_args()
    z = np.load(os.path.join(args.run, "rel_vectors.npz"))
    rel, rule, attr, layout, puzzle, component = z["rel"], z["rule"], z["attr"], z["layout"], z["puzzle"], z["component"]
    untrained = np.load(os.path.join(args.run, "rel_vectors_untrained.npz"))["rel"]
    assert (np.load(os.path.join(args.run, "rel_vectors_untrained.npz"))["rule"] == rule).all()
    raw = raw_codes(puzzle, layout, component, attr, (puzzle.max() + 1) // (layout.max() + 1))
    names, attrs, layouts = list(z["rule_names"]), list(z["attrs"]), list(z["layouts"])
    grid = np.isin(layout, [layouts.index(g) for g in GRIDS if g in layouts])
    print(f"{len(rel)} attribute instances, rules {names}")
    print("                                     trained  untrained  raw input  commonest  AMI trained / untrained")
    for what, part in (("single-object layouts", ~grid), ("grids", grid)):
        for a, at in list(enumerate(attrs)) + [(None, "all")]:
            sel = part & (attr == a) if a is not None else part
            acc, chance = accuracy(rel[sel], rule[sel], puzzle[sel])
            acc_u = accuracy(untrained[sel], rule[sel], puzzle[sel])[0]
            acc_r = accuracy(raw[sel], rule[sel], puzzle[sel])[0]
            print(f"  {what:22s} {at:6s} n={sel.sum():5d} {100 * acc:7.1f} {100 * acc_u:10.1f} {100 * acc_r:10.1f} {100 * chance:10.1f}"
                  f"  {clusters(rel[sel], rule[sel]):.3f} / {clusters(untrained[sel], rule[sel]):.3f}", flush=True)
    if args.figure:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        sel = ~grid
        p = PCA(2, random_state=0).fit_transform(StandardScaler().fit_transform(rel[sel]))
        fig, ax = plt.subplots(figsize=(5.2, 4.2))
        for r, name in enumerate(names):
            m = rule[sel] == r
            ax.scatter(p[m, 0], p[m, 1], s=5, alpha=0.5, label=name.replace("_Three", ""))
        ax.set_xlabel("first principal component")
        ax.set_ylabel("second principal component")
        ax.legend(markerscale=3, frameon=False)
        fig.tight_layout()
        fig.savefig(args.figure, dpi=200)


if __name__ == "__main__":
    main()

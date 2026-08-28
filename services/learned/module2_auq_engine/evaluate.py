"""Standard OOD-detection metrics (Hendrycks & Gimpel 2017) + ECE (Guo 2017),
in NumPy so there is no sklearn dependency. score = epistemic u; positive = OOD."""
import numpy as np


def auroc(pos, neg):
    s = np.concatenate([pos,neg])
    order = np.argsort(s)
    ranks = np.empty_like(order,float)
    ranks[order] = np.arange(1,len(s)+1)
    r_pos = ranks[:len(pos)].sum()
    return (r_pos - len(pos)*(len(pos)+1)/2) / (len(pos)*len(neg))


def aupr(pos, neg):
    s = np.concatenate([pos,neg])
    lab = np.concatenate([np.ones_like(pos),np.zeros_like(neg)])
    idx = np.argsort(-s)
    lab = lab[idx]
    tp = np.cumsum(lab)
    fp = np.cumsum(1-lab)
    prec = tp/(tp+fp)
    rec = tp/lab.sum()
    o = np.argsort(rec)
    return float(np.trapezoid(prec[o], rec[o]))


def fpr95(pos, neg):
    thr = np.quantile(pos, 0.05)         # 95% of OOD above this
    return float((neg >= thr).mean())


def risk_coverage(u, correct, steps=21):
    """Selective prediction: sort by ascending u (keep most confident first); at each
    coverage report the error rate (risk) of the kept set. Returns (coverages, risks, aurc).

    Low AURC => u ranks errors to the top, i.e. u predicts error. This is the claim the
    module actually rests on: not that the probabilities are calibrated, but that the
    uncertainty is usable as a decision to abstain.
    """
    order = np.argsort(np.asarray(u))
    c = np.asarray(correct, float)[order]
    covs, risks = [], []
    for k in range(1, steps + 1):
        cov = k / steps
        keep = max(1, int(round(cov * len(c))))
        covs.append(cov)
        risks.append(1.0 - c[:keep].mean())
    covs, risks = np.array(covs), np.array(risks)
    return covs, risks, float(np.trapezoid(risks, covs))


def retained_composition(u, groups, steps=21):
    """Share of each group surviving in the kept set, by coverage. The companion to
    risk_coverage for a mixed stream: it says *what* gets rejected first without
    depending on how "error" was defined for unlabelled states."""
    order = np.argsort(np.asarray(u))
    g = np.asarray(groups)[order]
    names = sorted(set(g.tolist()))
    rows = []
    for k in range(1, steps + 1):
        cov = k / steps
        keep = max(1, int(round(cov * len(g))))
        kept = g[:keep]
        rows.append({"coverage": cov,
                     **{n: float((kept == n).mean()) for n in names}})
    return rows


def reliability_table(probs, labels, n_bins=10):
    """Per-bin calibration data (reliability diagram + ECE), emitted as data not a plot."""
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    conf = probs.max(1)
    pred = probs.argmax(1)
    acc = (pred == labels).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for i in range(n_bins):
        mrow = (conf > edges[i]) & (conf <= edges[i + 1])
        if mrow.sum():
            rows.append({"bin_lo": float(edges[i]), "bin_hi": float(edges[i + 1]),
                         "confidence": float(conf[mrow].mean()),
                         "accuracy": float(acc[mrow].mean()), "count": int(mrow.sum())})
    return rows


def ece(probs, labels, n_bins=10):
    conf = probs.max(1)
    pred = probs.argmax(1)
    acc = (pred==labels)
    bins = np.linspace(0,1,n_bins+1)
    e = 0.0
    for i in range(n_bins):
        m = (conf>bins[i]) & (conf<=bins[i+1])
        if m.sum()>0:
            e += m.mean()*abs(acc[m].mean()-conf[m].mean())
    return float(e)

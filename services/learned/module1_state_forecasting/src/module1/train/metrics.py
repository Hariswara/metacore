"""Evaluation metrics for the vulnerability forecaster (proposal Table 2).

All functions take ``pred`` and ``true`` shaped (B, N, H): forecast origin, node, horizon.
"""

from __future__ import annotations

import numpy as np


def error_by_horizon(pred: np.ndarray, true: np.ndarray) -> dict[str, list[float]]:
    err = pred - true
    return {
        "rmse": [float(np.sqrt(np.mean(err[..., h] ** 2))) for h in range(err.shape[-1])],
        "mae": [float(np.mean(np.abs(err[..., h]))) for h in range(err.shape[-1])],
    }


def event_detection(pred: np.ndarray, true: np.ndarray, tau: float) -> dict[str, list[float]]:
    """Per-horizon precision / recall / F1 for 'node is in a high-vulnerability state'."""
    out: dict[str, list[float]] = {"precision": [], "recall": [], "f1": [], "positives": []}
    for h in range(pred.shape[-1]):
        p, t = pred[..., h] > tau, true[..., h] > tau
        tp, fp, fn = int((p & t).sum()), int((p & ~t).sum()), int((~p & t).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        out["precision"].append(prec)
        out["recall"].append(rec)
        out["f1"].append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
        out["positives"].append(int(t.sum()))
    return out


def onset_lead_time(pred: np.ndarray, v: np.ndarray, origins: np.ndarray, tau: float) -> dict:
    """For every onset (node crosses tau upward at hour t0), the largest horizon h whose forecast,
    issued at t0 - h, already exceeded tau for t0. Zero means the onset was missed.

    ``v`` is the full (T, N) label series; ``origins`` maps each forecast row to its issue hour.
    """
    horizon = pred.shape[-1]
    row_of = {int(t): i for i, t in enumerate(origins)}
    leads = []
    for n in range(v.shape[1]):
        for t0 in range(1, v.shape[0]):
            if not (v[t0, n] > tau >= v[t0 - 1, n]):
                continue
            if any((t0 - h) not in row_of for h in range(1, horizon + 1)):
                continue  # onset not fully covered by test-split forecasts
            lead = 0
            for h in range(horizon, 0, -1):
                if pred[row_of[t0 - h], n, h - 1] > tau:
                    lead = h
                    break
            leads.append(lead)
    leads_a = np.array(leads)
    caught = leads_a[leads_a > 0]
    return {
        "onsets": int(len(leads_a)),
        "onset_recall": float((leads_a > 0).mean()) if len(leads_a) else 0.0,
        "mean_lead_h": float(caught.mean()) if len(caught) else 0.0,
    }


def spatial_hit_rate(pred: np.ndarray, true: np.ndarray, tau: float, k: int, h: int) -> dict:
    """At origins where some node is truly above tau at horizon h: is the truly most vulnerable
    node among the model's top-k?"""
    hits, total = 0, 0
    for i in range(pred.shape[0]):
        t = true[i, :, h]
        if t.max() <= tau:
            continue
        total += 1
        top_k = np.argsort(-pred[i, :, h])[:k]
        hits += int(int(np.argmax(t)) in top_k)
    return {"hit_rate": hits / total if total else 0.0, "cases": total}

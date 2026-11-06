#!/usr/bin/env python3
"""Continual learning utilities for neural CPR policies.

Provides Elastic Weight Consolidation (EWC) and experience replay
support for incremental training on new reservoir decks without
catastrophic forgetting of previously learned solver knowledge.

EWC prevents forgetting by penalising changes to weights that were
important for previous decks:

    L_total = L_new + λ_ewc * Σ_i F_i * (θ_i - θ*_i)²

where F_i is the diagonal Fisher information and θ*_i are the anchor
weights from the previous training round.

Experience replay mixes a small representative buffer of old training
data into each new training run, keeping old decision boundaries alive.

Usage from training scripts::

    from continual import (load_anchor, save_anchor, merge_fisher,
                           load_replay_buffer, save_replay_buffer,
                           select_exemplars, compute_fisher_diagonal,
                           ewc_penalty_grads)
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np


# ---------------------------------------------------------------------------
# Fisher information
# ---------------------------------------------------------------------------

def compute_fisher_diagonal(params: dict[str, np.ndarray],
                            grad_fn,
                            features: list[np.ndarray],
                            labels,
                            n_samples: int | None = None,
                            seed: int = 42) -> dict[str, np.ndarray]:
    """Diagonal Fisher information matrix estimated from squared gradients.

    Parameters
    ----------
    params : dict
        Parameter name -> array (from ``model._params()``).
    grad_fn : callable(feature, label) -> dict[str, ndarray]
        Returns per-parameter gradients for one sample.
    features : list of ndarray
        Training inputs.
    labels : list
        Training targets (int labels or float targets).
    n_samples : int or None
        Cap on samples used for the estimate.  ``None`` or 0 uses all.
    seed : int
        RNG seed when sub-sampling.
    """
    fisher = {k: np.zeros_like(v) for k, v in params.items()}
    n = len(features)
    if n_samples and n_samples < n:
        rng = np.random.default_rng(seed)
        indices = rng.choice(n, n_samples, replace=False)
    else:
        indices = range(n)
        n_samples = n

    for i in indices:
        grads = grad_fn(features[i], labels[i])
        for k, g in grads.items():
            fisher[k] += g ** 2 / n_samples

    return fisher


def merge_fisher(old_fisher: dict[str, np.ndarray],
                 new_fisher: dict[str, np.ndarray],
                 decay: float = 0.9) -> dict[str, np.ndarray]:
    """Exponential moving average of Fisher diagonals.

    ``F_merged = decay * F_old + (1 - decay) * F_new``

    Use ``decay`` close to 1.0 to preserve strong memory of earlier decks
    (default 0.9).
    """
    merged = {}
    all_keys = set(old_fisher) | set(new_fisher)
    for k in all_keys:
        if k in old_fisher and k in new_fisher:
            merged[k] = decay * old_fisher[k] + (1 - decay) * new_fisher[k]
        elif k in new_fisher:
            merged[k] = new_fisher[k].copy()
        else:
            merged[k] = old_fisher[k].copy()
    return merged


# ---------------------------------------------------------------------------
# EWC gradient penalty
# ---------------------------------------------------------------------------

def ewc_penalty_grads(params: dict[str, np.ndarray],
                      anchor: dict[str, np.ndarray],
                      fisher: dict[str, np.ndarray],
                      ewc_lambda: float) -> dict[str, np.ndarray]:
    r"""EWC gradient penalty: ``λ * F_i * (θ_i - θ*_i)``.

    Add the returned dict to your normal gradients before taking the
    optimiser step.
    """
    penalty = {}
    for k in params:
        if k in anchor and k in fisher:
            penalty[k] = ewc_lambda * fisher[k] * (params[k] - anchor[k])
        else:
            penalty[k] = np.zeros_like(params[k])
    return penalty


def ewc_loss(params: dict[str, np.ndarray],
             anchor: dict[str, np.ndarray],
             fisher: dict[str, np.ndarray],
             ewc_lambda: float) -> float:
    """Scalar EWC penalty (for logging, not backprop)."""
    total = 0.0
    for k in params:
        if k in anchor and k in fisher:
            total += float(np.sum(fisher[k] * (params[k] - anchor[k]) ** 2))
    return 0.5 * ewc_lambda * total


# ---------------------------------------------------------------------------
# Anchor (weights + Fisher) persistence
# ---------------------------------------------------------------------------

def save_anchor(params: dict[str, np.ndarray],
                fisher: dict[str, np.ndarray],
                path: str,
                decks: list[str] | None = None):
    """Save anchor weights and Fisher diagonal to an ``.npz`` file.

    Parameters
    ----------
    params : dict
        Model parameters to freeze as the anchor point.
    fisher : dict
        Diagonal Fisher information matrix.
    path : str
        Output ``.npz`` path.
    decks : list of str, optional
        Names of decks that contributed to this anchor (for provenance).
    """
    data: dict[str, np.ndarray] = {}
    for k, v in params.items():
        data[f"anchor_{k}"] = v.copy()
    for k, v in fisher.items():
        data[f"fisher_{k}"] = v.copy()
    if decks is not None:
        data["meta_decks"] = np.array(decks, dtype=object)
    np.savez(path, **data)


def load_anchor(path: str) -> tuple[dict[str, np.ndarray],
                                     dict[str, np.ndarray],
                                     list[str] | None]:
    """Load anchor weights and Fisher diagonal from an ``.npz`` file.

    Returns ``(anchor_params, fisher_diagonal, deck_names_or_None)``.
    """
    data = np.load(path, allow_pickle=True)
    anchor: dict[str, np.ndarray] = {}
    fisher: dict[str, np.ndarray] = {}
    decks: list[str] | None = None
    for k in data:
        if k.startswith("anchor_"):
            anchor[k[7:]] = data[k]
        elif k.startswith("fisher_"):
            fisher[k[7:]] = data[k]
    if "meta_decks" in data:
        decks = data["meta_decks"].tolist()
    return anchor, fisher, decks


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

def save_replay_buffer(features: list[np.ndarray],
                       labels,
                       path: str,
                       decks: list[str] | None = None):
    """Persist a replay buffer as ``.npz``.

    Parameters
    ----------
    features : list of ndarray
        Normalised input features.
    labels : list
        Target labels (int for classification, float for regression).
    path : str
        Output file.
    decks : list of str, optional
        Per-sample deck identifier (used for stratified selection).
    """
    data: dict[str, np.ndarray] = {
        "features": np.array(features, dtype=np.float32),
        "labels":   np.array(labels),
    }
    if decks is not None:
        data["decks"] = np.array(decks, dtype=object)
    np.savez(path, **data)


def load_replay_buffer(path: str) -> tuple[list[np.ndarray],
                                            list,
                                            list[str] | None]:
    """Load a replay buffer.

    Returns ``(features, labels, deck_names_or_None)``.
    """
    data = np.load(path, allow_pickle=True)
    features = [row.astype(np.float32) for row in data["features"]]
    labels = data["labels"].tolist()
    decks = data["decks"].tolist() if "decks" in data else None
    return features, labels, decks


def _pick_by_label(indices, labels, budget: int,
                   rng: np.random.Generator) -> list[int]:
    """Up to *budget* of *indices*: one sample of every distinct label
    first, so rare labels survive, then the rest uniformly at random,
    which keeps the label proportions of the data.
    """
    indices = np.asarray(indices)
    if len(indices) <= budget:
        return indices.tolist()

    by_label: dict = defaultdict(list)
    for i in indices:
        by_label[labels[i]].append(i)
    first = np.array([rng.choice(g) for g in by_label.values()])
    if len(first) >= budget:                # more labels than budget
        return rng.choice(first, budget, replace=False).tolist()

    rest = np.setdiff1d(indices, first)
    return first.tolist() + rng.choice(rest, budget - len(first),
                                       replace=False).tolist()


def select_exemplars(features: list[np.ndarray],
                     labels,
                     decks: list[str] | None = None,
                     budget_per_deck: int = 200,
                     seed: int = 42) -> tuple[list[np.ndarray], list, list[str] | None]:
    """Select a representative subset for the replay buffer.

    Stratifies by deck (if known): keeps up to *budget_per_deck*
    samples per deck (per buffer when *decks* is None), see
    :func:`_pick_by_label` for how labels are covered.
    """
    rng = np.random.default_rng(seed)
    n = len(features)

    if decks is None:
        idx = sorted(_pick_by_label(range(n), labels, budget_per_deck, rng))
        return ([features[i] for i in idx],
                [labels[i] for i in idx],
                None)

    by_deck: dict[str, list[int]] = defaultdict(list)
    for i, d in enumerate(decks):
        by_deck[d].append(i)

    selected: list[int] = []
    for indices in by_deck.values():
        selected.extend(_pick_by_label(indices, labels, budget_per_deck, rng))

    selected.sort()                          # deterministic order for diffs
    return ([features[i] for i in selected],
            [labels[i] for i in selected],
            [decks[i] for i in selected])

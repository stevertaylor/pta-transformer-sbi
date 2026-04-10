"""Tests for evaluation metrics."""

import numpy as np
import pytest
from src.metrics import hellinger_distance_grid


def test_hellinger_identical():
    """Identical grids should give distance 0."""
    p = np.random.default_rng(0).exponential(size=(20, 20))
    p /= p.sum()
    assert hellinger_distance_grid(p, p) == pytest.approx(0.0, abs=1e-12)


def test_hellinger_disjoint():
    """Non-overlapping distributions should give distance close to 1."""
    p = np.zeros((20, 20))
    q = np.zeros((20, 20))
    p[:10, :] = 1.0
    q[10:, :] = 1.0
    assert hellinger_distance_grid(p, q) == pytest.approx(1.0, abs=1e-12)


def test_hellinger_bounded():
    """Hellinger distance should be in [0, 1]."""
    rng = np.random.default_rng(1)
    p = rng.exponential(size=(30, 30))
    q = rng.exponential(size=(30, 30))
    h = hellinger_distance_grid(p, q)
    assert 0.0 <= h <= 1.0


def test_hellinger_symmetric():
    """H(P, Q) == H(Q, P)."""
    rng = np.random.default_rng(2)
    p = rng.exponential(size=(15, 15))
    q = rng.exponential(size=(15, 15))
    assert hellinger_distance_grid(p, q) == pytest.approx(
        hellinger_distance_grid(q, p), abs=1e-12
    )


def test_hellinger_scale_invariant():
    """Scaling both grids should not change the distance."""
    rng = np.random.default_rng(3)
    p = rng.exponential(size=(20, 20))
    q = rng.exponential(size=(20, 20))
    h1 = hellinger_distance_grid(p, q)
    h2 = hellinger_distance_grid(100 * p, 0.01 * q)
    assert h1 == pytest.approx(h2, abs=1e-12)


def test_hellinger_known_value():
    """Check against a hand-computed Hellinger for two simple PMFs."""
    # Two-bin PMF: P = [0.8, 0.2], Q = [0.2, 0.8]
    p = np.array([[0.8, 0.2]])
    q = np.array([[0.2, 0.8]])
    sp = np.sqrt(p / p.sum())
    sq = np.sqrt(q / q.sum())
    expected = float(np.sqrt(0.5 * np.sum((sp - sq) ** 2)))
    assert hellinger_distance_grid(p, q) == pytest.approx(expected, abs=1e-12)


def test_hellinger_zero_grid():
    """All-zero grid should return 1.0 (maximally different)."""
    p = np.zeros((10, 10))
    q = np.ones((10, 10))
    assert hellinger_distance_grid(p, q) == 1.0
    assert hellinger_distance_grid(q, p) == 1.0
    assert hellinger_distance_grid(p, p) == 1.0

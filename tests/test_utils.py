"""Tests for utility functions."""

from __future__ import annotations

import random

import numpy as np

from toxic_comment_classifier.utils.seed import set_seed


class TestSetSeed:
    def test_python_random_is_deterministic(self) -> None:
        """Same seed should produce same random numbers."""
        set_seed(42)
        a = random.random()
        set_seed(42)
        b = random.random()
        assert a == b

    def test_numpy_random_is_deterministic(self) -> None:
        """Same seed should produce same numpy random numbers."""
        set_seed(42)
        a = np.random.rand()
        set_seed(42)
        b = np.random.rand()
        assert a == b

    def test_different_seeds_differ(self) -> None:
        """Different seeds should produce different random numbers."""
        set_seed(1)
        a = np.random.rand()
        set_seed(2)
        b = np.random.rand()
        assert a != b

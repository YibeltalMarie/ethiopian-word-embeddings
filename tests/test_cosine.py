import math
import pytest
from src.evaluation.cosine import cosine_similarity

TOL = 1e-5


def test_concept_check_values():
    assert abs(cosine_similarity([1, 0], [2, 0]) - 1.0) < TOL
    assert abs(cosine_similarity([1, 0], [0, 5]) - 0.0) < TOL
    assert abs(cosine_similarity([1, 0], [-3, 0]) - (-1.0)) < TOL


def test_known_example():
    # 32 / (sqrt(14) * sqrt(77))
    assert abs(cosine_similarity([1, 2, 3], [4, 5, 6]) - 0.974632) < TOL


def test_scale_invariance():
    a, b = [1.0, 2.0, 3.0], [-1.0, 0.5, 2.0]
    base = cosine_similarity(a, b)
    assert abs(cosine_similarity([10 * x for x in a], b) - base) < TOL
    assert abs(cosine_similarity(a, [0.01 * x for x in b]) - base) < TOL


def test_symmetry():
    a, b = [1.0, 2.0, 3.0], [-1.0, 0.5, 2.0]
    assert abs(cosine_similarity(a, b) - cosine_similarity(b, a)) < TOL


def test_vector_with_itself_is_one():
    assert abs(cosine_similarity([0.3, -0.7, 2.0], [0.3, -0.7, 2.0]) - 1.0) < TOL


def test_zero_norm_vectors_return_zero_without_crashing():
    assert cosine_similarity([0, 0], [1, 1]) == 0.0
    assert cosine_similarity([1, 1], [0, 0]) == 0.0
    assert cosine_similarity([0, 0], [0, 0]) == 0.0


def test_result_always_within_range():
    for a, b in (([0.1, 0.2, 0.3], [0.1, 0.2, 0.3]),
                 ([1e-9, 2e-9], [3e-9, 6e-9]),
                 ([1e8, 1e8], [-1e8, -1e8])):
        assert -1.0 <= cosine_similarity(a, b) <= 1.0


def test_bad_inputs_fail_clearly():
    with pytest.raises(ValueError):
        cosine_similarity([1, 2], [1, 2, 3])
    with pytest.raises(ValueError):
        cosine_similarity([], [])

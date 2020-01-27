from random import shuffle

import numpy as np
import pytest

@pytest.fixture
def fake_vectors():
    return [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]

@pytest.fixture
def fake_prediction(fake_vectors):
    pred = shuffle(fake_vectors)
    return pred

def test_get_rank_score(fake_vectors, fake_prediction):
    print(fake_prediction)
    assert len(fake_prediction) == 3
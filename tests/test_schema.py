
import pytest
import sys
import os
from pydantic import ValidationError

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from schema import IrisInput


def test_valid_input():
    x = IrisInput(
        sepal_length=5.1,
        sepal_width=3.5,
        petal_length=1.4,
        petal_width=0.2
    )
    arr = x.to_array()
    assert len(arr[0]) == 4


def test_negative_rejected():
    with pytest.raises(ValidationError):
        IrisInput(
            sepal_length=-1,
            sepal_width=3.5,
            petal_length=1.4,
            petal_width=0.2
        )

# pylint: disable=missing-function-docstring,redefined-outer-name
import pytest
import pandas as pd


@pytest.fixture
def closing_series():
    index = pd.date_range(start="2021-01-01", periods=10, freq="D")
    close = pd.Series([100, 102, 101, 103, 104, 103, 105, 106, 104, 107], index=index)
    return close

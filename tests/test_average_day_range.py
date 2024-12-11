# pylint: disable=missing-function-docstring,redefined-outer-name
import pandas as pd
import pytest
from comprehensive_ta.main import average_day_range


@pytest.fixture
def high_low_series():
    index = pd.date_range(start="2021-01-01", periods=30, freq="D")
    high = pd.Series([100 + i for i in range(30)], index=index)
    low = pd.Series([90 + i for i in range(30)], index=index)
    return high, low


def test_get_average_day_range(high_low_series):
    """Test that the average day range is calculated correctly"""
    # Arrange
    high, low = high_low_series
    expected_value = sum(high[:14] - low[:14]) / 14

    # Act
    result_series = average_day_range(high, low, period=14)

    # Assert
    assert isinstance(result_series, pd.Series), "Output should be a pandas Series"
    assert len(result_series) == len(
        high
    ), "Result series should have the same length as input series"
    assert (
        result_series.iloc[13] == expected_value
    ), "Average calculation at the first full period should be correct"


def test_get_average_day_range_invalid_period(high_low_series):
    """Test that an invalid period raises a ValueError"""
    # Arrange
    high, low = high_low_series

    # Act & Assert
    with pytest.raises(ValueError):
        average_day_range(high, low, period=0)

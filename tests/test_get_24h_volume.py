# pylint: disable=missing-function-docstring,redefined-outer-name
import pandas as pd
import pytest
from comprehensive_ta.main import get_24h_volume


@pytest.fixture
def volume_series():
    # Create a DateTime indexed pandas Series for volumes
    index = pd.date_range(start="2021-01-01", periods=24, freq="H")  # 24-hour data
    data = [1000 + i * 10 for i in range(24)]  # Increasing volume data
    return pd.Series(data=data, index=index)


# We are not testing this properlyhow do we know that given a series it is appropriately geting the 24h volume?
def test_get_24h_volume(volume_series):
    """Test that the 24h volume is calculated correctly"""
    # Arrange
    expected_sum = sum([1000 + i * 10 for i in range(24)])

    # Act
    result_series = get_24h_volume(volume_series)

    # Assert
    assert isinstance(result_series, pd.Series), "Output should be a pandas Series"
    assert result_series.iloc[-1] == expected_sum, "The last entry should be the sum of all volumes"
    assert len(result_series) == len(
        volume_series
    ), "Result series should have the same length as input series"


def test_get_24h_volume_empty_series():
    """Test that an empty series returns an empty series"""

    # Arrange
    empty_series = pd.Series(data=[], dtype=float)  # Empty series
    empty_series.index = pd.to_datetime(empty_series.index)  # Ensure datetime index even if empty

    # Act
    result_series = get_24h_volume(empty_series)

    # Assert
    assert result_series.empty, "Result should be an empty Series for empty input"

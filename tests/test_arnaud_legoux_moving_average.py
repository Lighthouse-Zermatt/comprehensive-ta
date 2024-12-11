# pylint: disable=missing-function-docstring,redefined-outer-name
import pandas as pd
import numpy as np
import pytest
from comprehensive_ta.main import arnaud_legoux_moving_average


@pytest.fixture
def sample_price_series():
    """Create a sample price series for testing"""
    index = pd.date_range(start="2021-01-01", periods=50, freq="D")
    prices = [
        100 + i + np.sin(i / 5) * 10 for i in range(50)
    ]  # Creating a sinusoidal price movement
    return pd.Series(prices, index=index)


def test_arnaud_legoux_moving_average(sample_price_series):
    """Test ALMA calculation"""
    # Act
    result = arnaud_legoux_moving_average(sample_price_series)

    # Assert
    assert isinstance(result, pd.Series), "Output should be a pandas Series"
    assert len(result) == len(sample_price_series), "Result length should match input length"
    assert not result.isnull().all(), "Result should not be all NaN"
    assert result.dtype == float, "Result should be floating point numbers"


def test_arnaud_legoux_moving_average_empty_series():
    """Test ALMA with empty series"""
    # Arrange
    empty_series = pd.Series([], dtype=float)

    # Act
    result = arnaud_legoux_moving_average(empty_series)

    # Assert
    assert isinstance(result, pd.Series), "Output should be a pandas Series"
    assert result.empty, "Result should be empty for empty input"


def test_arnaud_legoux_moving_average_single_value():
    """Test ALMA with single value"""
    # Arrange
    single_value = pd.Series([100.0], index=[pd.Timestamp("2021-01-01")])

    # Act
    result = arnaud_legoux_moving_average(single_value)

    # Assert
    assert isinstance(result, pd.Series), "Output should be a pandas Series"
    assert len(result) == 1, "Result should have length 1"
    assert pd.isna(result.iloc[0]), "Result should be NaN for single value"

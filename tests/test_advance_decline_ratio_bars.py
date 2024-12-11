# pylint: disable=missing-function-docstring,redefined-outer-name
import pytest
import pandas as pd
from comprehensive_ta.main import advance_decline_ratio_bars


def test_advance_decline_ratio_bars(closing_series):
    """Test advance decline ratio bars calculation"""
    # Arrange
    window = 5

    # Act
    result = advance_decline_ratio_bars(closing_series, window)

    # Assert
    assert isinstance(result, pd.Series), "Output should be a pandas Series"
    assert len(result) == len(closing_series), "Result length should match input length"
    assert not result.isnull().all(), "Result should not be all NaN"
    assert result.iloc[window - 1 :].notna().all(), "Values after window period should not be NaN"


def test_advance_decline_ratio_bars_empty_series():
    """Test advance decline ratio bars with empty series"""
    # Arrange
    empty_series = pd.Series([], dtype=float)
    window = 5

    # Act
    result = advance_decline_ratio_bars(empty_series, window)

    # Assert
    assert isinstance(result, pd.Series), "Output should be a pandas Series"
    assert result.empty, "Result should be empty for empty input"


def test_advance_decline_ratio_bars_constant_series():
    """Test advance decline ratio bars with constant values"""
    # Arrange
    constant_series = pd.Series(
        [100] * 10, index=pd.date_range(start="2021-01-01", periods=10, freq="D")
    )
    window = 5

    # Act
    result = advance_decline_ratio_bars(constant_series, window)

    # Assert
    assert isinstance(result, pd.Series), "Output should be a pandas Series"
    assert (result.fillna(0) == 0).all(), "Constant series should yield 0 ratio"


def test_advance_decline_ratio_bars_window_validation():
    """Test advance decline ratio bars with invalid window size"""
    # Arrange
    series = pd.Series(
        [100, 101, 102, 103, 104], index=pd.date_range(start="2021-01-01", periods=5, freq="D")
    )

    # Act & Assert
    with pytest.raises(ValueError):
        advance_decline_ratio_bars(series, window=0)

    with pytest.raises(ValueError):
        advance_decline_ratio_bars(series, window=-1)


def test_advance_decline_ratio_bars_increasing_series():
    """Test advance decline ratio bars with consistently increasing values"""
    # Arrange
    increasing_series = pd.Series(
        [100, 101, 102, 103, 104, 105], index=pd.date_range(start="2021-01-01", periods=6, freq="D")
    )
    window = 3

    # Act
    result = advance_decline_ratio_bars(increasing_series, window)

    # Assert
    assert isinstance(result, pd.Series), "Output should be a pandas Series"
    assert (
        result.iloc[window:] > 0
    ).all(), "Increasing series should have positive ratios after window period"

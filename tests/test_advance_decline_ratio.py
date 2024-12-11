# pylint: disable=missing-function-docstring,redefined-outer-name
import pandas as pd
from comprehensive_ta.main import advance_decline_ratio


def test_advance_decline_ratio(closing_series):
    # Arrange
    expected_ratio = [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]

    # Act
    result_series = advance_decline_ratio(closing_series)

    # Assert
    assert isinstance(result_series, pd.Series), "Output should be a pandas Series"
    assert (
        result_series.tolist() == expected_ratio
    ), "Advance/Decline Ratio calculation is incorrect"


def test_advance_decline_ratio_zero_declines():
    # Arrange
    close = pd.Series(
        [100, 102, 104, 106, 108], index=pd.date_range("2021-01-01", periods=5, freq="D")
    )
    expected_ratio = [0.0, 1.0, 1.0, 1.0, 1.0]

    # Act
    result_series = advance_decline_ratio(close)

    # Assert
    assert result_series.tolist() == expected_ratio, "Ratio with zero declines should be 1.0"


def test_advance_decline_ratio_empty_series():
    """Test that an empty series returns an empty series"""

    # Arrange
    empty_series = pd.Series([], dtype=float)

    # Act
    result_series = advance_decline_ratio(empty_series)

    # Assert
    assert result_series.empty, "Result should be an empty Series for empty input"


def test_advance_decline_ratio_constant_series():
    # Arrange
    constant_series = pd.Series(
        [100] * 5, index=pd.date_range(start="2021-01-01", periods=5, freq="D")
    )
    expected_ratio = [None, 0.0, 0.0, 0.0, 0.0]

    # Act
    result_series = advance_decline_ratio(constant_series)

    # Assert
    assert result_series.fillna(0).tolist() == [
        0 if v is None else v for v in expected_ratio
    ], "Constant series should yield a 0.0 ratio"

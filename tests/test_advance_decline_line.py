# pylint: disable=missing-function-docstring,redefined-outer-name
import pandas as pd
from comprehensive_ta.main import advance_decline_line


def test_advance_decline_line(closing_series):

    # Arrange
    expected_advances = [0, 1, 0, 1, 1, 0, 1, 1, 0, 1]
    expected_declines = [0, 0, 1, 0, 0, 1, 0, 0, 1, 0]

    # Act
    result_df = advance_decline_line(closing_series)

    # Assert
    assert isinstance(result_df, pd.DataFrame), "Output should be a pandas DataFrame"

    assert "advances" in result_df.columns, "DataFrame should have an 'advances' column"
    assert "declines" in result_df.columns, "DataFrame should have a 'declines' column"

    assert result_df.iloc[0]["advances"] == 0, "First row 'advances' should be 0"
    assert result_df.iloc[0]["declines"] == 0, "First row 'declines' should be 0"

    assert result_df["advances"].tolist() == expected_advances, "Advances calculation is incorrect"
    assert result_df["declines"].tolist() == expected_declines, "Declines calculation is incorrect"


def test_advance_decline_line_empty_series():
    """Test that an empty series has no advances or declines"""

    # Arrange
    empty_series = pd.Series(data=[], dtype=float)
    empty_series.index = pd.to_datetime(empty_series.index)

    # Act
    result_df = advance_decline_line(empty_series)

    # Assert
    assert result_df.empty, "Result should be an empty DataFrame for empty input"


def test_advance_decline_line_constant_series():
    """Test that a constant series has no advances or declines"""

    # Arrange
    constant_series = pd.Series(
        [100] * 5, index=pd.date_range(start="2021-01-01", periods=5, freq="D")
    )

    # Act
    result_df = advance_decline_line(constant_series)

    # Assert
    assert (result_df["advances"].fillna(0) == 0).all(), "Constant series should have no advances"
    assert (result_df["declines"].fillna(0) == 0).all(), "Constant series should have no declines"

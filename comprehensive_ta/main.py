import pandas as pd
import pandas_ta
import numpy as np
import talipp
from talipp.indicators import PivotsHL
from talipp.indicators.PivotsHL import PivotType

# TODO: ensure data is normalized to a frequency each one expects

PPL_HIGH_PERIOD = 5
PPL_LOW_PERIOD = 5


def get_24h_volume(volume: pd.Series):
    """
    source: https://www.tradingview.com/support/solutions/43000718736-indicators-volume-24h-in-usd-and-volume-24h-change/#:~:text=The%20%E2%80%9CVolume%2024h%20in%20USD,may%20indirectly%20indicate%20market%20moves.
    Calculate the sum of volumes over the last 24 hours for a financial instrument.
    Args:
        volume (pd.Series): A pandas Series with trading volumes indexed by DateTime.
    Returns:
        pd.Series: A Series with the sum of volumes calculated over 24-hour rolling windows.
    """
    # Ensure the index is datetime for time-based rolling operations
    if not pd.api.types.is_datetime64_any_dtype(volume.index):
        volume.index = pd.to_datetime(volume.index)
    return volume.rolling(window=24, min_periods=1).sum()


# THIS IS A MARKET WIDE INDICATOR


def advance_decline_line(close):
    """
    Calculate the advancing and declining movements based on closing prices from one day to the next.

    Shows how many stocks are involved in a rising or falling market.

    source: https://www.tradingview.com/support/solutions/43000589092-advance-decline-line/

    Args:
        closings (pd.Series): A pandas Series of closing prices indexed by date.
    Returns:
        pd.DataFrame: DataFrame with daily counts of advances and declines.
    """
    # Calculate daily changes
    changes = close.diff()

    # Determine advances and declines
    advances = (changes > 0).astype(int)
    declines = (changes < 0).astype(int)

    return pd.DataFrame({"advances": advances, "declines": declines}, index=close.index)


def advance_decline_ratio(close):
    """
    Calculate the Advance/Decline Ratio based on closing prices from one day to the next.

    Args:
        close (pd.Series): A pandas Series of closing prices indexed by date.
    Returns:
        pd.Series: A Series with the Advance/Decline Ratio calculated over the given period.
    """
    # Return empty Series if the input is empty
    if close.empty:
        return pd.Series(dtype=float)

    # Calculate daily changes
    changes = close.diff()

    # Record daily advances and declines
    advances = (changes > 0).astype(int)
    declines = (changes < 0).astype(int)

    # Calculate the Advance/Decline Ratio
    adv_dec_ratio = advances / declines.replace(
        0, 1
    )  # Replace 0 in declines to avoid division by zero

    return adv_dec_ratio


def advance_decline_ratio_bars(close: pd.Series, window: int = 14) -> pd.Series:
    """
    Calculate the Advance/Decline Ratio Bars based on closing prices in a given period,
    smoothed over a specified window period.

    Args:
        close (pd.Series): A pandas Series of closing prices indexed by date.
        window (int): The window size for calculating the rolling advance/decline ratio.

    Returns:
        pd.Series: A Series with the Advance/Decline Ratio Bars calculated over the given period.
    """
    # Calculate daily changes
    changes = close.diff()

    # Record daily advances and declines as booleans
    advances = (changes > 0).astype(int)
    declines = (changes < 0).astype(int)

    # Calculate rolling sums for advances and declines
    rolling_advances = advances.rolling(window=window, min_periods=1).sum()
    rolling_declines = declines.rolling(window=window, min_periods=1).sum()

    # Compute the advance/decline ratio
    adv_dec_ratio = rolling_advances / rolling_declines.replace(0, 1)  # Avoid division by zero

    return adv_dec_ratio


def average_day_range(high, low, period=14):
    """
    Calculate the average range of trading days over a specified period.
    Args:
        high (pd.Series): A pandas Series with 'high' prices indexed by DateTime.
        low (pd.Series): A pandas Series with 'low' prices indexed by DateTime.
        period (int): Number of days to consider for the average calculation.
    Returns:
        pd.Series: A Series with the average day range calculated over the specified period.
    """

    if period <= 0:
        raise ValueError("Period must be a positive integer.")

    # Calculate the daily range
    daily_range = high - low

    # Calculate the rolling average of the daily range
    result = daily_range.rolling(window=period).mean()

    # Set the name for the resulting Series
    result.name = "average_day_range"

    return result


def arnaud_legoux_moving_average(close: pd.Series):
    """
    Wrapper function to call the ALMA (Arnaud Legoux Moving Average) from pandas_ta
    with dynamic length based on dataset size.

    Args:
        close (pd.Series): A pandas Series of closing prices indexed by date.
        length (int): The window size for calculating the ALMA. If None, dynamically set based on data size.
        sigma (float): Standard deviation to control smoothness.
        distribution_offset (float): The offset to control the center of mass.
        offset (int): The offset for shifting the result.
        **kwargs: Additional keyword arguments for fillna, fill_method, etc.

    Returns:
        pd.Series: A pandas Series with the calculated ALMA values.
    """
    # Determine the size of the data
    data_size = len(close)
    length = max(2, int(0.1 * data_size))  # Default to 10% of the data size, with a minimum of 2
    # Return the result
    return pandas_ta.alma(
        close,
        length=length,
    )


# TODO: Tests done up till here


def balance_of_power(
    open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series
):
    """
    source: https://www.tradingview.com/support/solutions/43000589100-balance-of-power-bop/#:~:text=Takeaways,equal%20in%20the%20current%20market.
    Wrapper function for Balance of Power that accepts 'open' instead of 'open_'.
    Args:
    open (pd.Series): Series of opening prices.
    high (pd.Series): Series of high prices.
    low (pd.Series): Series of low prices.
    close (pd.Series): Series of closing prices.
    volume (pd.Series): Series of trading volumes.
    Returns:
    pd.Series: Series containing the Balance of Power values.
    """
    # Call the original bop function from pandas_ta library
    return pandas_ta.bop(open_=open_, high=high, low=low, close=close, volume=volume)


def bbtrend(
    close: pd.Series, short_window: int = 20, long_window: int = 50, num_std_dev: int = 2
) -> pd.Series:
    """
    source: https://www.tradingview.com/support/solutions/43000726749-bbtrend/
    Calculate Bollinger Bands Trend (BBTrend) based on the specific formula provided.
    Args:
        close (pd.Series): A pandas Series with closing prices.
        short_window (int): Moving average window size for the short period.
        long_window (int): Moving average window size for the long period.
        num_std_dev (int): Number of standard deviations for the bands.
    Returns:
        pd.Series: A Series with 'bbtrend' indicating trend changes.
    """
    # Calculate the short-term and long-term rolling mean and standard deviation
    short_mean = close.rolling(window=short_window).mean()
    long_mean = close.rolling(window=long_window).mean()

    short_std = close.rolling(window=short_window).std()
    long_std = close.rolling(window=long_window).std()

    # Calculate short-term and long-term upper and lower Bollinger Bands
    short_upper = short_mean + (short_std * num_std_dev)
    short_lower = short_mean - (short_std * num_std_dev)
    long_upper = long_mean + (long_std * num_std_dev)
    long_lower = long_mean - (long_std * num_std_dev)

    # Calculate BBTrend
    result = (
        (np.abs(short_lower - long_lower) - np.abs(short_upper - long_upper)) / short_mean * 100
    )
    result.name = "BBTrend"

    return result


def bollinger_b(close: pd.Series, window: int = 20, num_std_dev: int = 2) -> pd.Series:
    """
    Calculate the Bollinger Bands %B indicator.
    Args:
        close (pd.Series): Series of closing prices.
        window (int, optional): The window size for the moving average and standard deviation. Defaults to 20.
        num_std_dev (int, optional): The number of standard deviations for the Bollinger Bands. Defaults to 2.
    Returns:
        pd.Series: A Series representing the Bollinger Bands %B.
    """
    # Calculate the Simple Moving Average (SMA)
    sma = close.rolling(window=window).mean()

    # Calculate the Standard Deviation (SD)
    sd = close.rolling(window=window).std()

    # Calculate the Upper and Lower Bollinger Bands
    upper_band = sma + (sd * num_std_dev)
    lower_band = sma - (sd * num_std_dev)

    # Calculate the Bollinger Bands %B
    bollinger_b = (close - lower_band) / (upper_band - lower_band)

    return pd.Series(bollinger_b, index=close.index, name="Bollinger_Bands_B")


def bollinger_wband(close: pd.Series, window: int = 20, num_std_dev: int = 2) -> pd.Series:
    """
    Calculate the Bollinger Bands Width (BBW).
    Args:
        close (pd.Series): Series of closing prices.
        window (int, optional): The window size for the moving average and standard deviation. Defaults to 20.
        num_std_dev (int, optional): The number of standard deviations for the Bollinger Bands. Defaults to 2.
    Returns:
        pd.Series: A Series representing the Bollinger Bands Width (BBW).
    """
    # Calculate the Simple Moving Average (SMA)
    sma = close.rolling(window=window).mean()

    # Calculate the Standard Deviation (SD)
    sd = close.rolling(window=window).std()

    # Calculate the Upper and Lower Bollinger Bands
    upper_band = sma + (sd * num_std_dev)
    lower_band = sma - (sd * num_std_dev)

    # Calculate the Bollinger Bands Width
    bb_width = ((upper_band - lower_band) / sma) * 100

    # Set the name for the resulting Series
    bb_width.name = "bollinger_wband"

    return bb_width


def bull_bear_power(
    close: pd.Series, high: pd.Series, low: pd.Series, period: int = 14
) -> pd.DataFrame:
    """
    Calculate Bull and Bear Power.
    Args:
        close (pd.Series): Series of closing prices.
        high (pd.Series): Series of high prices.
        low (pd.Series): Series of low prices.
        period (int): Period for calculating EMA.
    Returns:
        pd.DataFrame: DataFrame with 'bull_power' and 'bear_power'.
    """
    ema = close.ewm(span=period, adjust=False).mean()
    bull_power = high - ema
    bear_power = low - ema

    result = pd.DataFrame({"bull_power": bull_power, "bear_power": bear_power}, index=close.index)

    return result


def chaikin_oscillator(
    close: pd.Series, high: pd.Series, low: pd.Series, volume: pd.Series
) -> pd.Series:
    """
    Calculate the Chaikin Oscillator using talipp library.
    The Chaikin Oscillator is calculated as the difference between the 3-day EMA and the 10-day EMA
    of the Accumulation/Distribution Line (ADL).
    Args:
        close (pd.Series): Series of closing prices.
        high (pd.Series): Series of high prices.
        low (pd.Series): Series of low prices.
        volume (pd.Series): Series of volume data.
    Returns:
        pd.Series: A Series representing the Chaikin Oscillator.
    """
    # Initialize the Chaikin Oscillator with the desired parameters
    chaikin_osc = talipp.indicators.ChaikinOsc(fast_period=3, slow_period=10)

    # Feed the talipp.ohlcv.OHLCV data to the Chaikin Oscillator
    for ohlcv in zip(close.index, high, low, close, volume):
        chaikin_osc.add_input_value(
            talipp.ohlcv.OHLCV(ohlcv[0], ohlcv[1], ohlcv[2], ohlcv[3], ohlcv[4])
        )

    # Extract the Chaikin Oscillator values
    chaikin_osc_values = list(chaikin_osc)

    # Create a pandas Series for the Chaikin Oscillator values
    chaikin_osc_series = pd.Series(
        chaikin_osc_values,
        index=close.index[-len(chaikin_osc_values) :],
        name="ChaikinOscillator",
    )

    return chaikin_osc_series


def _chande_kroll_stop(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    atr_period: int = 10,
    atr_mult: float = 1.5,
    period: int = 20,
) -> pd.DataFrame:
    """
    Calculate the Chande Kroll Stop.
    The Chande Kroll Stop is a volatility-based indicator that consists of two lines: a stop line above the price
    and a stop line below the price.
    Args:
        close (pd.Series): Series of closing prices.
        high (pd.Series): Series of high prices.
        low (pd.Series): Series of low prices.
        atr_period (int): Period for the Average True Range (ATR).
        atr_mult (float): Multiplier for the ATR.
        period (int): Period for the Chande Kroll Stop.
    Returns:
        pd.DataFrame: A DataFrame with columns 'ChandeKrollStop_Low' and 'ChandeKrollStop_High'.
    """
    # Initialize the Chande Kroll Stop with the required parameters
    chande_kroll = talipp.indicators.ChandeKrollStop(atr_period, atr_mult, period)

    # Check if the indicator was created successfully
    if chande_kroll is None:
        print("Failed to create Chande Kroll Stop indicator. Check the parameters and library.")
        return None

    # Feed the talipp.ohlcv.OHLCV data to the Chande Kroll Stop
    for h, l, c in zip(high, low, close):
        chande_kroll.add_input_value(talipp.ohlcv.OHLCV(0, h, l, c, 0))

    # Check if values are computed
    if not chande_kroll:
        print("No values computed for Chande Kroll Stop. Check input data and indicator logic.")
        return None

    # Extract the Chande Kroll Stop values
    chande_kroll_values = [(val[0], val[1]) for val in chande_kroll]

    if not chande_kroll_values:
        print("Chande Kroll Stop values are empty.")
        return None

    # Create a pandas DataFrame for the Chande Kroll Stop values
    result = pd.DataFrame(
        {
            "ChandeKrollStop_Low": [val[0] for val in chande_kroll_values],
            "ChandeKrollStop_High": [val[1] for val in chande_kroll_values],
        },
        index=close.index[-len(chande_kroll_values) :],
    )

    return result


def calculate_true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Calculate the true range of a trading day."""
    previous_close = close.shift()
    result = pd.DataFrame(
        {
            "high_low": high - low,
            "high_prev_close": (high - previous_close).abs(),
            "low_prev_close": (low - previous_close).abs(),
        }
    ).max(axis=1)
    return result


def chande_kroll_stop(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    atr_period: int = 10,
    atr_multiplier: float = 1.5,
    stop_period: int = 20,
) -> pd.DataFrame:
    """Calculate the Chande Kroll Stop indicator."""
    atr = average_true_range(high=high, low=low, close=close, period=atr_period)
    highest_high = high.rolling(window=stop_period).max()
    lowest_low = low.rolling(window=stop_period).min()

    chande_kroll_stop_high = highest_high + (atr * atr_multiplier)
    chande_kroll_stop_low = lowest_low - (atr * atr_multiplier)

    return pd.DataFrame(
        {
            "ChandeKrollStop_High": chande_kroll_stop_high,
            "ChandeKrollStop_Low": chande_kroll_stop_low,
        },
        index=high.index,
    )


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    return pd.concat(
        [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1
    ).max(axis=1)


def average_true_range(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """
    source: https://www.investopedia.com/terms/a/atr.asp
    """
    tr = true_range(high, low, close)
    return tr.rolling(window=period, min_periods=1).mean()


def choppiness_index(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    tr = true_range(high, low, close)
    atr = tr.rolling(window=period, min_periods=1).sum()
    high_max = high.rolling(window=period, min_periods=1).max()
    low_min = low.rolling(window=period, min_periods=1).min()
    ci = 100 * np.log10(atr / (high_max - low_min)) / np.log10(period)
    return ci


def chop_zone(
    open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.DataFrame:
    """
    Calculate the Chop Zone indicator.
    Args:
        open (pd.Series): Series of opening prices.
        high (pd.Series): Series of high prices.
        low (pd.Series): Series of low prices.
        close (pd.Series): Series of closing prices.
        period (int): Period for calculating the indicator.
    Returns:
        pd.DataFrame: A DataFrame with 'ChopZone'.
    """
    # Calculate Average True Range (ATR)
    atr = average_true_range(high, low, close, period)

    # Calculate Choppiness Index (CI)
    ci = choppiness_index(high, low, close, period)

    # Calculate Chop Zone
    result = (close - (open_ + high + low) / 3) * ci / atr

    # Name the resulting DataFrame
    result_df = pd.DataFrame(chop_zone, columns=["ChopZone"])

    return result_df


def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = np.maximum(delta, 0)
    loss = -np.minimum(delta, 0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()

    # Avoid division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.fillna(100)  # set initial rsi to 100 where avg_loss == 0

    return rsi


def _streak(series: pd.Series) -> pd.Series:
    streak = (series.diff() > 0).astype(int) - (series.diff() < 0).astype(int)
    return streak.groupby((streak != streak.shift()).cumsum()).cumsum()


def _percent_rank(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=1).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    )


def connors_rsi(
    close: pd.Series,
    rsi_period: int = 3,
    streak_rsi_period: int = 2,
    rank_period: int = 100,
) -> pd.Series:
    rsi_closing = _rsi(close, rsi_period)

    streak = _streak(close)

    rsi_streak = _rsi(streak, streak_rsi_period)

    price_change = close.diff()
    rank = _percent_rank(price_change, rank_period)

    result = (rsi_closing + rsi_streak + rank) / 3

    return pd.Series(result, index=close.index, name="ConnorsRSI")


def correlation_coefficient(close: pd.Series, volume: pd.Series, period: int = 30) -> pd.Series:
    """
    Calculate the Correlation Coefficient between two price series.
    Args:
        series1 (pd.Series): First series of prices or returns.
        series2 (pd.Series): Second series of prices or returns.
        period (int): The number of periods over which to calculate the correlation.
    Returns:
        pd.Series: A series representing the correlation coefficient over the specified period.
    """
    # Ensure both series have the same index to properly align them
    close, volume = close.align(volume, join="inner")

    # Calculate the rolling correlation
    correlation = close.rolling(window=period).corr(volume)

    return correlation.rename("CorrelationCoefficient")


def _cumulative_volume_delta(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Calculate Cumulative Volume Delta with enhanced handling for unchanged prices.
    Args:
        close (pd.Series): Series of closing prices.
        volume (pd.Series): Series of trading volumes.
    Returns:
        pd.Series: Series with 'cumulative_volume_delta'.
    """
    # Calculate price changes
    price_change = close.diff()

    # Initialize volume direction series
    volume_direction = pd.Series(0, index=close.index)

    # Assign volume based on the direction of price change
    volume_direction[price_change > 0] = volume[
        price_change > 0
    ]  # Positive volume for price increase
    volume_direction[price_change < 0] = -volume[
        price_change < 0
    ]  # Negative volume for price decrease

    # Handle cases where the price doesn't change
    # Option: carry over the previous day's volume direction
    unchanged_indices = price_change == 0
    if unchanged_indices.any():
        # Forward-fill the last known volume direction
        last_direction = volume_direction[~unchanged_indices].reindex(close.index).ffill()
        volume_direction[unchanged_indices] = last_direction[unchanged_indices]

    # Cumulative sum of directed volume
    result = volume_direction.cumsum()

    return result


def cumulative_volume_delta(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Calculate Cumulative Volume Delta using close prices and volume.
    Args:
        close (pd.Series): Series of closing prices indexed by date.
        volume (pd.Series): Series of trading volumes indexed by date.
    Returns:
        pd.Series: Cumulative Volume Delta.
    """
    # Determine the direction of the price move
    direction = close.diff().apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)

    # Calculate volume delta: volume multiplied by direction of price movement
    volume_delta = volume * direction

    # Cumulative sum to get the Cumulative Volume Delta
    cumulative_delta = volume_delta.cumsum()

    return cumulative_delta


def cumulative_volume_index(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Calculate Cumulative Volume Index.
    Args:
        close (pd.Series): Series of closing prices.
        volume (pd.Series): Series of volume data.
    Returns:
        pd.Series: Series with 'cumulative_volume_index'.
    """
    volume_change = volume * close.diff().apply(lambda x: 1 if x > 0 else 0)
    result = volume_change.cumsum()
    return pd.Series(result, index=close.index, name="cumulative_volume_index")


def directional_movement_index(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.DataFrame:
    """
    Calculate the Directional Movement Index (DMI).
    Args:
        high (pd.Series): Series of high prices.
        low (pd.Series): Series of low prices.
        close (pd.Series): Series of closing prices.
        period (int): The period over which to calculate the indicators.
    Returns:
        pd.DataFrame: DataFrame containing +DI, -DI, and ADX.
    """
    # Calculate the differences
    up_move = high.diff()
    down_move = low.diff().abs()

    # Determine the directional movement
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    # Calculate the True Range
    tr = np.maximum.reduce([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()])

    # Smooth the True Range and Directional Movements
    atr = pd.Series(tr).rolling(window=period).sum()
    smooth_plus_dm = pd.Series(plus_dm).rolling(window=period).sum()
    smooth_minus_dm = pd.Series(minus_dm).rolling(window=period).sum()

    # Calculate +DI and -DI
    plus_di = 100 * (smooth_plus_dm / atr)
    minus_di = 100 * (smooth_minus_dm / atr)

    # Calculate ADX
    dx = 100 * np.abs((plus_di - minus_di) / (plus_di + minus_di))
    adx = dx.rolling(window=period).mean()

    return pd.DataFrame({"+DI": plus_di, "-DI": minus_di, "ADX": adx}, index=high.index)


def envelopes(close: pd.Series, period: int = 20, percentage: float = 0.025) -> pd.DataFrame:
    """
    Calculate Envelopes.
    Args:
        close (pd.Series): Series of closing prices.
        period (int): The period for the simple moving average.
        percentage (float): The percentage above and below the moving average to set the bands.
    Returns:
        pd.DataFrame: DataFrame containing the moving average, upper envelope, and lower envelope.
    """
    # Calculate the moving average (SMA)
    sma = close.rolling(window=period).mean()

    # Calculate the upper and lower envelopes
    upper_envelope = sma * (1 + percentage)
    lower_envelope = sma * (1 - percentage)

    return pd.DataFrame(
        {"SMA": sma, "Upper Envelope": upper_envelope, "Lower Envelope": lower_envelope},
        index=close.index,
    )


def gaps(high: pd.Series, low: pd.Series) -> pd.DataFrame:
    """
    Identify gaps in trading.
    Args:
        high (pd.Series): Series of high prices.
        low (pd.Series): Series of low prices.
    Returns:
        pd.DataFrame: DataFrame indicating gaps with boolean values.
    """
    gap_up = low.shift(-1) > high
    gap_down = high.shift(-1) < low
    return pd.DataFrame({"gap_up": gap_up, "gap_down": gap_down}, index=high.index)


def historical_volatility(close: pd.Series, period: int = 30, trading_days: int = 252) -> pd.Series:
    """
    Calculate Historical Volatility using logarithmic returns.
    Args:
        close (pd.Series): Series of closing prices.
        period (int): The number of periods to calculate the standard deviation over.
        trading_days (int): Number of trading days in a year, used to annualize the volatility.
    Returns:
        pd.Series: Series representing the Historical Volatility, annualized.
    """
    # Calculate logarithmic returns
    log_returns = np.log(close / close.shift(1))

    # Calculate the rolling standard deviation of the log returns
    rolling_std = log_returns.rolling(window=period).std()

    # Annualize the volatility
    annualized_volatility = rolling_std * np.sqrt(trading_days)

    annualized_volatility.name = "HistoricalVolatility"
    return annualized_volatility


def keltner_channels(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    ema_period: int = 20,
    atr_period: int = 10,
    multiplier: float = 2.0,
) -> pd.DataFrame:
    """
    Calculate Keltner Channels.
    Args:
        high (pd.Series): Series of high prices.
        low (pd.Series): Series of low prices.
        close (pd.Series): Series of closing prices.
        ema_period (int): The period for the exponential moving average.
        atr_period (int): The period for the Average True Range.
        multiplier (float): The multiplier for setting channel width.
    Returns:
        pd.DataFrame: DataFrame containing the central line (EMA), upper channel, and lower channel.
    """
    # Calculate the EMA of the closing prices
    central_line = close.ewm(span=ema_period, adjust=False).mean()

    # Calculate the Average True Range (ATR)
    tr = high - low
    previous_close = close.shift(1)
    tr = pd.DataFrame(
        {
            "High-Low": tr,
            "High-PrevClose": (high - previous_close).abs(),
            "Low-PrevClose": (low - previous_close).abs(),
        }
    )
    atr = tr.max(axis=1).rolling(window=atr_period).mean()

    # Calculate the upper and lower bands
    upper_channel = central_line + (multiplier * atr)
    lower_channel = central_line - (multiplier * atr)

    return pd.DataFrame(
        {
            "Central Line": central_line,
            "Upper Channel": upper_channel,
            "Lower Channel": lower_channel,
        },
        index=high.index,
    )


def _know_sure_thing(
    close: pd.Series,
    roc1_period=10,
    roc1_ma_period=10,
    roc2_period=15,
    roc2_ma_period=10,
    roc3_period=20,
    roc3_ma_period=10,
    roc4_period=30,
    roc4_ma_period=15,
    signal_period=9,
) -> pd.Series:
    """
    Calculate the Know Sure Thing (KST) indicator.
    Args:
        close (pd.Series): Series of closing prices.
        roc1_period, roc2_period, roc3_period, roc4_period (int): Rates of change periods.
        roc1_ma_period, roc2_ma_period, roc3_ma_period, roc4_ma_period (int): MA periods for each ROC.
        signal_period (int): Moving average period for the signal line.
    Returns:
        pd.Series: A Series representing the Know Sure Thing (KST) indicator, or None if not enough data.
    """
    kst = talipp.indicators.KST(
        roc1_period,
        roc1_ma_period,
        roc2_period,
        roc2_ma_period,
        roc3_period,
        roc3_ma_period,
        roc4_period,
        roc4_ma_period,
        signal_period,
    )
    for c in close:
        kst.add_input_value(c)

    # Check if the KST indicator has generated enough data
    if len(kst) > 0:
        result = pd.Series(
            [val.value for val in kst], index=close.index[-len(kst) :], name="KnowSureThing"
        )
        return result
    # Return an empty Series if not enough data
    return pd.Series([], index=close.index, name="KnowSureThing")


def know_sure_thing(close, roc_periods=None, ma_periods=None, signal_ma_period=9):
    """
    Calculate the Know Sure Thing (KST) indicator with default periods based on TradingView settings.
    Args:
        close (pd.Series): Series of closing prices.
        roc_periods (list): List of periods for rates of change.
        ma_periods (list): List of periods for moving averages of the ROCs.
        signal_ma_period (int): Period for the signal line moving average.
    Returns:
        pd.DataFrame: DataFrame containing the KST and signal line.
    """

    roc_periods = [10, 15, 20, 30] if not roc_periods else roc_periods
    ma_periods = [10, 10, 10, 15] if not ma_periods else ma_periods

    # Calculate ROCs and their MAs
    def smaroc(roc_len, sma_len):
        roc = close.diff(roc_len) / close.shift(roc_len)
        return roc.rolling(window=sma_len).mean()

    rocs = [smaroc(roc_len, sma_len) for roc_len, sma_len in zip(roc_periods, ma_periods)]
    weights = [1, 2, 3, 4]  # Increasing weights for the contribution of each ROC period

    # Weighted sum of the MAs of ROCs
    kst = sum(w * roc for w, roc in zip(weights, rocs))

    # Calculate the signal line
    signal = kst.rolling(window=signal_ma_period).mean()

    return pd.DataFrame({"KST": kst, "Signal": signal}, index=close.index)


def least_squares_moving_average(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate the Least Squares Moving Average (LSMA) or Linear Regression Moving Average.
    Args:
        close (pd.Series): Series of closing prices.
        period (int): The number of periods over which to calculate the LSMA.
    Returns:
        pd.Series: A series containing the LSMA values.
    """
    # Initialize the index for the X variable (time)
    x = np.arange(period)

    # Function to calculate LSMA for each window
    def lin_reg_end_point(y):
        if len(y) < period:
            return y.iloc[-1]  # return last value if not enough data
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        return m * (period - 1) + c  # Calculate the endpoint y value of the regression line

    # Apply the function over a rolling window
    lsma = close.rolling(window=period).apply(lin_reg_end_point, raw=True)

    return lsma.rename("LSMA")


def ma_cross(close: pd.Series, fast_period: int = 12, slow_period: int = 26) -> pd.DataFrame:
    """
    Calculate the Moving Average Cross (MA Cross).
    Args:
        close (pd.Series): Series of closing prices.
        fast_period (int): The period for the fast moving average.
        slow_period (int): The period for the slow moving average.
    Returns:
        pd.DataFrame: DataFrame containing the fast and slow moving averages.
    """
    # Calculate moving averages
    fast_ma = close.rolling(window=fast_period).mean()
    slow_ma = close.rolling(window=slow_period).mean()

    # Create a DataFrame to store both MAs
    ma_cross_df = pd.DataFrame(
        {
            "Fast_MA": fast_ma,
            "Slow_MA": slow_ma,
            "Signal": (fast_ma > slow_ma).astype(int) - (fast_ma < slow_ma).astype(int),
        },
        index=close.index,
    )

    return ma_cross_df


def mcginley_dynamic(close: pd.Series, N: int = 14, K: float = 0.6) -> pd.Series:
    """
    Calculate the McGinley Dynamic.
    Args:
        close (pd.Series): Series of closing prices.
        N (int): The smoothing period.
        K (float): The smoothing factor (typically around 0.6).
    Returns:
        pd.Series: Series representing the McGinley Dynamic.
    """
    md = pd.Series(index=close.index)
    md.iloc[0] = close.iloc[0]  # Initialize the first value to the first close price

    for i in range(1, len(close)):
        md.iloc[i] = md.iloc[i - 1] + (close.iloc[i] - md.iloc[i - 1]) / (
            K * N * ((close.iloc[i] / md.iloc[i - 1]) ** 4)
        )

    md.name = "McGinleyDynamic"
    return md


def median_price(high: pd.Series, low: pd.Series) -> pd.Series:
    """
    Calculate the Median Price.
    Args:
        high (pd.Series): Series of high prices.
        low (pd.Series): Series of low prices.
    Returns:
        pd.Series: Series representing the Median Price.
    """
    median = (high + low) / 2
    median.name = "MedianPrice"
    return median


def money_flow_index(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series
) -> pd.Series:
    """
    Calculate Money Flow.
    Args:
        high (pd.Series): Series of high prices.
        low (pd.Series): Series of low prices.
        close (pd.Series): Series of closing prices.
        volume (pd.Series): Series of volume data.
    Returns:
        pd.Series: Series with 'money_flow'.
    """
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume
    return pd.Series(money_flow, index=close.index, name="money_flow")


def moon_phases(timestamps: pd.DatetimeIndex) -> pd.Series:
    """
    Simulated function to assign moon phases based on the index of the data.
    This function is purely illustrative since real moon phase calculation requires specific astronomical data.
    Args:
        index (pd.DatetimeIndex): Datetime index of the data.
    Returns:
        pd.Series: Series with 'moon_phase'.
    """
    phases = ["New Moon", "First Quarter", "Full Moon", "Last Quarter"]
    moon_phase = [phases[i % 4] for i in range(len(timestamps))]
    return pd.Series(moon_phase, index=timestamps, name="moon_phase")


def moving_average_ribbon(close: pd.Series, lengths: list = [5, 10, 20, 50, 100]) -> pd.DataFrame:
    """
    Calculate Moving Average Ribbon.
    Args:
        close (pd.Series): Series of closing prices.
        lengths (list, optional): List of integers representing the lengths of the moving averages.
                                Defaults to [5, 10, 20, 50, 100] if not provided.
    Returns:
        pd.DataFrame: DataFrame with each column as a moving average of specified length.
    """
    ribbons = pd.DataFrame(index=close.index)
    for length in lengths:
        ribbons[f"SMA_{length}"] = close.rolling(window=length).mean()

    return ribbons


def multi_time_period_charts(close: pd.Series) -> pd.Series:
    """
    Calculate representative values for multiple time periods.
    Args:
        close (pd.Series): Series of closing prices.
    Returns:
        pd.Series: Series with 'multi_time_period'.
    """
    multi_time_period = close.expanding().mean()
    return pd.Series(multi_time_period, index=close.index, name="multi_time_period")


def net_volume(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Calculate Net Volume.
    Args:
        close (pd.Series): Series of closing prices.
        volume (pd.Series): Series of volume data.
    Returns:
        pd.Series: Series with 'net_volume'.
    """
    direction = close.diff().apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
    result = (volume * direction).cumsum()
    return pd.Series(result, index=close.index, name="net_volume")


def on_balance_volume(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Calculate On Balance Volume (OBV).
    OBV measures buying and selling pressure as a cumulative indicator, adding volume on up days and subtracting volume on down days.
    Args:
        close (pd.Series): Series of closing prices.
        volume (pd.Series): Series of trading volumes.
    Returns:
        pd.Series: Series representing the On Balance Volume.
    """
    # Initialize OBV series with zeros
    obv = pd.Series(0, index=close.index)

    # Calculate direction of the day
    direction = close.diff()
    direction.fillna(0, inplace=True)  # Replace NaN values for the first element

    # Applying conditions to calculate OBV
    for i in range(1, len(close)):
        if direction[i] > 0:
            obv[i] = obv[i - 1] + volume[i]
        elif direction[i] < 0:
            obv[i] = obv[i - 1] - volume[i]
        else:
            obv[i] = obv[i - 1]

    obv.name = "OBV"
    return obv


def open_interest(volume: pd.Series) -> pd.Series:
    """
    Simulated Open Interest calculation (as real data is required).
    Args:
        volume (pd.Series): Series of volume data.
    Returns:
        pd.Series: Series with 'open_interest'.
    """
    open_interest = volume.cumsum()  # Assuming open interest is a cumulative sum of volumes
    return pd.Series(open_interest, index=volume.index, name="open_interest")


def pivot_points_high_low(close: pd.Series, high: pd.Series, low: pd.Series) -> pd.DataFrame:
    """
    Calculate Pivot Points High Low.
    Args:
        close (pd.Series): Series of closing prices.
        high (pd.Series): Series of high prices.
        low (pd.Series): Series of low prices.
    Returns:
        pd.DataFrame: A DataFrame with columns 'PivotPoints_High' and 'PivotPoints_Low'.
    """
    # Create the PivotsHL object with defined periods for high and low pivots
    pivot_points = PivotsHL(PPL_HIGH_PERIOD, PPL_LOW_PERIOD)

    # Feed data into the pivot_points object
    for _, (h, l, c) in enumerate(zip(high, low, close)):
        pivot_points.add_input_value(
            talipp.ohlcv.OHLCV(h, l, c, 0, 0)
        )  # Assuming volume and open are not used

    # Prepare results to store in DataFrame
    dates = close.index
    highs = [
        (dates[i], val.ohlcv.high)
        for i, val in enumerate(pivot_points.output_values)
        if val.type == PivotType.HIGH
    ]
    lows = [
        (dates[i], val.ohlcv.low)
        for i, val in enumerate(pivot_points.output_values)
        if val.type == PivotType.LOW
    ]

    # Create DataFrames and combine them
    df_highs = pd.DataFrame(highs, columns=["Date", "PivotPoints_High"]).set_index("Date")
    df_lows = pd.DataFrame(lows, columns=["Date", "PivotPoints_Low"]).set_index("Date")

    # Combine the high and low pivot points into a single DataFrame
    return pd.concat([df_highs, df_lows], axis=1)


def pivot_points_standard(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.DataFrame:
    """
    Calculate the Pivot Points Standard.
    Args:
        high (pd.Series): Series of high prices.
        low (pd.Series): Series of low prices.
        close (pd.Series): Series of closing prices.
    Returns:
        pd.DataFrame: DataFrame containing the Pivot Point and support/resistance levels (S1, S2, R1, R2).
    """
    # Pivot Point Calculation
    PP = (high + low + close) / 3

    # Support and Resistance Levels
    R1 = 2 * PP - low
    S1 = 2 * PP - high
    R2 = PP + (high - low)
    S2 = PP - (high - low)

    # Additional levels can be added as needed
    R3 = high + 2 * (PP - low)
    S3 = low - 2 * (high - PP)

    # Organize data into a DataFrame
    pivot_data = pd.DataFrame(
        {
            "Pivot_Point": PP,
            "Resistance1": R1,
            "Support1": S1,
            "Resistance2": R2,
            "Support2": S2,
            "Resistance3": R3,
            "Support3": S3,
        },
        index=high.index,
    )

    return pivot_data


def price_target(high: pd.Series) -> pd.Series:
    """
    Example function to simulate price target based on some strategy.
    Args:
        high (pd.Series): Series of high prices.
    Returns:
        pd.Series: Series with 'price_target'.
    """
    result = high * 1.1  # Example: setting target as 110% of recent high
    return pd.Series(result, index=high.index, name="price_target")


def relative_volume_at_time(volume: pd.Series) -> pd.Series:
    """
    Calculate relative volume at a specific time compared to the average volume at that time over a period.
    Args:
        volume (pd.Series): Series of volume data.
    Returns:
        pd.Series: Series with 'relative_volume'.
    """
    # Ensure the index is a DateTimeIndex
    if not pd.api.types.is_datetime64_any_dtype(volume.index):
        volume.index = pd.to_datetime(volume.index)  # Assuming the index can be converted directly

    hour = volume.index.hour
    avg_volume = volume.groupby(hour).transform("mean")
    relative_volume = volume / avg_volume
    return pd.Series(relative_volume, index=volume.index, name="relative_volume")


def rob_booker_ghost_pivots_v2(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """
    Simulated function for Rob Booker's Ghost Pivots V2.
    Args:
        high (pd.Series): Series of high prices.
        low (pd.Series): Series of low prices.
        close (pd.Series): Series of closing prices.
    Returns:
        pd.Series: Series with 'ghost_pivots'.
    """
    ghost_pivots = (high + low + close) / 3
    return pd.Series(ghost_pivots, index=close.index, name="ghost_pivots")


def divergence_indicator(close: pd.Series) -> pd.Series:
    """
    Simulated function to calculate a divergence indicator.
    Args:
        close (pd.Series): Series of closing prices.
    Returns:
        pd.Series: Series with 'divergence'.
    """
    divergence = np.where(close.diff() > 0, 1, -1)
    return pd.Series(divergence, index=close.index, name="divergence")


def seasonality(index: pd.DatetimeIndex) -> pd.Series:
    """
    Example function to simulate seasonality effects.
    Args:
        index (pd.DatetimeIndex): Datetime index of the data.
    Returns:
        pd.Series: Series with 'seasonality'.
    """
    month = index.month
    result = month % 4
    return pd.Series(result, index=index, name="seasonality")


def smi_ergodic(close: pd.Series, k_period: int = 5, signal_period: int = 5) -> pd.DataFrame:
    """
    Calculate the SMI Ergodic Indicator/Oscillator.
    Args:
        close (pd.Series): Series of closing prices.
        k_period (int): Period for the moving average calculation.
        signal_period (int): Period for the signal line calculation.
    Returns:
        pd.DataFrame: A DataFrame with the SMI Ergodic Indicator and its signal line.
    """
    # Price change and absolute price change
    delta_close = close.diff()
    abs_delta_close = delta_close.abs()

    # Smoothed values
    smoothed_delta = delta_close.ewm(span=k_period, adjust=False).mean()
    smoothed_abs_delta = abs_delta_close.ewm(span=k_period, adjust=False).mean()

    # SMI Ergodic Calculation
    result = smoothed_delta / smoothed_abs_delta

    # Signal line
    signal_line = result.ewm(span=signal_period, adjust=False).mean()

    return pd.DataFrame({"SMI_Ergodic": result, "Signal": signal_line}, index=close.index)


def smoothed_moving_average(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate the Smoothed Moving Average (SMMA).
    Args:
        close (pd.Series): Series of closing prices.
        period (int): Period for the moving average.
    Returns:
        pd.Series: Smoothed moving average series.
    """
    # Initial simple moving average to start the SMMA calculation
    initial_sma = close.head(period).mean()

    # Initialize the SMMA series
    smma = pd.Series(0, index=close.index)
    smma.iloc[period - 1] = initial_sma

    # Calculate SMMA
    for i in range(period, len(smma)):
        smma.iloc[i] = (smma.iloc[i - 1] * (period - 1) + close.iloc[i]) / period

    return smma


def stochastic_momentum_index(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 5,
    d_period: int = 3,
    smoothing: int = 3,
) -> pd.DataFrame:
    """
    Calculate the Stochastic Momentum Index (SMI).
    Args:
        high (pd.Series): Series of high prices.
        low (pd.Series): Series of low prices.
        close (pd.Series): Series of closing prices.
        k_period (int): The lookback period for the high and low (default 5).
        d_period (int): The smoothing period for %K line (default 3).
        smoothing (int): The smoothing period for %D line (default 3).
    Returns:
        pd.DataFrame: A DataFrame containing the SMI lines '%K' and '%D'.
    """
    # Calculate the midpoint of the high-low range
    midpoint = (high + low) / 2

    # Calculate the closing price relative to the midpoint
    close_midpoint_diff = close - midpoint

    # Calculate the range (difference between high and low)
    high_low_range = high - low

    # Smooth the close-midpoint differences and range using an EMA
    smoothed_close_mid_diff = close_midpoint_diff.ewm(span=k_period, adjust=False).mean()
    smoothed_range = high_low_range.ewm(span=k_period, adjust=False).mean()

    # Calculate %K as smoothed midpoint deviation divided by smoothed range
    K = 100 * (smoothed_close_mid_diff / (smoothed_range / 2))

    # Smooth %K to get %D using another EMA
    D = K.ewm(span=d_period, adjust=False).mean()

    # Apply final smoothing to %D to align with traditional SMI calculations
    smoothed_D = D.ewm(span=smoothing, adjust=False).mean()

    # Create a DataFrame to store both %K and smoothed %D
    result = pd.DataFrame({"K": K, "D": smoothed_D}, index=high.index)

    return result


def technical_ratings(close: pd.Series) -> pd.Series:
    """
    Example function to simulate technical ratings based on moving averages.
    Args:
        close (pd.Series): Series of closing prices.
    Returns:
        pd.Series: Series with 'technical_ratings'.
    """
    sma = close.rolling(window=20).mean()
    result = np.where(close > sma, "Buy", "Sell")
    return pd.Series(result, index=close.index, name="technical_ratings")


def time_weighted_average_price(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Calculate Time Weighted Average Price (TWAP).
    Args:
        close (pd.Series): Series of closing prices.
        volume (pd.Series): Series of volume data.
    Returns:
        pd.Series: Series with 'twap'.
    """
    cumulative_sum = (close * volume).cumsum()
    cumulative_volume = volume.cumsum()
    twap = cumulative_sum / cumulative_volume
    return pd.Series(twap, index=close.index, name="twap")


def trading_sessions(timestamps: pd.Series) -> pd.Series:
    """
    Simulated function to indicate trading session based on time.
    Args:
        timestamps (pd.Series): Series containing datetime values.
    Returns:
        pd.Series: Series with 'trading_session'.
    """
    # Ensure the series contains datetime data
    if not pd.api.types.is_datetime64_any_dtype(timestamps.dtype):
        timestamps = pd.to_datetime(timestamps)

    hour = timestamps.hour
    return pd.Series(
        np.where((hour >= 9) & (hour < 16), "Regular", "After-hours"), index=timestamps
    )


def trend_strength_index(close: pd.Series, r: int = 25, s: int = 13) -> pd.Series:
    """
    Calculate the Trend Strength Index (TSI).
    Args:
        close (pd.Series): Series of closing prices.
        r (int): The length of the first smoothing of the momentum, default 25.
        s (int): The length of the second smoothing, default 13.
    Returns:
        pd.Series: Series with TSI values.
    """
    # Calculate the price momentum (price change)
    momentum = close.diff()

    # Smooth the momentum with an EMA (first smoothing)
    ema_first = momentum.ewm(span=r, adjust=False).mean()

    # Smooth the first EMA with another EMA (second smoothing)
    ema_second = ema_first.ewm(span=s, adjust=False).mean()

    # Calculate the absolute of the momentum to normalize TSI
    abs_momentum = abs(momentum)

    # Smooth the absolute momentum with an EMA (first smoothing)
    abs_ema_first = abs_momentum.ewm(span=r, adjust=False).mean()

    # Smooth the first EMA of the absolute momentum with another EMA (second smoothing)
    abs_ema_second = abs_ema_first.ewm(span=s, adjust=False).mean()

    # Calculate TSI
    tsi = 100 * (ema_second / abs_ema_second)

    return pd.Series(tsi, index=close.index, name="TSI")


def up_down_volume(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Calculate Up/Down Volume.
    Args:
        close (pd.Series): Series of closing prices.
        volume (pd.Series): Series of volume data.
    Returns:
        pd.Series: Series with 'up_down_volume'.
    """
    price_change = close.diff()
    result = np.where(price_change > 0, volume, -volume)
    return pd.Series(result, index=close.index, name="up_down_volume")


def visible_average_price(close: pd.Series) -> pd.Series:
    """
    Example function to simulate a visible average price.
    Args:
        close (pd.Series): Series of closing prices.
    Returns:
        pd.Series: Series with 'visible_average_price'.
    """
    result = close.expanding().mean()
    return pd.Series(result, index=close.index, name="visible_average_price")


def volatility_stop(high: pd.Series, low: pd.Series) -> pd.Series:
    """
    Simulate calculation of a volatility stop indicator.
    Args:
        high (pd.Series): Series of high prices.
        low (pd.Series): Series of low prices.
    Returns:
        pd.Series: Series with 'volatility_stop'.
    """
    result = (high - low).rolling(window=14).mean()
    return pd.Series(result, index=high.index, name="volatility_stop")


def vol(volume: pd.Series) -> pd.Series:
    """
    Calculate a Volume Indicator that highlights significant changes in volume relative to the recent average.
    Args:
        close (pd.Series): Series of closing prices.
        open (pd.Series): Series of opening prices.
        high (pd.Series): Series of high prices.
        low (pd.Series): Series of low prices.
        volume (pd.Series): Series of volume data.
    Returns:
        pd.Series: A Series indicating where the volume is significantly higher or lower than the recent average.
    """
    # Calculate the 10-period moving average of the volume
    moving_average_volume = volume.rolling(window=10).mean()

    # Calculate the volume change signal as the ratio of current volume to moving average
    volume_change_signal = volume / moving_average_volume

    # Return the signal, highlighted as significant if greater than 1.5 times the average or less than 0.5
    volume_indicator = (volume_change_signal > 1.5) | (volume_change_signal < 0.5)

    return pd.Series(volume_indicator.astype(int), index=volume.index, name="VolumeIndicator")


def volume_delta(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Calculate Volume Delta (difference between buying and selling volume).
    Args:
        close (pd.Series): Series of closing prices.
        volume (pd.Series): Series of volume data.
    Returns:
        pd.Series: Series with 'volume_delta'.
    """
    # Calculate price direction
    price_direction = close.diff().apply(lambda x: 1 if x > 0 else -1)

    # Calculate volume delta
    result = volume * price_direction

    # Return as a pandas Series
    return pd.Series(result, index=close.index, name="volume_delta")


def alligator(close: pd.Series) -> pd.DataFrame:
    """
    Calculate Williams Alligator indicator.
    Args:
        close (pd.Series): Series of closing prices.
    Returns:
        pd.DataFrame: A DataFrame with columns 'jaw', 'teeth', 'lips'.
    """

    # Smoothed moving average function
    def smoothed_ma(series, window, shift):
        ma = series.rolling(window=window).mean()
        return ma.shift(shift)

    # Calculating the Alligator indicator lines
    jaw = smoothed_ma(close, 13, 8)  # Blue line
    teeth = smoothed_ma(close, 8, 5)  # Red line
    lips = smoothed_ma(close, 5, 3)  # Green line

    return pd.DataFrame({"jaw": jaw, "teeth": teeth, "lips": lips}, index=close.index)


def williams_fractals(high: pd.Series, low: pd.Series) -> pd.DataFrame:
    """
    Calculate Williams Fractals.
    Args:
        high (pd.Series): Series of high prices.
        low (pd.Series): Series of low prices.
    Returns:
        pd.DataFrame: A DataFrame with columns 'fractal_bear', 'fractal_bull'.
    """
    shift_size = 2
    bear_fractal = (
        (high.shift(shift_size) < high.shift(shift_size * 2))
        & (high.shift(shift_size) < high.shift(-shift_size * 2))
        & (high.shift(shift_size) < high.shift(shift_size))
        & (high.shift(shift_size) < high.shift(-shift_size))
    )
    bull_fractal = (
        (low.shift(shift_size) > low.shift(shift_size * 2))
        & (low.shift(shift_size) > low.shift(-shift_size * 2))
        & (low.shift(shift_size) > low.shift(shift_size))
        & (low.shift(shift_size) > low.shift(-shift_size))
    )

    return pd.DataFrame(
        {"fractal_bear": bear_fractal.astype(int), "fractal_bull": bull_fractal.astype(int)},
        index=high.index,
    )


def williams_percent_r(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """
    Calculate Williams %R.
    Args:
        high (pd.Series): Series of high prices.
        low (pd.Series): Series of low prices.
        close (pd.Series): Series of closing prices.
        period (int): Look-back period.
    Returns:
        pd.Series: Series with Williams %R values.
    """
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()
    williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))

    return williams_r


def woodies_cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    """
    Calculate Woodie's CCI.
    Args:
        high (pd.Series): Series of high prices.
        low (pd.Series): Series of low prices.
        close (pd.Series): Series of closing prices.
        period (int): Look-back period typically set to 20.
    Returns:
        pd.Series: Series with Woodie's CCI values.
    """
    tp = (high + low + close) / 3
    cci = (tp - tp.rolling(window=period).mean()) / (0.015 * tp.rolling(window=period).std())

    return cci


def zigzag(high: pd.Series, low: pd.Series, pct_change: float = 7) -> pd.Series:
    """
    Calculate Zig Zag indicator.
    Args:
        high (pd.Series): Series of high prices.
        low (pd.Series): Series of low prices.
        pct_change (float): Percentage change to determine reversals.
    Returns:
        pd.Series: Series with Zig Zag values.
    """
    # Define lists to store pivot points and their indices
    pivots = []
    pivot_indices = []

    # Initial setup
    direction = 0  # 0 indicates no direction, 1 for up, -1 for down
    last_pivot = high[0] if high[0] > low[0] else low[0]
    last_index = high.index[0]

    # Start with the second bar to compare changes
    for i in range(1, len(high)):
        if direction == 0:  # Identify the initial pivot direction
            if (high[i] / last_pivot - 1) * 100 > pct_change:
                direction = 1  # upward pivot
                pivots.append(last_pivot)
                pivot_indices.append(last_index)
            elif (last_pivot / low[i] - 1) * 100 > pct_change:
                direction = -1  # downward pivot
                pivots.append(last_pivot)
                pivot_indices.append(last_index)
        elif direction == 1:  # Current direction upward, looking for downward pivot
            if (last_pivot / low[i] - 1) * 100 > pct_change:
                pivots.append(last_pivot)  # Confirm the last pivot
                pivot_indices.append(last_index)
                last_pivot = low[i]
                last_index = low.index[i]
                direction = -1
            elif high[i] > last_pivot:
                last_pivot = high[i]
                last_index = high.index[i]
        elif direction == -1:  # Current direction downward, looking for upward pivot
            if (high[i] / last_pivot - 1) * 100 > pct_change:
                pivots.append(last_pivot)  # Confirm the last pivot
                pivot_indices.append(last_index)
                last_pivot = high[i]
                last_index = high.index[i]
                direction = 1
            elif low[i] < last_pivot:
                last_pivot = low[i]
                last_index = low.index[i]

    # Add the last pivot found
    pivots.append(last_pivot)
    pivot_indices.append(last_index)

    # Create a Series from the pivots
    zz = pd.Series(index=high.index, data=np.nan)
    zz.loc[pivot_indices] = pivots
    zz = zz.ffill()  # Forward fill the values to draw lines

    return zz


def ichimoku_cloud(high: pd.Series, low: pd.Series, close: pd.Series):
    """
    P.B. source: https://www.investopedia.com/terms/i/ichimoku-cloud.asp
    """
    tenkan_sen, kijun_sen = pandas_ta.ichimoku(high, low, close)

    if tenkan_sen is None and kijun_sen is None:
        return pd.DataFrame()

    result = pd.concat([tenkan_sen, kijun_sen], axis=1)
    result.columns = [
        "Tenkan_ISA9",
        "Tenkan_ISB26",
        "Tenkan_ITS9",
        "Tenkan_IKS26",
        "Tenkan_ICS26",
        "Kijun_ISA9",
        "Kijun_ISB26",
    ]

    return result


def aroon(high: pd.Series, low: pd.Series, length=14):
    """
    P.B. source: https://www.investopedia.com/terms/a/aroon.asp
    """
    # Dynamically set length if not provided, or if the series is shorter than 14
    data_size = len(high)

    if data_size < length:
        length = max(2, data_size)  # Set length to at least 2 to allow calculation

    # Use pandas_ta's aroon method to calculate the indicator
    ans = pandas_ta.aroon(high, low, length=length)

    # Return the result
    return ans if ans is not None else pd.DataFrame()


def accumulation_distirbution(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series):
    return pandas_ta.ad(high, low, close, volume)


def average_directional_index(high: pd.Series, low: pd.Series, close: pd.Series):
    ans = pandas_ta.adx(high, low, close)
    return ans if isinstance(ans, pd.DataFrame) else pd.Series()

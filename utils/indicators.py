"""
Indicator Calculation Utilities
Collection of functions to calculate various trading indicators
"""

import pandas as pd
import numpy as np
import talib
import pandas_ta as ta
from typing import Union, Tuple, Optional


class IndicatorCalculator:
    """
    Comprehensive indicator calculation utilities
    """
    
    def __init__(self):
        pass
    
    @staticmethod
    def sma(data: pd.Series, period: int) -> pd.Series:
        """
        Simple Moving Average
        """
        return data.rolling(window=period).mean()
    
    @staticmethod
    def ema(data: pd.Series, period: int) -> pd.Series:
        """
        Exponential Moving Average
        """
        return data.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """
        Relative Strength Index
        """
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def macd(data: pd.Series, 
             fast_period: int = 12, 
             slow_period: int = 26, 
             signal_period: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Moving Average Convergence Divergence
        Returns: MACD line, Signal line, Histogram
        """
        ema_fast = IndicatorCalculator.ema(data, fast_period)
        ema_slow = IndicatorCalculator.ema(data, slow_period)
        macd_line = ema_fast - ema_slow
        signal_line = IndicatorCalculator.ema(macd_line, signal_period)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def bollinger_bands(data: pd.Series, 
                       period: int = 20, 
                       std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Bollinger Bands
        Returns: Upper band, Middle band, Lower band
        """
        middle_band = IndicatorCalculator.sma(data, period)
        std = data.rolling(window=period).std()
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)
        return upper_band, middle_band, lower_band
    
    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, 
                  k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """
        Stochastic Oscillator
        Returns: %K line, %D line
        """
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        return k_percent, d_percent
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """
        Average True Range
        """
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()
    
    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """
        Average Directional Index
        """
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate Directional Movement
        up_move = high - high.shift()
        down_move = low.shift() - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Calculate Smoothed True Range and Directional Movement
        atr = tr.ewm(com=period-1, adjust=False).mean()
        plus_di = 100 * pd.Series(plus_dm).ewm(com=period-1, adjust=False).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).ewm(com=period-1, adjust=False).mean() / atr
        
        # Calculate ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.ewm(com=period-1, adjust=False).mean()
        
        return adx
    
    @staticmethod
    def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """
        Williams %R
        """
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        return -100 * (highest_high - close) / (highest_high - lowest_low)
    
    @staticmethod
    def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
        """
        Commodity Channel Index
        """
        tp = (high + low + close) / 3
        sma_tp = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
        return (tp - sma_tp) / (0.015 * mad)
    
    @staticmethod
    def roc(data: pd.Series, period: int = 12) -> pd.Series:
        """
        Rate of Change
        """
        return ((data - data.shift(period)) / data.shift(period)) * 100
    
    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        On-Balance Volume
        """
        obv = np.where(close > close.shift(), volume, 
                      np.where(close < close.shift(), -volume, 0))
        return pd.Series(obv).cumsum()
    
    @staticmethod
    def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Volume Weighted Average Price
        """
        typical_price = (high + low + close) / 3
        return (typical_price * volume).cumsum() / volume.cumsum()
    
    @staticmethod
    def fibonacci_retracement(high_price: float, low_price: float) -> dict:
        """
        Calculate Fibonacci retracement levels
        """
        diff = high_price - low_price
        levels = {
            '0%': high_price,
            '23.6%': high_price - 0.236 * diff,
            '38.2%': high_price - 0.382 * diff,
            '50%': high_price - 0.5 * diff,
            '61.8%': high_price - 0.618 * diff,
            '78.6%': high_price - 0.786 * diff,
            '100%': low_price
        }
        return levels
    
    @staticmethod
    def pivot_points(high: pd.Series, low: pd.Series, close: pd.Series, 
                    method: str = 'standard') -> pd.DataFrame:
        """
        Calculate Pivot Points
        Methods: 'standard', 'woodie', 'camarilla', 'fibonacci'
        """
        pivot = (high + low + close) / 3
        
        if method == 'standard':
            r1 = 2 * pivot - low
            s1 = 2 * pivot - high
            r2 = pivot + (high - low)
            s2 = pivot - (high - low)
            r3 = high + 2 * (pivot - low)
            s3 = low - 2 * (high - pivot)
            
        elif method == 'woodie':
            pivot = (high + low + 2 * close) / 4
            r1 = 2 * pivot - low
            s1 = 2 * pivot - high
            r2 = pivot + (high - low)
            s2 = pivot - (high - low)
            
        elif method == 'camarilla':
            r1 = close + 1.1 * (high - low) / 4
            r2 = close + 1.1 * (high - low) / 2
            r3 = close + 1.1 * (high - low) / 4
            r4 = close + 1.1 * (high - low)
            s1 = close - 1.1 * (high - low) / 4
            s2 = close - 1.1 * (high - low) / 2
            s3 = close - 1.1 * (high - low) / 4
            s4 = close - 1.1 * (high - low)
            
        elif method == 'fibonacci':
            r1 = pivot + 0.382 * (high - low)
            r2 = pivot + 0.618 * (high - low)
            r3 = pivot + 1.0 * (high - low)
            s1 = pivot - 0.382 * (high - low)
            s2 = pivot - 0.618 * (high - low)
            s3 = pivot - 1.0 * (high - low)
        
        # Create DataFrame with results
        result = pd.DataFrame({
            'pivot': pivot,
            'r1': r1 if method != 'camarilla' else r1,
            'r2': r2 if method != 'camarilla' else r2,
            'r3': r3 if method != 'camarilla' else r3,
            's1': s1 if method != 'camarilla' else s1,
            's2': s2 if method != 'camarilla' else s2,
            's3': s3 if method != 'camarilla' else s3,
        })
        
        if method == 'camarilla':
            result['r4'] = r4
            result['s4'] = s4
        
        return result
    
    @staticmethod
    def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all available indicators for a DataFrame
        """
        result_df = df.copy()
        
        # Trend Indicators
        result_df['sma_20'] = IndicatorCalculator.sma(df['close'], 20)
        result_df['sma_50'] = IndicatorCalculator.sma(df['close'], 50)
        result_df['ema_12'] = IndicatorCalculator.ema(df['close'], 12)
        result_df['ema_26'] = IndicatorCalculator.ema(df['close'], 26)
        
        # MACD
        macd, signal, hist = IndicatorCalculator.macd(df['close'])
        result_df['macd'] = macd
        result_df['macd_signal'] = signal
        result_df['macd_histogram'] = hist
        
        # Momentum Indicators
        result_df['rsi'] = IndicatorCalculator.rsi(df['close'])
        result_df['stoch_k'], result_df['stoch_d'] = IndicatorCalculator.stochastic(
            df['high'], df['low'], df['close'])
        result_df['williams_r'] = IndicatorCalculator.williams_r(
            df['high'], df['low'], df['close'])
        result_df['cci'] = IndicatorCalculator.cci(df['high'], df['low'], df['close'])
        result_df['roc'] = IndicatorCalculator.roc(df['close'])
        
        # Volatility Indicators
        bb_upper, bb_middle, bb_lower = IndicatorCalculator.bollinger_bands(df['close'])
        result_df['bb_upper'] = bb_upper
        result_df['bb_middle'] = bb_middle
        result_df['bb_lower'] = bb_lower
        result_df['bb_width'] = bb_upper - bb_lower
        result_df['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
        
        result_df['atr'] = IndicatorCalculator.atr(df['high'], df['low'], df['close'])
        result_df['adx'] = IndicatorCalculator.adx(df['high'], df['low'], df['close'])
        
        # Volume Indicators
        result_df['obv'] = IndicatorCalculator.obv(df['close'], df['volume'])
        result_df['vwap'] = IndicatorCalculator.vwap(df['high'], df['low'], df['close'], df['volume'])
        
        # Pivot Points
        pivots = IndicatorCalculator.pivot_points(df['high'], df['low'], df['close'])
        for col in pivots.columns:
            result_df[col] = pivots[col]
        
        return result_df
    
    @staticmethod
    def get_trading_signals(df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate basic trading signals based on common indicators
        """
        signals = pd.DataFrame(index=df.index)
        
        # RSI signals
        signals['rsi_oversold'] = df['rsi'] < 30
        signals['rsi_overbought'] = df['rsi'] > 70
        
        # MACD signals
        signals['macd_bullish_cross'] = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
        signals['macd_bearish_cross'] = (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))
        
        # Moving Average signals
        signals['ma_bullish_cross'] = (df['sma_20'] > df['sma_50']) & (df['sma_20'].shift(1) <= df['sma_50'].shift(1))
        signals['ma_bearish_cross'] = (df['sma_20'] < df['sma_50']) & (df['sma_20'].shift(1) >= df['sma_50'].shift(1))
        
        # Bollinger Bands signals
        signals['bb_lower_touch'] = df['close'] <= df['bb_lower']
        signals['bb_upper_touch'] = df['close'] >= df['bb_upper']
        
        # Stochastic signals
        signals['stoch_oversold'] = (df['stoch_k'] < 20) & (df['stoch_d'] < 20)
        signals['stoch_overbought'] = (df['stoch_k'] > 80) & (df['stoch_d'] > 80)
        
        # Combined signals
        signals['buy_signal'] = (
            signals['rsi_oversold'] | 
            signals['macd_bullish_cross'] | 
            signals['ma_bullish_cross'] |
            signals['bb_lower_touch'] |
            signals['stoch_oversold']
        )
        
        signals['sell_signal'] = (
            signals['rsi_overbought'] | 
            signals['macd_bearish_cross'] | 
            signals['ma_bearish_cross'] |
            signals['bb_upper_touch'] |
            signals['stoch_overbought']
        )
        
        return signals


# Convenience functions for direct use
def calculate_sma(data, period):
    return IndicatorCalculator.sma(data, period)

def calculate_ema(data, period):
    return IndicatorCalculator.ema(data, period)

def calculate_rsi(data, period=14):
    return IndicatorCalculator.rsi(data, period)

def calculate_macd(data, fast=12, slow=26, signal=9):
    return IndicatorCalculator.macd(data, fast, slow, signal)

def calculate_bollinger_bands(data, period=20, std_dev=2):
    return IndicatorCalculator.bollinger_bands(data, period, std_dev)

def calculate_stochastic(high, low, close, k_period=14, d_period=3):
    return IndicatorCalculator.stochastic(high, low, close, k_period, d_period)

def calculate_atr(high, low, close, period=14):
    return IndicatorCalculator.atr(high, low, close, period)


if __name__ == "__main__":
    # Example usage
    print("Indicator Calculator Examples")
    print("=" * 40)
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    sample_data = pd.DataFrame({
        'open': np.random.uniform(100, 200, 100),
        'high': np.random.uniform(100, 200, 100),
        'low': np.random.uniform(100, 200, 100),
        'close': np.random.uniform(100, 200, 100),
        'volume': np.random.uniform(1000, 10000, 100)
    }, index=dates)
    
    # Ensure high >= low >= close >= open
    sample_data['high'] = sample_data[['open', 'high', 'low', 'close']].max(axis=1)
    sample_data['low'] = sample_data[['open', 'high', 'low', 'close']].min(axis=1)
    
    print("Sample data shape:", sample_data.shape)
    print(sample_data.head())
    
    # Calculate indicators
    indicators = IndicatorCalculator.calculate_all_indicators(sample_data)
    print("\nIndicators calculated:")
    print("Shape:", indicators.shape)
    print("New columns:", set(indicators.columns) - set(sample_data.columns))
    
    # Get signals
    signals = IndicatorCalculator.get_trading_signals(indicators)
    print("\nSignals generated:")
    print("Buy signals count:", signals['buy_signal'].sum())
    print("Sell signals count:", signals['sell_signal'].sum())
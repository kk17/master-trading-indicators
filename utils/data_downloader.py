"""
Data Download Utility for Crypto Trading Indicators
Supports multiple data sources including freqtrade, ccxt, and yfinance
"""

import pandas as pd
import numpy as np
import ccxt
import yfinance as yf
from datetime import datetime, timedelta
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

class DataDownloader:
    """
    Universal data downloader for cryptocurrency market data
    """
    
    def __init__(self, data_dir="../data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Initialize exchanges
        self.binance = ccxt.binance({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',
            }
        })
        
    def download_with_ccxt(self, pair, timeframe, days, exchange='binance'):
        """
        Download data using CCXT
        """
        try:
            if exchange == 'binance':
                exchange_instance = self.binance
            
            # Calculate timeframe in milliseconds
            timeframe_ms = self._timeframe_to_ms(timeframe)
            since = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
            
            # Download OHLCV data
            ohlcv = exchange_instance.fetch_ohlcv(pair, timeframe, since)
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Add typical price and other useful columns
            df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
            df['median_price'] = (df['high'] + df['low']) / 2
            df['weighted_close'] = (df['high'] + df['low'] + 2 * df['close']) / 4
            
            print(f"Downloaded {len(df)} candles of {pair} data from {exchange}")
            return df
            
        except Exception as e:
            print(f"Error downloading {pair} from {exchange}: {e}")
            return None
    
    def download_with_yfinance(self, symbol, period="1y", interval="1d"):
        """
        Download data using Yahoo Finance
        Note: Limited crypto support
        """
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            
            if df.empty:
                print(f"No data found for {symbol}")
                return None
            
            # Standardize column names
            df.columns = [col.lower() for col in df.columns]
            df.index.name = 'timestamp'
            
            # Add typical price
            df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
            df['median_price'] = (df['high'] + df['low']) / 2
            df['weighted_close'] = (df['high'] + df['low'] + 2 * df['close']) / 4
            
            print(f"Downloaded {len(df)} candles of {symbol} data from Yahoo Finance")
            return df
            
        except Exception as e:
            print(f"Error downloading {symbol} from Yahoo Finance: {e}")
            return None
    
    def download_with_freqtrade(self, pair, timeframe, days, config_file=None):
        """
        Download data using Freqtrade
        Note: Requires freqtrade to be installed and configured
        """
        try:
            from freqtrade.data.history import load_data, refresh_data
            from freqtrade.configuration import Configuration
            
            # Create configuration
            config = {
                'exchange': {
                    'name': 'binance',
                    'key': '',
                    'secret': '',
                },
                'pairs': [pair],
                'timeframes': [timeframe],
                'datadir': str(self.data_dir),
                'timerange': f'{days}d ago',
            }
            
            if config_file and os.path.exists(config_file):
                user_config = Configuration.from_file(config_file)
                config.update(user_config)
            
            # Download data
            refresh_data(config, [pair], timeframe)
            
            # Load the downloaded data
            data = load_data(datadir=str(self.data_dir), pairs=[pair], timeframe=timeframe)
            
            if pair in data:
                df = data[pair]
                print(f"Downloaded {len(df)} candles of {pair} data using Freqtrade")
                return df
            else:
                print(f"No data downloaded for {pair}")
                return None
                
        except ImportError:
            print("Freqtrade not installed. Please install it first: pip install freqtrade")
            return None
        except Exception as e:
            print(f"Error downloading {pair} with Freqtrade: {e}")
            return None
    
    def _timeframe_to_ms(self, timeframe):
        """
        Convert timeframe string to milliseconds
        """
        timeframe_map = {
            '1m': 60 * 1000,
            '3m': 3 * 60 * 1000,
            '5m': 5 * 60 * 1000,
            '15m': 15 * 60 * 1000,
            '30m': 30 * 60 * 1000,
            '1h': 60 * 60 * 1000,
            '2h': 2 * 60 * 60 * 1000,
            '4h': 4 * 60 * 60 * 1000,
            '6h': 6 * 60 * 60 * 1000,
            '8h': 8 * 60 * 60 * 1000,
            '12h': 12 * 60 * 60 * 1000,
            '1d': 24 * 60 * 60 * 1000,
            '3d': 3 * 24 * 60 * 60 * 1000,
            '1w': 7 * 24 * 60 * 60 * 1000,
            '1M': 30 * 24 * 60 * 60 * 1000,
        }
        
        return timeframe_map.get(timeframe, 24 * 60 * 60 * 1000)  # Default to 1d
    
    def save_data(self, df, filename):
        """
        Save DataFrame to CSV file
        """
        if df is not None and not df.empty:
            filepath = self.data_dir / filename
            df.to_csv(filepath)
            print(f"Data saved to {filepath}")
            return filepath
        else:
            print("No data to save")
            return None
    
    def load_data(self, filename):
        """
        Load DataFrame from CSV file
        """
        filepath = self.data_dir / filename
        if filepath.exists():
            df = pd.read_csv(filepath, index_col='timestamp', parse_dates=True)
            print(f"Data loaded from {filepath}")
            return df
        else:
            print(f"File not found: {filepath}")
            return None
    
    def get_available_pairs(self, exchange='binance'):
        """
        Get available trading pairs from exchange
        """
        try:
            if exchange == 'binance':
                markets = self.binance.load_markets()
                pairs = [market['symbol'] for market in markets.values() 
                        if market['quote'] == 'USDT' and market['type'] == 'future']
                return sorted(pairs)
        except Exception as e:
            print(f"Error getting pairs from {exchange}: {e}")
            return []
    
    def preprocess_data(self, df):
        """
        Preprocess data for indicator calculation
        """
        if df is None or df.empty:
            return None
        
        # Remove rows with missing values
        df = df.dropna()
        
        # Sort by timestamp
        df = df.sort_index()
        
        # Ensure numeric columns
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove rows with zero or negative prices
        df = df[df['close'] > 0]
        
        return df
    
    def get_market_data(self, pair, timeframe, days, source='ccxt', save=True):
        """
        Get market data from specified source
        """
        print(f"Downloading {pair} data ({timeframe}, {days} days) from {source}...")
        
        if source == 'ccxt':
            df = self.download_with_ccxt(pair, timeframe, days)
        elif source == 'yfinance':
            # Convert crypto pair to yfinance format
            yf_symbol = pair.replace('/', '-') + '=X'
            df = self.download_with_yfinance(yf_symbol, period=f"{days}d", interval=timeframe)
        elif source == 'freqtrade':
            df = self.download_with_freqtrade(pair, timeframe, days)
        else:
            print(f"Unknown data source: {source}")
            return None
        
        if df is not None:
            df = self.preprocess_data(df)
            
            if save:
                filename = f"{pair.replace('/', '_')}_{timeframe}_{days}d.csv"
                self.save_data(df, filename)
        
        return df


def download_crypto_data(pair="BTC/USDT", timeframe="1d", days=365, source='ccxt', data_dir="../data"):
    """
    Convenience function to download crypto data
    """
    downloader = DataDownloader(data_dir)
    return downloader.get_market_data(pair, timeframe, days, source)


def load_crypto_data(filename, data_dir="../data"):
    """
    Convenience function to load crypto data from file
    """
    downloader = DataDownloader(data_dir)
    return downloader.load_data(filename)


if __name__ == "__main__":
    # Example usage
    print("Crypto Data Downloader")
    print("=" * 50)
    
    # Initialize downloader
    downloader = DataDownloader()
    
    # Example 1: Download BTC data using CCXT
    print("\n1. Downloading BTC/USDT data using CCXT...")
    btc_data = downloader.get_market_data("BTC/USDT", "1d", 30, source='ccxt')
    
    if btc_data is not None:
        print(f"BTC data shape: {btc_data.shape}")
        print(f"Columns: {btc_data.columns.tolist()}")
        print(btc_data.head())
    
    # Example 2: Get available pairs
    print("\n2. Getting available trading pairs...")
    pairs = downloader.get_available_pairs()[:10]  # Show first 10
    print(f"Available pairs (first 10): {pairs}")
    
    # Example 3: Download ETH data
    print("\n3. Downloading ETH/USDT data...")
    eth_data = downloader.get_market_data("ETH/USDT", "4h", 7, source='ccxt')
    
    if eth_data is not None:
        print(f"ETH data shape: {eth_data.shape}")
        print(eth_data.tail())
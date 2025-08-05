#!/usr/bin/env python3
"""
Example usage of the crypto trading indicators tutorial repository
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from utils.data_downloader import download_crypto_data
from utils.indicators import IndicatorCalculator
from utils.backtest_engine import BacktestEngine, StrategyGenerator

def main():
    """
    Main example function demonstrating the repository usage
    """
    print("🚀 Master Trading Indicators - Example Usage")
    print("=" * 60)
    
    # 1. Download crypto data
    print("\n1. Downloading Crypto Data...")
    pair = "BTC/USDT"
    timeframe = "1d"
    days = 90
    
    try:
        df = download_crypto_data(pair, timeframe, days)
        if df is not None:
            print(f"✅ Downloaded {len(df)} days of {pair} data")
            print(f"   Date range: {df.index[0].date()} to {df.index[-1].date()}")
            print(f"   Current price: ${df['close'].iloc[-1]:,.2f}")
        else:
            print("❌ Failed to download data")
            return
    except Exception as e:
        print(f"❌ Error downloading data: {e}")
        return
    
    # 2. Calculate indicators
    print("\n2. Calculating Technical Indicators...")
    try:
        # Calculate all indicators
        df_with_indicators = IndicatorCalculator.calculate_all_indicators(df)
        print(f"✅ Calculated {len(df_with_indicators.columns) - len(df.columns)} indicators")
        print(f"   Total columns: {len(df_with_indicators.columns)}")
        
        # Show some key indicators
        key_indicators = ['close', 'sma_20', 'sma_50', 'rsi', 'macd', 'bb_upper', 'bb_lower']
        available_indicators = [col for col in key_indicators if col in df_with_indicators.columns]
        print(f"   Key indicators: {available_indicators}")
        
    except Exception as e:
        print(f"❌ Error calculating indicators: {e}")
        return
    
    # 3. Generate trading signals
    print("\n3. Generating Trading Signals...")
    try:
        signals = IndicatorCalculator.get_trading_signals(df_with_indicators)
        
        # Count signals
        buy_signals = signals['buy_signal'].sum()
        sell_signals = signals['sell_signal'].sum()
        
        print(f"✅ Generated trading signals")
        print(f"   Buy signals: {buy_signals}")
        print(f"   Sell signals: {sell_signals}")
        print(f"   Signal coverage: {((buy_signals + sell_signals) / len(signals) * 100):.1f}% of days")
        
    except Exception as e:
        print(f"❌ Error generating signals: {e}")
        return
    
    # 4. Backtest strategies
    print("\n4. Backtesting Trading Strategies...")
    try:
        engine = BacktestEngine(initial_capital=10000)
        
        # Test multiple strategies
        strategies = {
            'MA_Crossover': StrategyGenerator.ma_crossover_strategy(df_with_indicators),
            'RSI_Mean_Reversion': StrategyGenerator.rsi_strategy(df_with_indicators),
            'MACD_Crossover': StrategyGenerator.macd_strategy(df_with_indicators),
            'Bollinger_Bands': StrategyGenerator.bollinger_bands_strategy(df_with_indicators),
        }
        
        for name, signals in strategies.items():
            result = engine.run_backtest(df_with_indicators, signals[['signal']], name)
            if result:
                metrics = result['metrics']
                print(f"   {name}: {metrics['total_return']:.1%} return, {metrics['sharpe_ratio']:.2f} Sharpe")
        
        # Compare strategies
        comparison = engine.compare_strategies(list(strategies.keys()))
        print(f"\n📊 Strategy Comparison:")
        print(comparison[['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']].round(3))
        
    except Exception as e:
        print(f"❌ Error backtesting: {e}")
        return
    
    # 5. Basic visualization
    print("\n5. Creating Visualization...")
    try:
        plt.figure(figsize=(15, 10))
        
        # Price and moving averages
        plt.subplot(2, 2, 1)
        plt.plot(df_with_indicators.index, df_with_indicators['close'], label='Price', alpha=0.7)
        plt.plot(df_with_indicators.index, df_with_indicators['sma_20'], label='SMA(20)', alpha=0.8)
        plt.plot(df_with_indicators.index, df_with_indicators['sma_50'], label='SMA(50)', alpha=0.8)
        plt.title(f'{pair} - Price with Moving Averages')
        plt.xlabel('Date')
        plt.ylabel('Price (USDT)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # RSI
        plt.subplot(2, 2, 2)
        plt.plot(df_with_indicators.index, df_with_indicators['rsi'], label='RSI', color='purple')
        plt.axhline(y=70, color='red', linestyle='--', alpha=0.5, label='Overbought')
        plt.axhline(y=30, color='green', linestyle='--', alpha=0.5, label='Oversold')
        plt.title('Relative Strength Index (RSI)')
        plt.xlabel('Date')
        plt.ylabel('RSI')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # MACD
        plt.subplot(2, 2, 3)
        plt.plot(df_with_indicators.index, df_with_indicators['macd'], label='MACD', alpha=0.8)
        plt.plot(df_with_indicators.index, df_with_indicators['macd_signal'], label='Signal', alpha=0.8)
        plt.bar(df_with_indicators.index, df_with_indicators['macd_histogram'], 
                label='Histogram', alpha=0.6, width=1)
        plt.title('MACD')
        plt.xlabel('Date')
        plt.ylabel('MACD')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Bollinger Bands
        plt.subplot(2, 2, 4)
        plt.plot(df_with_indicators.index, df_with_indicators['close'], label='Price', alpha=0.7)
        plt.plot(df_with_indicators.index, df_with_indicators['bb_upper'], label='Upper BB', alpha=0.8, color='red')
        plt.plot(df_with_indicators.index, df_with_indicators['bb_middle'], label='Middle BB', alpha=0.8, color='blue')
        plt.plot(df_with_indicators.index, df_with_indicators['bb_lower'], label='Lower BB', alpha=0.8, color='green')
        plt.fill_between(df_with_indicators.index, df_with_indicators['bb_upper'], 
                        df_with_indicators['bb_lower'], alpha=0.1, color='gray')
        plt.title('Bollinger Bands')
        plt.xlabel('Date')
        plt.ylabel('Price (USDT)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('crypto_indicators_example.png', dpi=150, bbox_inches='tight')
        print(f"✅ Visualization saved as 'crypto_indicators_example.png'")
        
    except Exception as e:
        print(f"❌ Error creating visualization: {e}")
    
    # 6. Show recent market analysis
    print("\n6. Recent Market Analysis...")
    try:
        recent_data = df_with_indicators.tail(5)
        current_price = recent_data['close'].iloc[-1]
        sma_20 = recent_data['sma_20'].iloc[-1]
        sma_50 = recent_data['sma_50'].iloc[-1]
        rsi = recent_data['rsi'].iloc[-1]
        
        print(f"   Current Price: ${current_price:,.2f}")
        print(f"   SMA(20): ${sma_20:,.2f}")
        print(f"   SMA(50): ${sma_50:,.2f}")
        print(f"   RSI: {rsi:.1f}")
        
        # Trend analysis
        if current_price > sma_20 > sma_50:
            trend = "Strong Uptrend 📈"
        elif current_price < sma_20 < sma_50:
            trend = "Strong Downtrend 📉"
        elif current_price > sma_20 and sma_20 < sma_50:
            trend = "Potential Reversal ⚠️"
        else:
            trend = "Range-bound or Weak Trend ➡️"
        
        print(f"   Trend Analysis: {trend}")
        
        # RSI analysis
        if rsi > 70:
            rsi_status = "Overbought"
        elif rsi < 30:
            rsi_status = "Oversold"
        else:
            rsi_status = "Neutral"
        
        print(f"   RSI Status: {rsi_status}")
        
    except Exception as e:
        print(f"❌ Error in market analysis: {e}")
    
    print("\n✅ Example completed successfully!")
    print("\n📚 Next Steps:")
    print("   1. Open Jupyter: jupyter notebook")
    print("   2. Start with: notebooks/trend/01_SMA_Complete_Tutorial.ipynb")
    print("   3. Explore other indicators in their respective folders")
    print("   4. Modify strategies and test your own ideas")
    
    print(f"\n📁 Files created:")
    print(f"   - crypto_indicators_example.png (visualization)")
    if os.path.exists('data'):
        data_files = [f for f in os.listdir('data') if f.endswith('.csv')]
        if data_files:
            print(f"   - data/{data_files[-1]} (market data)")

if __name__ == "__main__":
    main()
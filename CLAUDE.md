# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a comprehensive educational repository for learning cryptocurrency trading indicators using Python, designed for programmers with limited finance background. The project combines theoretical learning with practical implementation using real market data.

## Development Commands

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Start Jupyter notebooks for interactive learning
jupyter notebook

# Run the example usage script
python example_usage.py
```

### Data Management
```bash
# Download crypto data (uses CCXT by default)
python -c "from utils.data_downloader import download_crypto_data; download_crypto_data('BTC/USDT', '1d', 365)"

# Test indicator calculations
python utils/indicators.py

# Test backtesting engine
python utils/backtest_engine.py
```

## Architecture Overview

### Core Components

1. **Data Layer** (`utils/data_downloader.py`):
   - Multi-source data downloading (CCXT, Yahoo Finance, Freqtrade)
   - Data preprocessing and validation
   - Automatic data caching and management

2. **Indicator Engine** (`utils/indicators.py`):
   - 30+ technical indicators with manual implementations
   - Support for TA-Lib and pandas-ta libraries
   - Signal generation and pattern detection

3. **Backtesting Framework** (`utils/backtest_engine.py`):
   - Strategy backtesting with performance metrics
   - Risk analysis and drawdown calculations
   - Strategy comparison and reporting

4. **Educational Content** (`notebooks/`):
   - Structured tutorials by indicator category
   - Step-by-step implementations from scratch
   - TradingView Pine Script examples
   - Platform integration guides

### Tutorial Structure

Each tutorial follows a comprehensive 16-section format:
- Theory and mathematical formulas
- Manual Python implementation
- Library comparison (TA-Lib, pandas-ta)
- TradingView Pine Script integration
- Binance platform usage
- 2-3 practical trading strategies
- Risk management techniques
- Exercises and solutions

### Data Flow

```
Raw Data → Preprocessing → Indicator Calculation → Signal Generation → Backtesting → Analysis
```

## Key Development Patterns

### Data Handling
- Always use `utils.data_downloader.download_crypto_data()` for data fetching
- Data is automatically preprocessed with typical price, median price calculations
- Supports multiple timeframes: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M

### Indicator Implementation
- Manual implementations prioritize educational clarity over performance
- Library integrations use TA-Lib for production scenarios
- All indicators return pandas Series for consistency
- Error handling for edge cases (division by zero, insufficient data)

### Strategy Development
- Strategies return DataFrame with 'signal' column (-1, 0, 1)
- Backtesting engine expects standardized signal format
- Performance metrics include Sharpe ratio, max drawdown, win rate
- Risk management built into strategy evaluation

## File Organization

```
master-trading-indicators/
├── utils/                    # Core utilities
│   ├── data_downloader.py   # Multi-source data fetching
│   ├── indicators.py        # Technical indicators library
│   └── backtest_engine.py   # Strategy testing framework
├── notebooks/               # Educational tutorials by category
│   ├── trend/              # Trend indicators (SMA, EMA, MACD, etc.)
│   ├── momentum/           # Momentum indicators (RSI, Stochastic, etc.)
│   ├── volatility/         # Volatility indicators (Bollinger Bands, ATR)
│   ├── volume/             # Volume indicators (OBV, VWAP, Volume Profile)
│   ├── support_resistance/ # S&R indicators (Fibonacci, Pivot Points)
│   └── mean_reversion/    # Mean reversion strategies
├── data/                   # Cached market data (auto-generated)
├── crypto-trading-indicators/  # Legacy structure (being migrated)
└── example_usage.py        # Demonstration script
```

## Dependencies and Libraries

### Core Libraries
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **matplotlib/seaborn** - Visualization
- **plotly** - Interactive charts

### Trading Libraries
- **TA-Lib** - Professional technical analysis
- **pandas-ta** - Pandas technical analysis
- **freqtrade** - Crypto trading framework
- **ccxt** - Multi-exchange API

### Data Sources
- **CCXT** - Cryptocurrency exchange data
- **Yahoo Finance** - Market data (limited crypto)
- **Binance API** - Direct exchange integration

## Common Development Tasks

### Adding New Indicators
1. Implement in `utils/indicators.py` with manual calculation
2. Add library comparison examples
3. Create tutorial notebook following 16-section format
4. Include TradingView Pine Script equivalent
5. Add practical trading strategies

### Testing Strategies
1. Use `BacktestEngine` class for systematic testing
2. Compare with baseline strategies
3. Analyze risk metrics (Sharpe, drawdown, win rate)
4. Generate reports with recommendations

### Data Validation
- Check for missing values before indicator calculation
- Ensure sufficient data length for indicator periods
- Validate price relationships (high ≥ low ≥ close ≥ open)
- Handle timezone consistency across data sources

## Platform Integration

### TradingView Integration
- Each tutorial includes Pine Script examples
- Shows how to add indicators to TradingView charts
- Compares Python vs Pine Script implementations

### Binance Integration
- Demonstrates adding indicators to Binance trading interface
- Shows real-world application of indicators
- Includes platform-specific considerations

### Freqtrade Integration
- Compatible with Freqtrade strategy format
- Shows how to convert educational examples to live trading
- Includes backtesting with Freqtrade framework

## Best Practices

### Code Quality
- Prefer clear, educational implementations over optimized code
- Include comprehensive docstrings with mathematical formulas
- Handle edge cases gracefully (insufficient data, division by zero)
- Maintain consistency with existing code patterns

### Educational Focus
- Explain the "why" behind each indicator
- Include multiple implementation approaches
- Provide practical trading context
- Emphasize risk management and proper usage

### Performance Considerations
- Use vectorized operations with pandas/numpy
- Cache expensive calculations where appropriate
- Consider memory usage with large datasets
- Profile performance bottlenecks in backtesting

## Testing and Validation

### Data Quality Tests
- Validate OHLC data relationships
- Check for missing or anomalous values
- Ensure timestamp consistency
- Test across different timeframes

### Indicator Accuracy
- Compare manual implementations with library results
- Test edge cases (empty data, single values, extreme values)
- Validate against known examples and documentation
- Cross-reference with TradingView calculations

### Strategy Validation
- Test with multiple market conditions
- Validate signal generation logic
- Check performance metric calculations
- Compare with benchmark strategies
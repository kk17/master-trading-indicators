"""
Strategy Backtesting Utilities
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from datetime import datetime


class BacktestEngine:
    """
    Comprehensive backtesting engine for trading strategies
    """
    
    def __init__(self, initial_capital: float = 10000):
        self.initial_capital = initial_capital
        self.results = {}
        
    def run_backtest(self, data: pd.DataFrame, signals: pd.DataFrame, 
                    strategy_name: str = "Strategy") -> Dict:
        """
        Run a backtest on the given signals
        """
        # Create results DataFrame
        results = data.copy()
        results['signal'] = signals['signal']
        results['position'] = results['signal'].shift(1)
        results['returns'] = results['close'].pct_change()
        results['strategy_returns'] = results['position'] * results['returns']
        
        # Calculate cumulative returns
        results['cumulative_returns'] = (1 + results['strategy_returns']).cumprod()
        results['portfolio_value'] = self.initial_capital * results['cumulative_returns']
        
        # Calculate drawdown
        results['peak'] = results['portfolio_value'].cummax()
        results['drawdown'] = (results['portfolio_value'] - results['peak']) / results['peak']
        
        # Store results
        self.results[strategy_name] = {
            'data': results,
            'metrics': self._calculate_metrics(results)
        }
        
        return self.results[strategy_name]
    
    def _calculate_metrics(self, results: pd.DataFrame) -> Dict:
        """
        Calculate performance metrics
        """
        total_return = results['portfolio_value'].iloc[-1] / self.initial_capital - 1
        
        # Annualized return
        days_held = (results.index[-1] - results.index[0]).days
        annualized_return = (1 + total_return) ** (365 / days_held) - 1 if days_held > 0 else 0
        
        # Volatility (annualized)
        volatility = results['strategy_returns'].std() * np.sqrt(365)
        
        # Sharpe ratio
        risk_free_rate = 0.02  # Assume 2% risk-free rate
        sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0
        
        # Maximum drawdown
        max_drawdown = results['drawdown'].min()
        
        # Win rate
        winning_trades = results[results['strategy_returns'] > 0]['strategy_returns'].count()
        total_trades = results[results['strategy_returns'] != 0]['strategy_returns'].count()
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Profit factor
        gross_profit = results[results['strategy_returns'] > 0]['strategy_returns'].sum()
        gross_loss = abs(results[results['strategy_returns'] < 0]['strategy_returns'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': total_trades
        }
    
    def compare_strategies(self, strategy_names: List[str]) -> pd.DataFrame:
        """
        Compare multiple strategies
        """
        comparison_data = []
        
        for name in strategy_names:
            if name in self.results:
                metrics = self.results[name]['metrics']
                metrics['strategy'] = name
                comparison_data.append(metrics)
        
        return pd.DataFrame(comparison_data).set_index('strategy')
    
    def plot_equity_curve(self, strategy_names: List[str] = None, 
                         figsize: Tuple[int, int] = (12, 6)):
        """
        Plot equity curves for strategies
        """
        if strategy_names is None:
            strategy_names = list(self.results.keys())
        
        plt.figure(figsize=figsize)
        
        for name in strategy_names:
            if name in self.results:
                data = self.results[name]['data']
                plt.plot(data.index, data['portfolio_value'], 
                        label=name, linewidth=2)
        
        plt.axhline(y=self.initial_capital, color='red', linestyle='--', 
                   label='Initial Capital')
        plt.title('Strategy Comparison - Equity Curves')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_drawdown(self, strategy_names: List[str] = None, 
                     figsize: Tuple[int, int] = (12, 6)):
        """
        Plot drawdown curves for strategies
        """
        if strategy_names is None:
            strategy_names = list(self.results.keys())
        
        plt.figure(figsize=figsize)
        
        for name in strategy_names:
            if name in self.results:
                data = self.results[name]['data']
                plt.fill_between(data.index, data['drawdown'] * 100, 0, 
                               alpha=0.3, label=name)
        
        plt.title('Strategy Comparison - Drawdown')
        plt.xlabel('Date')
        plt.ylabel('Drawdown (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_returns_distribution(self, strategy_name: str, 
                                 figsize: Tuple[int, int] = (12, 6)):
        """
        Plot returns distribution for a strategy
        """
        if strategy_name not in self.results:
            print(f"Strategy {strategy_name} not found")
            return
        
        data = self.results[strategy_name]['data']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Histogram of returns
        ax1.hist(data['strategy_returns'].dropna() * 100, bins=50, alpha=0.7, edgecolor='black')
        ax1.axvline(x=0, color='red', linestyle='--', label='Zero')
        ax1.set_title('Returns Distribution')
        ax1.set_xlabel('Daily Returns (%)')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(data['strategy_returns'].dropna(), dist='norm', plot=ax2)
        ax2.set_title('Q-Q Plot vs Normal Distribution')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def generate_report(self, strategy_name: str) -> str:
        """
        Generate a detailed report for a strategy
        """
        if strategy_name not in self.results:
            return f"Strategy {strategy_name} not found"
        
        metrics = self.results[strategy_name]['metrics']
        
        report = f"""
{'='*60}
BACKTEST REPORT: {strategy_name}
{'='*60}

PERFORMANCE METRICS:
- Total Return: {metrics['total_return']:.2%}
- Annualized Return: {metrics['annualized_return']:.2%}
- Volatility: {metrics['volatility']:.2%}
- Sharpe Ratio: {metrics['sharpe_ratio']:.2f}
- Maximum Drawdown: {metrics['max_drawdown']:.2%}
- Win Rate: {metrics['win_rate']:.2%}
- Profit Factor: {metrics['profit_factor']:.2f}
- Total Trades: {metrics['total_trades']}

RISK METRICS:
- Calmar Ratio: {metrics['annualized_return'] / abs(metrics['max_drawdown']):.2f}
- Sortino Ratio: {self._calculate_sortino_ratio(self.results[strategy_name]['data']):.2f}
- Win/Loss Ratio: {self._calculate_win_loss_ratio(self.results[strategy_name]['data']):.2f}

TRADE ANALYSIS:
- Average Win: {self._calculate_average_win(self.results[strategy_name]['data']):.2%}
- Average Loss: {self._calculate_average_loss(self.results[strategy_name]['data']):.2%}
- Largest Win: {self._calculate_largest_win(self.results[strategy_name]['data']):.2%}
- Largest Loss: {self._calculate_largest_loss(self.results[strategy_name]['data']):.2%}

RECOMMENDATIONS:
{self._generate_recommendations(metrics)}
{'='*60}
"""
        return report
    
    def _calculate_sortino_ratio(self, data: pd.DataFrame) -> float:
        """
        Calculate Sortino ratio
        """
        returns = data['strategy_returns'].dropna()
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(365)
        
        annualized_return = (1 + returns.sum()) ** (365 / len(returns)) - 1
        risk_free_rate = 0.02
        
        return (annualized_return - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
    
    def _calculate_win_loss_ratio(self, data: pd.DataFrame) -> float:
        """
        Calculate win/loss ratio
        """
        wins = data[data['strategy_returns'] > 0]['strategy_returns']
        losses = data[data['strategy_returns'] < 0]['strategy_returns']
        
        if len(wins) == 0 or len(losses) == 0:
            return 0
        
        return wins.mean() / abs(losses.mean())
    
    def _calculate_average_win(self, data: pd.DataFrame) -> float:
        """
        Calculate average win
        """
        wins = data[data['strategy_returns'] > 0]['strategy_returns']
        return wins.mean() if len(wins) > 0 else 0
    
    def _calculate_average_loss(self, data: pd.DataFrame) -> float:
        """
        Calculate average loss
        """
        losses = data[data['strategy_returns'] < 0]['strategy_returns']
        return losses.mean() if len(losses) > 0 else 0
    
    def _calculate_largest_win(self, data: pd.DataFrame) -> float:
        """
        Calculate largest win
        """
        return data['strategy_returns'].max()
    
    def _calculate_largest_loss(self, data: pd.DataFrame) -> float:
        """
        Calculate largest loss
        """
        return data['strategy_returns'].min()
    
    def _generate_recommendations(self, metrics: Dict) -> str:
        """
        Generate recommendations based on metrics
        """
        recommendations = []
        
        if metrics['sharpe_ratio'] < 1.0:
            recommendations.append("- Low Sharpe ratio: Consider adding risk management or improving strategy")
        
        if metrics['max_drawdown'] < -0.2:
            recommendations.append("- High drawdown: Consider implementing stop-loss or position sizing")
        
        if metrics['win_rate'] < 0.5:
            recommendations.append("- Low win rate: Consider refining entry/exit criteria")
        
        if metrics['profit_factor'] < 1.5:
            recommendations.append("- Low profit factor: Consider improving risk-reward ratio")
        
        if len(recommendations) == 0:
            recommendations.append("- Strategy shows good performance metrics")
        
        return '\n'.join(recommendations)


class StrategyGenerator:
    """
    Generate trading strategies based on indicators
    """
    
    @staticmethod
    def ma_crossover_strategy(data: pd.DataFrame, fast_period: int = 20, 
                            slow_period: int = 50) -> pd.DataFrame:
        """
        Simple moving average crossover strategy
        """
        signals = pd.DataFrame(index=data.index)
        
        # Calculate moving averages
        signals['ma_fast'] = data['close'].rolling(window=fast_period).mean()
        signals['ma_slow'] = data['close'].rolling(window=slow_period).mean()
        
        # Generate signals
        signals['signal'] = 0
        signals.loc[signals['ma_fast'] > signals['ma_slow'], 'signal'] = 1
        signals.loc[signals['ma_fast'] < signals['ma_slow'], 'signal'] = -1
        
        return signals
    
    @staticmethod
    def rsi_strategy(data: pd.DataFrame, oversold: int = 30, overbought: int = 70) -> pd.DataFrame:
        """
        RSI mean reversion strategy
        """
        signals = pd.DataFrame(index=data.index)
        
        # Calculate RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        signals['rsi'] = 100 - (100 / (1 + rs))
        
        # Generate signals
        signals['signal'] = 0
        signals.loc[signals['rsi'] < oversold, 'signal'] = 1  # Buy when oversold
        signals.loc[signals['rsi'] > overbought, 'signal'] = -1  # Sell when overbought
        
        return signals
    
    @staticmethod
    def macd_strategy(data: pd.DataFrame, fast: int = 12, slow: int = 26, 
                     signal: int = 9) -> pd.DataFrame:
        """
        MACD crossover strategy
        """
        signals = pd.DataFrame(index=data.index)
        
        # Calculate MACD
        ema_fast = data['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = data['close'].ewm(span=slow, adjust=False).mean()
        signals['macd'] = ema_fast - ema_slow
        signals['macd_signal'] = signals['macd'].ewm(span=signal, adjust=False).mean()
        signals['macd_histogram'] = signals['macd'] - signals['macd_signal']
        
        # Generate signals
        signals['signal'] = 0
        signals.loc[signals['macd'] > signals['macd_signal'], 'signal'] = 1
        signals.loc[signals['macd'] < signals['macd_signal'], 'signal'] = -1
        
        return signals
    
    @staticmethod
    def bollinger_bands_strategy(data: pd.DataFrame, period: int = 20, 
                               std_dev: float = 2.0) -> pd.DataFrame:
        """
        Bollinger Bands mean reversion strategy
        """
        signals = pd.DataFrame(index=data.index)
        
        # Calculate Bollinger Bands
        signals['bb_middle'] = data['close'].rolling(window=period).mean()
        signals['bb_std'] = data['close'].rolling(window=period).std()
        signals['bb_upper'] = signals['bb_middle'] + (signals['bb_std'] * std_dev)
        signals['bb_lower'] = signals['bb_middle'] - (signals['bb_std'] * std_dev)
        
        # Generate signals
        signals['signal'] = 0
        signals.loc[data['close'] <= signals['bb_lower'], 'signal'] = 1  # Buy when below lower band
        signals.loc[data['close'] >= signals['bb_upper'], 'signal'] = -1  # Sell when above upper band
        
        return signals
    
    @staticmethod
    def combined_strategy(data: pd.DataFrame, ma_fast: int = 20, ma_slow: int = 50,
                         rsi_oversold: int = 30, rsi_overbought: int = 70) -> pd.DataFrame:
        """
        Combined strategy using multiple indicators
        """
        # Get individual signals
        ma_signals = StrategyGenerator.ma_crossover_strategy(data, ma_fast, ma_slow)
        rsi_signals = StrategyGenerator.rsi_strategy(data, rsi_oversold, rsi_overbought)
        
        # Combine signals
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0
        
        # Buy signal: MA crossover AND RSI oversold
        buy_condition = (ma_signals['signal'] == 1) & (rsi_signals['signal'] == 1)
        
        # Sell signal: MA crossover AND RSI overbought
        sell_condition = (ma_signals['signal'] == -1) & (rsi_signals['signal'] == -1)
        
        signals.loc[buy_condition, 'signal'] = 1
        signals.loc[sell_condition, 'signal'] = -1
        
        return signals


if __name__ == "__main__":
    # Example usage
    print("Backtesting Engine Example")
    print("=" * 40)
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=252, freq='D')
    sample_data = pd.DataFrame({
        'open': np.random.uniform(100, 200, 252),
        'high': np.random.uniform(100, 200, 252),
        'low': np.random.uniform(100, 200, 252),
        'close': np.random.uniform(100, 200, 252),
        'volume': np.random.uniform(1000, 10000, 252)
    }, index=dates)
    
    # Generate realistic price data
    sample_data['close'] = 150 + np.cumsum(np.random.normal(0, 1, 252))
    sample_data['high'] = sample_data['close'] * np.random.uniform(1.0, 1.05, 252)
    sample_data['low'] = sample_data['close'] * np.random.uniform(0.95, 1.0, 252)
    sample_data['open'] = sample_data['close'].shift(1) * np.random.uniform(0.99, 1.01, 252)
    sample_data.iloc[0, sample_data.columns.get_loc('open')] = sample_data.iloc[0]['close']
    
    # Initialize backtest engine
    engine = BacktestEngine(initial_capital=10000)
    
    # Test different strategies
    strategies = {
        'MA_Crossover': StrategyGenerator.ma_crossover_strategy(sample_data),
        'RSI': StrategyGenerator.rsi_strategy(sample_data),
        'MACD': StrategyGenerator.macd_strategy(sample_data),
        'Bollinger_Bands': StrategyGenerator.bollinger_bands_strategy(sample_data),
        'Combined': StrategyGenerator.combined_strategy(sample_data)
    }
    
    # Run backtests
    for name, signals in strategies.items():
        engine.run_backtest(sample_data, signals, name)
    
    # Compare strategies
    comparison = engine.compare_strategies(list(strategies.keys()))
    print("\nStrategy Comparison:")
    print(comparison)
    
    # Plot equity curves
    engine.plot_equity_curve()
    
    # Generate detailed report for best strategy
    best_strategy = comparison['total_return'].idxmax()
    print(engine.generate_report(best_strategy))
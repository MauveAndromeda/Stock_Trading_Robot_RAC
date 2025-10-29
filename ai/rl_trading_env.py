"""
强化学习交易环境
基于Gymnasium实现股票交易环境
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from loguru import logger


class StockTradingEnv(gym.Env):
    """
    股票交易强化学习环境

    状态空间：价格历史、技术指标、Agent情绪、持仓信息
    动作空间：买入比例（-1到1，负数卖出，正数买入）
    奖励：收益率 - 波动惩罚 - 交易成本
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        initial_balance: float = 100000,
        commission_rate: float = 0.0003,
        lookback_window: int = 60,
        max_steps: Optional[int] = None
    ):
        """
        初始化交易环境

        Args:
            df: 历史数据DataFrame，必须包含OHLCV和技术指标
            initial_balance: 初始资金
            commission_rate: 交易佣金率
            lookback_window: 回看窗口大小
            max_steps: 最大步数
        """
        super().__init__()

        self.df = df
        self.initial_balance = initial_balance
        self.commission_rate = commission_rate
        self.lookback_window = lookback_window
        self.max_steps = max_steps or len(df) - lookback_window

        # 定义状态空间
        # [价格历史(60), 技术指标(10), 持仓信息(3), Agent情绪(2)]
        self.state_dim = lookback_window + 10 + 3 + 2
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_dim,),
            dtype=np.float32
        )

        # 定义动作空间
        # 连续动作：-1(全卖) 到 +1(全买)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )

        # 初始化状态
        self.reset()

        logger.info(f"StockTradingEnv initialized: {len(df)} timesteps")

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """重置环境"""
        super().reset(seed=seed)

        # 重置状态
        self.current_step = self.lookback_window
        self.balance = self.initial_balance
        self.position = 0  # 持仓数量
        self.entry_price = 0  # 入场价格
        self.total_trades = 0
        self.total_profit = 0

        # 历史记录
        self.portfolio_values = [self.initial_balance]
        self.trades = []

        state = self._get_state()
        info = self._get_info()

        return state, info

    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        执行动作

        Args:
            action: 动作值 [-1, 1]

        Returns:
            state, reward, terminated, truncated, info
        """
        # 获取当前价格
        current_price = self.df.iloc[self.current_step]['close']

        # 计算交易数量
        action_value = float(action[0])
        trade_amount = self._calculate_trade_amount(action_value, current_price)

        # 执行交易
        if trade_amount != 0:
            self._execute_trade(trade_amount, current_price)

        # 更新步数
        self.current_step += 1

        # 计算奖励
        reward = self._calculate_reward()

        # 检查是否结束
        terminated = self.current_step >= len(self.df) - 1
        truncated = (self.max_steps is not None and
                    self.current_step - self.lookback_window >= self.max_steps)

        # 获取新状态
        if not (terminated or truncated):
            state = self._get_state()
        else:
            state = self._get_state()  # 最终状态

        info = self._get_info()

        # 记录组合价值
        portfolio_value = self.balance + self.position * current_price
        self.portfolio_values.append(portfolio_value)

        return state, reward, terminated, truncated, info

    def _get_state(self) -> np.ndarray:
        """获取当前状态"""
        state_features = []

        # 1. 价格历史（归一化）
        prices = self.df.iloc[
            self.current_step - self.lookback_window:self.current_step
        ]['close'].values

        # 归一化价格（使用收益率）
        if len(prices) > 1:
            returns = np.diff(prices) / prices[:-1]
            returns = np.append(returns, 0)  # 补齐长度
        else:
            returns = np.zeros(self.lookback_window)

        state_features.extend(returns)

        # 2. 技术指标
        current_data = self.df.iloc[self.current_step]
        tech_indicators = [
            current_data.get('rsi', 50) / 100,  # 归一化到0-1
            current_data.get('macd', 0) / 100,
            (current_data.get('ma5', 0) - current_data['close']) / current_data['close'],
            (current_data.get('ma20', 0) - current_data['close']) / current_data['close'],
            (current_data.get('ma60', 0) - current_data['close']) / current_data['close'],
            current_data.get('volume_ratio', 1),
            current_data.get('turnover_rate', 1) / 10,
            current_data.get('volatility', 0.02) * 10,
            current_data.get('bb_upper', current_data['close']) / current_data['close'] - 1,
            current_data.get('bb_lower', current_data['close']) / current_data['close'] - 1,
        ]
        state_features.extend(tech_indicators)

        # 3. 持仓信息
        current_price = current_data['close']
        position_ratio = self.position * current_price / self.initial_balance
        unrealized_pnl = (
            (current_price - self.entry_price) / self.entry_price
            if self.entry_price > 0 else 0
        )
        cash_ratio = self.balance / self.initial_balance

        state_features.extend([position_ratio, unrealized_pnl, cash_ratio])

        # 4. Agent情绪（模拟）
        # 实际使用时，这里应该从Agent系统获取
        retail_sentiment = current_data.get('market_sentiment', 0.5)
        inst_sentiment = 0.5  # 占位

        state_features.extend([retail_sentiment, inst_sentiment])

        return np.array(state_features, dtype=np.float32)

    def _calculate_trade_amount(
        self,
        action: float,
        current_price: float
    ) -> float:
        """
        计算交易数量

        Args:
            action: 动作值 [-1, 1]
            current_price: 当前价格

        Returns:
            交易数量（正数买入，负数卖出）
        """
        if action > 0:
            # 买入
            max_buyable = self.balance / current_price * (1 + self.commission_rate)
            trade_amount = max_buyable * action
        elif action < 0:
            # 卖出
            trade_amount = self.position * action
        else:
            trade_amount = 0

        return trade_amount

    def _execute_trade(self, amount: float, price: float):
        """执行交易"""
        if amount > 0:
            # 买入
            cost = amount * price * (1 + self.commission_rate)
            if cost <= self.balance:
                self.balance -= cost
                self.position += amount
                self.entry_price = price  # 简化：使用最新价格
                self.total_trades += 1

                self.trades.append({
                    'step': self.current_step,
                    'action': 'buy',
                    'amount': amount,
                    'price': price,
                    'cost': cost
                })

        elif amount < 0:
            # 卖出
            sell_amount = min(-amount, self.position)
            if sell_amount > 0:
                proceeds = sell_amount * price * (1 - self.commission_rate)
                self.balance += proceeds
                self.position -= sell_amount

                profit = (price - self.entry_price) * sell_amount
                self.total_profit += profit
                self.total_trades += 1

                self.trades.append({
                    'step': self.current_step,
                    'action': 'sell',
                    'amount': sell_amount,
                    'price': price,
                    'proceeds': proceeds,
                    'profit': profit
                })

                if self.position == 0:
                    self.entry_price = 0

    def _calculate_reward(self) -> float:
        """
        计算奖励

        奖励设计：
        1. 收益率（主要）
        2. 波动率惩罚
        3. 交易成本惩罚
        4. 夏普比率奖励
        """
        if len(self.portfolio_values) < 2:
            return 0.0

        # 1. 当前收益率
        current_value = self.portfolio_values[-1]
        previous_value = self.portfolio_values[-2]
        return_rate = (current_value - previous_value) / previous_value

        # 2. 波动率惩罚
        if len(self.portfolio_values) >= 20:
            recent_returns = np.diff(self.portfolio_values[-20:]) / self.portfolio_values[-21:-1]
            volatility = np.std(recent_returns)
            volatility_penalty = -volatility * 2
        else:
            volatility_penalty = 0

        # 3. 夏普比率奖励（如果有足够历史）
        if len(self.portfolio_values) >= 30:
            returns = np.diff(self.portfolio_values[-30:]) / self.portfolio_values[-31:-1]
            sharpe = np.mean(returns) / (np.std(returns) + 1e-9) * np.sqrt(252)
            sharpe_reward = sharpe * 0.1
        else:
            sharpe_reward = 0

        # 综合奖励
        reward = (
            return_rate * 100 +  # 收益率（放大）
            volatility_penalty +   # 波动惩罚
            sharpe_reward         # 夏普奖励
        )

        return float(reward)

    def _get_info(self) -> Dict[str, Any]:
        """获取额外信息"""
        current_price = self.df.iloc[min(self.current_step, len(self.df)-1)]['close']
        portfolio_value = self.balance + self.position * current_price

        return {
            'step': self.current_step,
            'balance': self.balance,
            'position': self.position,
            'portfolio_value': portfolio_value,
            'total_return': (portfolio_value - self.initial_balance) / self.initial_balance,
            'total_trades': self.total_trades,
            'total_profit': self.total_profit
        }

    def render(self):
        """渲染环境（用于可视化）"""
        info = self._get_info()
        print(f"Step: {info['step']}, "
              f"Portfolio: ${info['portfolio_value']:.2f}, "
              f"Return: {info['total_return']:.2%}, "
              f"Trades: {info['total_trades']}")


class MultiStockTradingEnv(StockTradingEnv):
    """
    多股票交易环境
    可以同时交易多只股票
    """

    def __init__(
        self,
        stock_dfs: Dict[str, pd.DataFrame],
        initial_balance: float = 100000,
        commission_rate: float = 0.0003
    ):
        """
        Args:
            stock_dfs: {stock_code: DataFrame}
        """
        self.stock_dfs = stock_dfs
        self.num_stocks = len(stock_dfs)

        # 使用第一只股票初始化
        first_df = list(stock_dfs.values())[0]
        super().__init__(first_df, initial_balance, commission_rate)

        # 重新定义动作空间（每只股票一个动作）
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_stocks,),
            dtype=np.float32
        )

        logger.info(f"MultiStockTradingEnv initialized: {self.num_stocks} stocks")

    # 需要重写相关方法以支持多股票...
    # 这里简化实现，实际使用时需要完整实现

"""
风险管理器
"""
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from loguru import logger
from .executor import Order, OrderSide, Trade


class RiskManager:
    """风险管理器"""

    def __init__(
        self,
        initial_capital: float = 1000000,
        max_total_position: float = 0.8,
        max_single_position: float = 0.1,
        min_cash_reserve: float = 0.2,
        stop_loss_pct: float = 0.05,
        stop_profit_pct: float = 0.15,
        max_daily_trades: int = 20,
        max_daily_loss_pct: float = 0.03,
        max_drawdown_pct: float = 0.10
    ):
        """
        初始化

        Args:
            initial_capital: 初始资金
            max_total_position: 最大总仓位
            max_single_position: 单只股票最大仓位
            min_cash_reserve: 最小现金储备
            stop_loss_pct: 止损比例
            stop_profit_pct: 止盈比例
            max_daily_trades: 单日最大交易次数
            max_daily_loss_pct: 单日最大亏损比例
            max_drawdown_pct: 最大回撤比例
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital

        # 仓位控制
        self.max_total_position = max_total_position
        self.max_single_position = max_single_position
        self.min_cash_reserve = min_cash_reserve

        # 止损止盈
        self.stop_loss_pct = stop_loss_pct
        self.stop_profit_pct = stop_profit_pct

        # 交易限制
        self.max_daily_trades = max_daily_trades
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_drawdown_pct = max_drawdown_pct

        # 统计
        self.daily_trades_count = 0
        self.daily_pnl = 0.0
        self.last_reset_date = datetime.now().date()
        self.peak_capital = initial_capital

        logger.info("RiskManager initialized")

    def check_order_risk(
        self,
        order: Order,
        current_positions: Dict[str, Dict[str, Any]],
        current_price: float
    ) -> Tuple[bool, str]:
        """
        检查订单风险

        Args:
            order: 订单
            current_positions: 当前持仓
            current_price: 当前价格

        Returns:
            (是否通过, 原因)
        """
        # 重置每日统计
        self._reset_daily_stats()

        # 1. 检查每日交易次数
        if self.daily_trades_count >= self.max_daily_trades:
            return False, f"超过每日最大交易次数限制({self.max_daily_trades})"

        # 2. 检查每日亏损
        if self.daily_pnl < -self.initial_capital * self.max_daily_loss_pct:
            return False, f"触发每日最大亏损限制({self.max_daily_loss_pct:.1%})"

        # 3. 检查最大回撤
        current_drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
        if current_drawdown > self.max_drawdown_pct:
            return False, f"触发最大回撤限制({self.max_drawdown_pct:.1%})"

        # 4. 检查仓位限制
        if order.side == OrderSide.BUY:
            # 买入订单
            order_value = current_price * order.quantity

            # 检查现金
            available_cash = self._calculate_available_cash(current_positions)
            if order_value > available_cash:
                return False, f"可用资金不足: 需要{order_value:.2f}, 可用{available_cash:.2f}"

            # 检查总仓位
            total_position_value = self._calculate_total_position_value(current_positions, current_price)
            new_total_position = (total_position_value + order_value) / self.current_capital

            if new_total_position > self.max_total_position:
                return False, f"超过总仓位限制: {new_total_position:.1%} > {self.max_total_position:.1%}"

            # 检查单只股票仓位
            current_stock_value = 0
            if order.stock_code in current_positions:
                current_stock_value = (
                    current_positions[order.stock_code]['quantity'] *
                    current_positions[order.stock_code]['current_price']
                )

            new_stock_position = (current_stock_value + order_value) / self.current_capital

            if new_stock_position > self.max_single_position:
                return False, f"超过单只股票仓位限制: {new_stock_position:.1%} > {self.max_single_position:.1%}"

        elif order.side == OrderSide.SELL:
            # 卖出订单
            if order.stock_code not in current_positions:
                return False, "没有持仓，无法卖出"

            available_quantity = current_positions[order.stock_code]['quantity']
            if order.quantity > available_quantity:
                return False, f"卖出数量超过持仓: {order.quantity} > {available_quantity}"

        return True, "风险检查通过"

    def check_stop_loss_profit(
        self,
        stock_code: str,
        position: Dict[str, Any],
        current_price: float
    ) -> Optional[str]:
        """
        检查止损止盈

        Args:
            stock_code: 股票代码
            position: 持仓信息
            current_price: 当前价格

        Returns:
            触发信号 (stop_loss/stop_profit/None)
        """
        cost_price = position['cost_price']
        pnl_pct = (current_price - cost_price) / cost_price

        # 止损
        if pnl_pct <= -self.stop_loss_pct:
            logger.warning(
                f"[STOP LOSS] {stock_code}: "
                f"current={current_price:.2f}, cost={cost_price:.2f}, "
                f"pnl={pnl_pct:.2%}"
            )
            return "stop_loss"

        # 止盈
        if pnl_pct >= self.stop_profit_pct:
            logger.info(
                f"[STOP PROFIT] {stock_code}: "
                f"current={current_price:.2f}, cost={cost_price:.2f}, "
                f"pnl={pnl_pct:.2%}"
            )
            return "stop_profit"

        return None

    def update_capital(self, new_capital: float):
        """更新资金"""
        self.current_capital = new_capital
        if new_capital > self.peak_capital:
            self.peak_capital = new_capital

    def record_trade(self, trade: Trade):
        """记录交易"""
        self.daily_trades_count += 1

        # 更新每日盈亏
        if trade.side == OrderSide.SELL:
            # 这里简化处理，实际需要根据成本计算盈亏
            pass

    def _reset_daily_stats(self):
        """重置每日统计"""
        today = datetime.now().date()
        if today > self.last_reset_date:
            self.daily_trades_count = 0
            self.daily_pnl = 0.0
            self.last_reset_date = today
            logger.info("Daily stats reset")

    def _calculate_available_cash(self, positions: Dict[str, Dict[str, Any]]) -> float:
        """计算可用资金"""
        position_value = self._calculate_total_position_value(positions)
        used_cash = position_value
        available = self.current_capital - used_cash

        # 保留最小现金储备
        min_reserve = self.current_capital * self.min_cash_reserve
        available = min(available, self.current_capital - min_reserve)

        return max(0, available)

    def _calculate_total_position_value(
        self,
        positions: Dict[str, Dict[str, Any]],
        additional_price: Optional[float] = None
    ) -> float:
        """计算总持仓市值"""
        total = 0
        for stock_code, position in positions.items():
            price = position.get('current_price', position.get('cost_price', 0))
            quantity = position.get('quantity', 0)
            total += price * quantity

        return total

    def get_risk_metrics(self) -> Dict[str, Any]:
        """获取风险指标"""
        drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
        daily_loss_pct = self.daily_pnl / self.initial_capital if self.initial_capital > 0 else 0

        return {
            'current_capital': self.current_capital,
            'peak_capital': self.peak_capital,
            'drawdown': drawdown,
            'drawdown_pct': drawdown,
            'daily_trades_count': self.daily_trades_count,
            'daily_pnl': self.daily_pnl,
            'daily_loss_pct': daily_loss_pct,
            'risk_status': self._get_risk_status()
        }

    def _get_risk_status(self) -> str:
        """获取风险状态"""
        drawdown = (self.peak_capital - self.current_capital) / self.peak_capital

        if drawdown > self.max_drawdown_pct * 0.8:
            return "危险"
        elif drawdown > self.max_drawdown_pct * 0.5:
            return "警告"
        elif self.daily_trades_count > self.max_daily_trades * 0.8:
            return "警告"
        else:
            return "正常"

    def is_trading_allowed(self) -> Tuple[bool, str]:
        """
        是否允许交易

        Returns:
            (是否允许, 原因)
        """
        self._reset_daily_stats()

        # 检查每日交易次数
        if self.daily_trades_count >= self.max_daily_trades:
            return False, "超过每日交易次数限制"

        # 检查每日亏损
        if self.daily_pnl < -self.initial_capital * self.max_daily_loss_pct:
            return False, "触发每日亏损限制"

        # 检查最大回撤
        drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
        if drawdown > self.max_drawdown_pct:
            return False, "触发最大回撤限制"

        return True, "允许交易"

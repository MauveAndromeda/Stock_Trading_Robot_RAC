"""
仓位管理器
"""
from typing import Dict, Any, List, Optional
from datetime import datetime
from loguru import logger
from .executor import Trade, OrderSide


class Position:
    """持仓"""

    def __init__(self, stock_code: str, stock_name: str):
        self.stock_code = stock_code
        self.stock_name = stock_name
        self.quantity = 0
        self.cost_price = 0.0
        self.current_price = 0.0
        self.market_value = 0.0
        self.pnl = 0.0
        self.pnl_pct = 0.0
        self.total_cost = 0.0
        self.trades: List[Trade] = []
        self.open_time: Optional[datetime] = None
        self.update_time: Optional[datetime] = None

    def add_trade(self, trade: Trade):
        """添加成交记录"""
        self.trades.append(trade)

        if trade.side == OrderSide.BUY:
            # 买入
            new_total_cost = self.total_cost + trade.amount + trade.commission
            new_quantity = self.quantity + trade.quantity
            self.cost_price = new_total_cost / new_quantity if new_quantity > 0 else 0
            self.quantity = new_quantity
            self.total_cost = new_total_cost

            if self.open_time is None:
                self.open_time = trade.trade_time

        elif trade.side == OrderSide.SELL:
            # 卖出
            sold_cost = self.cost_price * trade.quantity
            self.total_cost -= sold_cost
            self.quantity -= trade.quantity

            if self.quantity == 0:
                self.cost_price = 0
                self.total_cost = 0

        self.update_time = trade.trade_time
        logger.info(
            f"Position updated: {self.stock_code} "
            f"quantity={self.quantity}, cost={self.cost_price:.2f}"
        )

    def update_price(self, current_price: float):
        """更新当前价格"""
        self.current_price = current_price
        self.market_value = self.current_price * self.quantity

        if self.quantity > 0:
            self.pnl = self.market_value - self.total_cost
            self.pnl_pct = self.pnl / self.total_cost if self.total_cost > 0 else 0
        else:
            self.pnl = 0
            self.pnl_pct = 0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'stock_code': self.stock_code,
            'stock_name': self.stock_name,
            'quantity': self.quantity,
            'cost_price': self.cost_price,
            'current_price': self.current_price,
            'market_value': self.market_value,
            'pnl': self.pnl,
            'pnl_pct': self.pnl_pct,
            'total_cost': self.total_cost,
            'trades_count': len(self.trades),
            'open_time': self.open_time.isoformat() if self.open_time else None,
            'update_time': self.update_time.isoformat() if self.update_time else None
        }


class PositionManager:
    """仓位管理器"""

    def __init__(self):
        self.positions: Dict[str, Position] = {}
        logger.info("PositionManager initialized")

    def update_from_trade(self, trade: Trade, stock_name: str = ""):
        """
        从成交记录更新持仓

        Args:
            trade: 成交记录
            stock_name: 股票名称
        """
        stock_code = trade.stock_code

        # 创建或获取持仓
        if stock_code not in self.positions:
            self.positions[stock_code] = Position(stock_code, stock_name)

        position = self.positions[stock_code]
        position.add_trade(trade)

        # 如果仓位为0，删除持仓记录
        if position.quantity == 0:
            del self.positions[stock_code]
            logger.info(f"Position closed: {stock_code}")

    def update_prices(self, prices: Dict[str, float]):
        """
        批量更新价格

        Args:
            prices: {stock_code: current_price}
        """
        for stock_code, price in prices.items():
            if stock_code in self.positions:
                self.positions[stock_code].update_price(price)

    def get_position(self, stock_code: str) -> Optional[Position]:
        """获取持仓"""
        return self.positions.get(stock_code)

    def get_all_positions(self) -> Dict[str, Position]:
        """获取所有持仓"""
        return self.positions.copy()

    def get_position_list(self) -> List[Dict[str, Any]]:
        """获取持仓列表"""
        return [pos.to_dict() for pos in self.positions.values()]

    def get_total_market_value(self) -> float:
        """获取总市值"""
        return sum(pos.market_value for pos in self.positions.values())

    def get_total_pnl(self) -> float:
        """获取总盈亏"""
        return sum(pos.pnl for pos in self.positions.values())

    def get_total_pnl_pct(self, total_capital: float) -> float:
        """
        获取总盈亏比例

        Args:
            total_capital: 总资金

        Returns:
            盈亏比例
        """
        total_pnl = self.get_total_pnl()
        return total_pnl / total_capital if total_capital > 0 else 0

    def get_position_ratio(self, total_capital: float) -> float:
        """
        获取仓位比例

        Args:
            total_capital: 总资金

        Returns:
            仓位比例
        """
        total_value = self.get_total_market_value()
        return total_value / total_capital if total_capital > 0 else 0

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        positions = list(self.positions.values())

        if not positions:
            return {
                'total_positions': 0,
                'total_market_value': 0,
                'total_pnl': 0,
                'winning_count': 0,
                'losing_count': 0,
                'win_rate': 0
            }

        total_value = sum(pos.market_value for pos in positions)
        total_pnl = sum(pos.pnl for pos in positions)
        winning = [pos for pos in positions if pos.pnl > 0]
        losing = [pos for pos in positions if pos.pnl < 0]

        return {
            'total_positions': len(positions),
            'total_market_value': total_value,
            'total_pnl': total_pnl,
            'winning_count': len(winning),
            'losing_count': len(losing),
            'win_rate': len(winning) / len(positions) if positions else 0,
            'max_pnl': max(pos.pnl for pos in positions),
            'min_pnl': min(pos.pnl for pos in positions),
            'avg_pnl': total_pnl / len(positions) if positions else 0
        }

    def get_top_positions(self, n: int = 5, by: str = 'pnl') -> List[Dict[str, Any]]:
        """
        获取Top持仓

        Args:
            n: 数量
            by: 排序字段 (pnl, pnl_pct, market_value)

        Returns:
            Top持仓列表
        """
        positions = self.get_position_list()
        positions.sort(key=lambda x: x[by], reverse=True)
        return positions[:n]

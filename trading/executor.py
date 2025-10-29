"""
交易执行器
"""
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from datetime import datetime
from loguru import logger
import uuid


class OrderStatus(Enum):
    """订单状态"""
    PENDING = "pending"  # 待提交
    SUBMITTED = "submitted"  # 已提交
    PARTIAL_FILLED = "partial_filled"  # 部分成交
    FILLED = "filled"  # 完全成交
    CANCELLED = "cancelled"  # 已撤销
    REJECTED = "rejected"  # 被拒绝
    FAILED = "failed"  # 失败


class OrderType(Enum):
    """订单类型"""
    MARKET = "market"  # 市价单
    LIMIT = "limit"  # 限价单


class OrderSide(Enum):
    """买卖方向"""
    BUY = "buy"
    SELL = "sell"


@dataclass
class Order:
    """订单"""
    order_id: str
    stock_code: str
    stock_name: str
    side: OrderSide
    order_type: OrderType
    quantity: int
    price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    filled_price: float = 0.0
    create_time: datetime = field(default_factory=datetime.now)
    update_time: datetime = field(default_factory=datetime.now)
    commission: float = 0.0
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'order_id': self.order_id,
            'stock_code': self.stock_code,
            'stock_name': self.stock_name,
            'side': self.side.value,
            'order_type': self.order_type.value,
            'quantity': self.quantity,
            'price': self.price,
            'status': self.status.value,
            'filled_quantity': self.filled_quantity,
            'filled_price': self.filled_price,
            'create_time': self.create_time.isoformat(),
            'update_time': self.update_time.isoformat(),
            'commission': self.commission,
            'notes': self.notes
        }


@dataclass
class Trade:
    """成交记录"""
    trade_id: str
    order_id: str
    stock_code: str
    side: OrderSide
    quantity: int
    price: float
    amount: float
    commission: float
    trade_time: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'trade_id': self.trade_id,
            'order_id': self.order_id,
            'stock_code': self.stock_code,
            'side': self.side.value,
            'quantity': self.quantity,
            'price': self.price,
            'amount': self.amount,
            'commission': self.commission,
            'trade_time': self.trade_time.isoformat()
        }


class TradeExecutor:
    """交易执行器"""

    def __init__(
        self,
        mode: str = "simulation",
        broker_config: Optional[Dict[str, Any]] = None,
        commission_rate: float = 0.0003
    ):
        """
        初始化

        Args:
            mode: 模式 (simulation/live)
            broker_config: 券商配置
            commission_rate: 佣金率
        """
        self.mode = mode
        self.broker_config = broker_config or {}
        self.commission_rate = commission_rate

        self.orders: Dict[str, Order] = {}
        self.trades: List[Trade] = []

        logger.info(f"TradeExecutor initialized in {mode} mode")

    def create_order(
        self,
        stock_code: str,
        stock_name: str,
        side: OrderSide,
        quantity: int,
        price: Optional[float] = None,
        order_type: OrderType = OrderType.LIMIT,
        notes: str = ""
    ) -> Order:
        """
        创建订单

        Args:
            stock_code: 股票代码
            stock_name: 股票名称
            side: 买卖方向
            quantity: 数量
            price: 价格(市价单不需要)
            order_type: 订单类型
            notes: 备注

        Returns:
            Order对象
        """
        order_id = f"ORD_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"

        order = Order(
            order_id=order_id,
            stock_code=stock_code,
            stock_name=stock_name,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            notes=notes
        )

        self.orders[order_id] = order
        logger.info(f"Order created: {order_id} {side.value} {stock_code} {quantity}@{price}")

        return order

    def submit_order(self, order: Order) -> bool:
        """
        提交订单

        Args:
            order: 订单对象

        Returns:
            是否成功
        """
        try:
            if self.mode == "simulation":
                # 模拟模式：直接标记为已提交
                order.status = OrderStatus.SUBMITTED
                order.update_time = datetime.now()
                logger.info(f"[SIMULATION] Order submitted: {order.order_id}")
                return True

            elif self.mode == "live":
                # 实盘模式：调用券商API
                logger.info(f"[LIVE] Submitting order to broker: {order.order_id}")
                # TODO: 实现真实券商API调用
                # success = self._submit_to_broker(order)
                # return success
                return False

            else:
                logger.error(f"Unknown mode: {self.mode}")
                return False

        except Exception as e:
            logger.error(f"Error submitting order {order.order_id}: {e}")
            order.status = OrderStatus.FAILED
            return False

    def simulate_fill(self, order: Order, fill_price: float) -> bool:
        """
        模拟订单成交

        Args:
            order: 订单
            fill_price: 成交价格

        Returns:
            是否成功
        """
        try:
            if order.status not in [OrderStatus.SUBMITTED, OrderStatus.PARTIAL_FILLED]:
                logger.warning(f"Order {order.order_id} cannot be filled, status: {order.status}")
                return False

            # 计算佣金
            amount = fill_price * order.quantity
            commission = amount * self.commission_rate

            # 更新订单
            order.filled_quantity = order.quantity
            order.filled_price = fill_price
            order.commission = commission
            order.status = OrderStatus.FILLED
            order.update_time = datetime.now()

            # 创建成交记录
            trade = Trade(
                trade_id=f"TRD_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}",
                order_id=order.order_id,
                stock_code=order.stock_code,
                side=order.side,
                quantity=order.quantity,
                price=fill_price,
                amount=amount,
                commission=commission
            )

            self.trades.append(trade)

            logger.info(
                f"[SIMULATION] Order filled: {order.order_id} "
                f"{order.quantity}@{fill_price:.2f}, "
                f"commission={commission:.2f}"
            )

            return True

        except Exception as e:
            logger.error(f"Error filling order {order.order_id}: {e}")
            return False

    def cancel_order(self, order_id: str) -> bool:
        """
        撤销订单

        Args:
            order_id: 订单ID

        Returns:
            是否成功
        """
        if order_id not in self.orders:
            logger.warning(f"Order {order_id} not found")
            return False

        order = self.orders[order_id]

        if order.status not in [OrderStatus.PENDING, OrderStatus.SUBMITTED]:
            logger.warning(f"Order {order_id} cannot be cancelled, status: {order.status}")
            return False

        order.status = OrderStatus.CANCELLED
        order.update_time = datetime.now()

        logger.info(f"Order cancelled: {order_id}")
        return True

    def get_order(self, order_id: str) -> Optional[Order]:
        """获取订单"""
        return self.orders.get(order_id)

    def get_orders(
        self,
        stock_code: Optional[str] = None,
        status: Optional[OrderStatus] = None
    ) -> List[Order]:
        """
        获取订单列表

        Args:
            stock_code: 股票代码过滤
            status: 状态过滤

        Returns:
            订单列表
        """
        orders = list(self.orders.values())

        if stock_code:
            orders = [o for o in orders if o.stock_code == stock_code]

        if status:
            orders = [o for o in orders if o.status == status]

        return orders

    def get_trades(
        self,
        stock_code: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Trade]:
        """
        获取成交记录

        Args:
            stock_code: 股票代码过滤
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            成交记录列表
        """
        trades = self.trades.copy()

        if stock_code:
            trades = [t for t in trades if t.stock_code == stock_code]

        if start_date:
            trades = [t for t in trades if t.trade_time >= start_date]

        if end_date:
            trades = [t for t in trades if t.trade_time <= end_date]

        return trades

    def get_statistics(self) -> Dict[str, Any]:
        """获取交易统计"""
        total_orders = len(self.orders)
        filled_orders = len([o for o in self.orders.values() if o.status == OrderStatus.FILLED])
        total_trades = len(self.trades)

        total_buy_amount = sum(
            t.amount for t in self.trades if t.side == OrderSide.BUY
        )
        total_sell_amount = sum(
            t.amount for t in self.trades if t.side == OrderSide.SELL
        )
        total_commission = sum(t.commission for t in self.trades)

        return {
            'total_orders': total_orders,
            'filled_orders': filled_orders,
            'fill_rate': filled_orders / total_orders if total_orders > 0 else 0,
            'total_trades': total_trades,
            'total_buy_amount': total_buy_amount,
            'total_sell_amount': total_sell_amount,
            'net_amount': total_sell_amount - total_buy_amount,
            'total_commission': total_commission
        }

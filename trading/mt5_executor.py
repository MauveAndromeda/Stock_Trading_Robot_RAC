"""
MT5交易执行器
用于外汇交易的订单执行
"""
import MetaTrader5 as mt5
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from datetime import datetime
from loguru import logger


class MT5OrderType(Enum):
    """MT5订单类型"""
    BUY = mt5.ORDER_TYPE_BUY
    SELL = mt5.ORDER_TYPE_SELL
    BUY_LIMIT = mt5.ORDER_TYPE_BUY_LIMIT
    SELL_LIMIT = mt5.ORDER_TYPE_SELL_LIMIT
    BUY_STOP = mt5.ORDER_TYPE_BUY_STOP
    SELL_STOP = mt5.ORDER_TYPE_SELL_STOP


@dataclass
class MT5TradeRequest:
    """MT5交易请求"""
    symbol: str
    volume: float  # 手数
    order_type: MT5OrderType
    price: Optional[float] = None  # 限价单需要
    sl: Optional[float] = None  # 止损价格
    tp: Optional[float] = None  # 止盈价格
    deviation: int = 20  # 允许的价格偏差（点）
    magic: int = 234000  # 魔术数字（用于识别订单）
    comment: str = "AI_Trading"


class MT5Executor:
    """MT5交易执行器"""

    def __init__(
        self,
        leverage: int = 50,
        max_slippage_points: int = 20,
        magic_number: int = 234000
    ):
        """
        初始化MT5执行器

        Args:
            leverage: 杠杆倍数
            max_slippage_points: 最大滑点（点）
            magic_number: 魔术数字
        """
        self.leverage = leverage
        self.max_slippage_points = max_slippage_points
        self.magic_number = magic_number
        self.connected = False

        logger.info(f"MT5Executor initialized with {leverage}x leverage")

    def connect(self) -> bool:
        """连接MT5"""
        if not mt5.initialize():
            logger.error(f"MT5 initialize failed: {mt5.last_error()}")
            return False

        self.connected = True
        logger.info("MT5Executor connected")
        return True

    def disconnect(self):
        """断开连接"""
        if self.connected:
            mt5.shutdown()
            self.connected = False
            logger.info("MT5Executor disconnected")

    def calculate_lot_size(
        self,
        symbol: str,
        risk_amount: float,
        stop_loss_pips: float,
        account_currency: str = "USD"
    ) -> float:
        """
        计算手数（基于风险管理）

        Args:
            symbol: 交易品种
            risk_amount: 愿意承受的风险金额（美元）
            stop_loss_pips: 止损点数
            account_currency: 账户货币

        Returns:
            手数
        """
        # 获取品种信息
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logger.error(f"Failed to get symbol info for {symbol}")
            return 0.0

        # 计算每点价值
        # 对于外汇，1标准手 = 100,000基础货币
        contract_size = symbol_info.trade_contract_size
        point = symbol_info.point

        # 获取当前价格
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            logger.error(f"Failed to get tick for {symbol}")
            return 0.0

        price = tick.ask

        # 计算pip价值（外汇通常是point * 10）
        pip_value = (contract_size * point * 10) / price

        # 计算手数
        # risk_amount = lot_size * stop_loss_pips * pip_value
        # lot_size = risk_amount / (stop_loss_pips * pip_value)
        lot_size = risk_amount / (stop_loss_pips * pip_value)

        # 调整到允许的最小/最大手数
        lot_size = max(symbol_info.volume_min, lot_size)
        lot_size = min(symbol_info.volume_max, lot_size)

        # 调整到步长
        lot_size = round(lot_size / symbol_info.volume_step) * symbol_info.volume_step

        logger.info(
            f"Calculated lot size for {symbol}: {lot_size:.2f} "
            f"(risk=${risk_amount:.2f}, SL={stop_loss_pips} pips)"
        )

        return lot_size

    def send_market_order(
        self,
        symbol: str,
        volume: float,
        order_type: str,  # "buy" or "sell"
        sl_pips: Optional[float] = None,
        tp_pips: Optional[float] = None,
        comment: str = "AI_Trading"
    ) -> Optional[Dict[str, Any]]:
        """
        发送市价单

        Args:
            symbol: 交易品种
            volume: 手数
            order_type: 订单类型 ("buy"/"sell")
            sl_pips: 止损点数
            tp_pips: 止盈点数
            comment: 订单注释

        Returns:
            订单结果
        """
        if not self.connected:
            logger.error("Not connected to MT5")
            return None

        # 获取品种信息
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logger.error(f"Symbol {symbol} not found")
            return None

        if not symbol_info.visible:
            if not mt5.symbol_select(symbol, True):
                logger.error(f"Failed to select {symbol}")
                return None

        # 获取当前价格
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            logger.error(f"Failed to get tick for {symbol}")
            return None

        # 确定订单类型和价格
        if order_type.lower() == "buy":
            trade_type = mt5.ORDER_TYPE_BUY
            price = tick.ask
            sl_price = price - (sl_pips * symbol_info.point * 10) if sl_pips else 0
            tp_price = price + (tp_pips * symbol_info.point * 10) if tp_pips else 0
        else:  # sell
            trade_type = mt5.ORDER_TYPE_SELL
            price = tick.bid
            sl_price = price + (sl_pips * symbol_info.point * 10) if sl_pips else 0
            tp_price = price - (tp_pips * symbol_info.point * 10) if tp_pips else 0

        # 构建交易请求
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": trade_type,
            "price": price,
            "sl": sl_price,
            "tp": tp_price,
            "deviation": self.max_slippage_points,
            "magic": self.magic_number,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,  # Good Till Cancelled
            "type_filling": mt5.ORDER_FILLING_IOC,  # Immediate or Cancel
        }

        # 发送订单
        logger.info(f"Sending {order_type.upper()} order: {symbol} {volume} lots @ {price}")

        result = mt5.order_send(request)

        if result is None:
            logger.error(f"Order send failed: {mt5.last_error()}")
            return None

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Order failed: {result.comment} (code: {result.retcode})")
            return None

        logger.info(
            f"✅ Order executed successfully! "
            f"Ticket: {result.order}, "
            f"Volume: {result.volume}, "
            f"Price: {result.price}"
        )

        return {
            'ticket': result.order,
            'symbol': symbol,
            'volume': result.volume,
            'price': result.price,
            'type': order_type,
            'sl': sl_price,
            'tp': tp_price,
            'time': datetime.now(),
            'comment': comment
        }

    def close_position(
        self,
        ticket: int,
        volume: Optional[float] = None
    ) -> bool:
        """
        平仓

        Args:
            ticket: 订单号
            volume: 平仓手数（None表示全部）

        Returns:
            是否成功
        """
        if not self.connected:
            logger.error("Not connected to MT5")
            return False

        # 获取持仓信息
        positions = mt5.positions_get(ticket=ticket)
        if positions is None or len(positions) == 0:
            logger.error(f"Position {ticket} not found")
            return False

        position = positions[0]

        # 确定平仓手数
        close_volume = volume if volume else position.volume

        # 确定平仓类型（买单用卖平，卖单用买平）
        if position.type == mt5.ORDER_TYPE_BUY:
            close_type = mt5.ORDER_TYPE_SELL
        else:
            close_type = mt5.ORDER_TYPE_BUY

        # 获取当前价格
        tick = mt5.symbol_info_tick(position.symbol)
        if tick is None:
            logger.error(f"Failed to get tick for {position.symbol}")
            return False

        price = tick.bid if close_type == mt5.ORDER_TYPE_SELL else tick.ask

        # 构建平仓请求
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": position.symbol,
            "volume": close_volume,
            "type": close_type,
            "position": ticket,
            "price": price,
            "deviation": self.max_slippage_points,
            "magic": self.magic_number,
            "comment": "Close by AI",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        # 发送平仓请求
        logger.info(f"Closing position {ticket}: {close_volume} lots @ {price}")

        result = mt5.order_send(request)

        if result is None:
            logger.error(f"Close failed: {mt5.last_error()}")
            return False

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Close failed: {result.comment} (code: {result.retcode})")
            return False

        logger.info(f"✅ Position {ticket} closed successfully")
        return True

    def modify_position(
        self,
        ticket: int,
        sl: Optional[float] = None,
        tp: Optional[float] = None
    ) -> bool:
        """
        修改持仓的止损止盈

        Args:
            ticket: 订单号
            sl: 新止损价格
            tp: 新止盈价格

        Returns:
            是否成功
        """
        if not self.connected:
            logger.error("Not connected to MT5")
            return False

        # 获取持仓
        positions = mt5.positions_get(ticket=ticket)
        if positions is None or len(positions) == 0:
            logger.error(f"Position {ticket} not found")
            return False

        position = positions[0]

        # 构建修改请求
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "symbol": position.symbol,
            "sl": sl if sl is not None else position.sl,
            "tp": tp if tp is not None else position.tp,
        }

        # 发送请求
        result = mt5.order_send(request)

        if result is None:
            logger.error(f"Modify failed: {mt5.last_error()}")
            return False

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Modify failed: {result.comment}")
            return False

        logger.info(f"✅ Position {ticket} modified: SL={sl}, TP={tp}")
        return True

    def get_open_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        获取持仓列表

        Args:
            symbol: 筛选品种

        Returns:
            持仓列表
        """
        if not self.connected:
            logger.error("Not connected to MT5")
            return []

        if symbol:
            positions = mt5.positions_get(symbol=symbol)
        else:
            positions = mt5.positions_get()

        if positions is None:
            return []

        result = []
        for pos in positions:
            result.append({
                'ticket': pos.ticket,
                'symbol': pos.symbol,
                'type': 'buy' if pos.type == mt5.ORDER_TYPE_BUY else 'sell',
                'volume': pos.volume,
                'price_open': pos.price_open,
                'price_current': pos.price_current,
                'sl': pos.sl,
                'tp': pos.tp,
                'profit': pos.profit,
                'swap': pos.swap,
                'time': datetime.fromtimestamp(pos.time)
            })

        return result

    def __enter__(self):
        """上下文管理器"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文"""
        self.disconnect()


# 便捷函数
def test_mt5_trading():
    """测试MT5交易功能"""
    with MT5Executor(leverage=50) as executor:
        # 获取账户信息
        account = mt5.account_info()
        if account:
            logger.info(f"Balance: ${account.balance:.2f}")
            logger.info(f"Equity: ${account.equity:.2f}")
            logger.info(f"Leverage: 1:{account.leverage}")

        # 获取持仓
        positions = executor.get_open_positions()
        logger.info(f"Open positions: {len(positions)}")

        for pos in positions:
            logger.info(
                f"  {pos['symbol']} {pos['type'].upper()} "
                f"{pos['volume']} lots, "
                f"P/L: ${pos['profit']:.2f}"
            )


if __name__ == "__main__":
    test_mt5_trading()

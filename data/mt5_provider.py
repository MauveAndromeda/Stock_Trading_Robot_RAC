"""
MetaTrader 5 数据提供者
用于从本地MT5终端获取外汇数据
"""
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from loguru import logger
import pytz


class MT5DataProvider:
    """MT5数据提供者"""

    def __init__(
        self,
        login: Optional[int] = None,
        password: Optional[str] = None,
        server: Optional[str] = None,
        path: Optional[str] = None,
        timeout: int = 60000
    ):
        """
        初始化MT5连接

        Args:
            login: MT5账号
            password: MT5密码
            server: MT5服务器
            path: MT5终端路径（可选）
            timeout: 连接超时（毫秒）
        """
        self.login = login
        self.password = password
        self.server = server
        self.path = path
        self.timeout = timeout
        self.connected = False
        self.timezone = pytz.timezone("Etc/UTC")

        logger.info("MT5DataProvider initialized")

    def connect(self) -> bool:
        """
        连接到MT5终端

        Returns:
            是否成功连接
        """
        try:
            # 初始化MT5
            if self.path:
                if not mt5.initialize(path=self.path, timeout=self.timeout):
                    logger.error(f"MT5 initialize failed: {mt5.last_error()}")
                    return False
            else:
                if not mt5.initialize(timeout=self.timeout):
                    logger.error(f"MT5 initialize failed: {mt5.last_error()}")
                    return False

            # 如果提供了登录信息，则登录
            if self.login and self.password and self.server:
                if not mt5.login(self.login, password=self.password, server=self.server):
                    logger.error(f"MT5 login failed: {mt5.last_error()}")
                    mt5.shutdown()
                    return False
                logger.info(f"MT5 logged in: {self.login}@{self.server}")
            else:
                logger.info("MT5 initialized without login (using current terminal account)")

            self.connected = True

            # 获取账户信息
            account_info = mt5.account_info()
            if account_info:
                logger.info(f"Account balance: ${account_info.balance:.2f}")
                logger.info(f"Account leverage: 1:{account_info.leverage}")
                logger.info(f"Account currency: {account_info.currency}")

            return True

        except Exception as e:
            logger.error(f"Error connecting to MT5: {e}")
            return False

    def disconnect(self):
        """断开MT5连接"""
        if self.connected:
            mt5.shutdown()
            self.connected = False
            logger.info("MT5 disconnected")

    def get_symbols(self, group: str = "*") -> List[str]:
        """
        获取可用交易品种列表

        Args:
            group: 品种组过滤（如 "Forex*", "EURUSD*"）

        Returns:
            交易品种列表
        """
        if not self.connected:
            logger.warning("Not connected to MT5")
            return []

        symbols = mt5.symbols_get(group=group)
        if symbols is None:
            logger.error(f"Failed to get symbols: {mt5.last_error()}")
            return []

        return [s.name for s in symbols]

    def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        获取交易品种信息

        Args:
            symbol: 交易品种（如 "EURUSD"）

        Returns:
            品种信息字典
        """
        if not self.connected:
            logger.warning("Not connected to MT5")
            return None

        info = mt5.symbol_info(symbol)
        if info is None:
            logger.error(f"Failed to get symbol info for {symbol}: {mt5.last_error()}")
            return None

        return {
            'symbol': symbol,
            'description': info.description,
            'point': info.point,  # 最小价格变动
            'digits': info.digits,  # 小数位数
            'spread': info.spread,  # 点差
            'trade_contract_size': info.trade_contract_size,  # 合约大小
            'volume_min': info.volume_min,  # 最小手数
            'volume_max': info.volume_max,  # 最大手数
            'volume_step': info.volume_step,  # 手数步长
            'currency_base': info.currency_base,  # 基础货币
            'currency_profit': info.currency_profit,  # 利润货币
            'trade_mode': info.trade_mode,
            'margin_initial': info.margin_initial,  # 初始保证金
        }

    def get_realtime_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        获取实时行情

        Args:
            symbol: 交易品种

        Returns:
            实时数据
        """
        if not self.connected:
            logger.warning("Not connected to MT5")
            return None

        # 获取最新tick
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            logger.error(f"Failed to get tick for {symbol}: {mt5.last_error()}")
            return None

        # 获取品种信息
        info = self.get_symbol_info(symbol)

        return {
            'symbol': symbol,
            'time': datetime.fromtimestamp(tick.time, tz=self.timezone),
            'bid': tick.bid,  # 买价
            'ask': tick.ask,  # 卖价
            'last': tick.last,  # 最后成交价
            'volume': tick.volume,  # 成交量
            'spread': tick.ask - tick.bid,  # 点差
            'point': info['point'] if info else 0.0001,
            'digits': info['digits'] if info else 5
        }

    def get_historical_data(
        self,
        symbol: str,
        timeframe: str = "H1",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        count: int = 1000
    ) -> Optional[pd.DataFrame]:
        """
        获取历史K线数据

        Args:
            symbol: 交易品种
            timeframe: 时间周期 (M1, M5, M15, M30, H1, H4, D1, W1, MN1)
            start_date: 开始时间
            end_date: 结束时间
            count: 数据条数

        Returns:
            DataFrame with OHLCV data
        """
        if not self.connected:
            logger.warning("Not connected to MT5")
            return None

        # 时间周期映射
        timeframe_map = {
            'M1': mt5.TIMEFRAME_M1,
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'M30': mt5.TIMEFRAME_M30,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1,
            'W1': mt5.TIMEFRAME_W1,
            'MN1': mt5.TIMEFRAME_MN1
        }

        tf = timeframe_map.get(timeframe.upper(), mt5.TIMEFRAME_H1)

        # 获取数据
        if start_date and end_date:
            rates = mt5.copy_rates_range(symbol, tf, start_date, end_date)
        elif end_date:
            rates = mt5.copy_rates_from(symbol, tf, end_date, count)
        else:
            rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)

        if rates is None or len(rates) == 0:
            logger.error(f"Failed to get historical data for {symbol}: {mt5.last_error()}")
            return None

        # 转换为DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)

        # 重命名列
        df.rename(columns={
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'tick_volume': 'volume',
            'real_volume': 'real_volume'
        }, inplace=True)

        logger.info(f"Got {len(df)} bars for {symbol} {timeframe}")
        return df

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算技术指标

        Args:
            df: OHLCV数据

        Returns:
            添加了技术指标的DataFrame
        """
        if df is None or df.empty:
            return df

        try:
            # 移动平均线
            df['ma5'] = df['close'].rolling(window=5).mean()
            df['ma10'] = df['close'].rolling(window=10).mean()
            df['ma20'] = df['close'].rolling(window=20).mean()
            df['ma50'] = df['close'].rolling(window=50).mean()

            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))

            # MACD
            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = exp1 - exp2
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']

            # 布林带
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            bb_std = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)

            # ATR (Average True Range) - 外汇常用
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            df['atr'] = true_range.rolling(14).mean()

            # 波动率
            df['volatility'] = df['close'].pct_change().rolling(window=20).std()

            return df

        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return df

    def get_account_info(self) -> Optional[Dict[str, Any]]:
        """
        获取账户信息

        Returns:
            账户信息字典
        """
        if not self.connected:
            logger.warning("Not connected to MT5")
            return None

        account = mt5.account_info()
        if account is None:
            logger.error(f"Failed to get account info: {mt5.last_error()}")
            return None

        return {
            'login': account.login,
            'balance': account.balance,
            'equity': account.equity,
            'margin': account.margin,
            'free_margin': account.margin_free,
            'margin_level': account.margin_level,
            'leverage': account.leverage,
            'profit': account.profit,
            'currency': account.currency,
            'server': account.server,
            'name': account.name
        }

    def get_positions(self) -> List[Dict[str, Any]]:
        """
        获取当前持仓

        Returns:
            持仓列表
        """
        if not self.connected:
            logger.warning("Not connected to MT5")
            return []

        positions = mt5.positions_get()
        if positions is None:
            logger.error(f"Failed to get positions: {mt5.last_error()}")
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
                'sl': pos.sl,  # 止损
                'tp': pos.tp,  # 止盈
                'profit': pos.profit,
                'swap': pos.swap,
                'comment': pos.comment,
                'time': datetime.fromtimestamp(pos.time, tz=self.timezone)
            })

        return result

    def __enter__(self):
        """上下文管理器入口"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        self.disconnect()


# 便捷函数
def test_mt5_connection(
    login: Optional[int] = None,
    password: Optional[str] = None,
    server: Optional[str] = None
) -> bool:
    """
    测试MT5连接

    Args:
        login: MT5账号
        password: MT5密码
        server: MT5服务器

    Returns:
        是否连接成功
    """
    with MT5DataProvider(login, password, server) as provider:
        if provider.connected:
            logger.info("✅ MT5 connection successful!")

            # 显示账户信息
            account = provider.get_account_info()
            if account:
                logger.info(f"Account: {account['login']}")
                logger.info(f"Balance: ${account['balance']:.2f}")
                logger.info(f"Equity: ${account['equity']:.2f}")
                logger.info(f"Leverage: 1:{account['leverage']}")

            # 显示可用品种
            symbols = provider.get_symbols("Forex*")
            logger.info(f"Available Forex pairs: {len(symbols)}")
            if symbols:
                logger.info(f"Examples: {', '.join(symbols[:5])}")

            return True
        else:
            logger.error("❌ MT5 connection failed!")
            return False


if __name__ == "__main__":
    # 测试连接（使用当前MT5终端的账号）
    test_mt5_connection()

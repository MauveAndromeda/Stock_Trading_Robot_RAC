"""
市场数据获取模块
支持akshare和yfinance数据源
"""
import akshare as ak
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from loguru import logger
import time


class MarketDataProvider:
    """市场数据提供者"""

    def __init__(self, source: str = "akshare", cache_enabled: bool = True):
        """
        初始化

        Args:
            source: 数据源 (akshare, yfinance)
            cache_enabled: 是否启用缓存
        """
        self.source = source
        self.cache_enabled = cache_enabled
        self._cache = {}
        logger.info(f"MarketDataProvider initialized with source: {source}")

    def get_stock_list(
        self,
        market: str = "A股",
        min_market_cap: float = 10,
        min_liquidity: float = 5000000,
        max_stocks: int = 500
    ) -> List[Dict[str, Any]]:
        """
        获取股票列表

        Args:
            market: 市场类型
            min_market_cap: 最小市值(亿元)
            min_liquidity: 最小日成交额
            max_stocks: 最大数量

        Returns:
            股票列表
        """
        try:
            logger.info(f"Fetching stock list from {self.source}...")

            if self.source == "akshare":
                # 获取A股实时行情
                df = ak.stock_zh_a_spot_em()

                # 筛选
                df = df[df['总市值'] >= min_market_cap * 100000000]  # 转换为元
                df = df[df['成交额'] >= min_liquidity]
                df = df.head(max_stocks)

                stocks = []
                for _, row in df.iterrows():
                    stocks.append({
                        'code': row['代码'],
                        'name': row['名称'],
                        'price': float(row['最新价']),
                        'change_pct': float(row['涨跌幅']),
                        'volume': float(row['成交量']),
                        'amount': float(row['成交额']),
                        'market_cap': float(row['总市值']),
                        'turnover_rate': float(row['换手率'])
                    })

                logger.info(f"Fetched {len(stocks)} stocks")
                return stocks

            else:
                logger.warning(f"Unsupported data source: {self.source}")
                return []

        except Exception as e:
            logger.error(f"Error fetching stock list: {e}")
            return []

    def get_realtime_data(self, stock_code: str) -> Optional[Dict[str, Any]]:
        """
        获取实时行情数据

        Args:
            stock_code: 股票代码

        Returns:
            实时数据
        """
        cache_key = f"realtime_{stock_code}"

        # 检查缓存(60秒有效)
        if self.cache_enabled and cache_key in self._cache:
            cached_data, cached_time = self._cache[cache_key]
            if (datetime.now() - cached_time).seconds < 60:
                return cached_data

        try:
            if self.source == "akshare":
                # 获取实时数据
                df = ak.stock_zh_a_spot_em()
                stock_df = df[df['代码'] == stock_code]

                if stock_df.empty:
                    logger.warning(f"Stock {stock_code} not found")
                    return None

                row = stock_df.iloc[0]
                data = {
                    'code': stock_code,
                    'name': row['名称'],
                    'price': float(row['最新价']),
                    'open': float(row['今开']),
                    'high': float(row['最高']),
                    'low': float(row['最低']),
                    'close': float(row['昨收']),
                    'change': float(row['涨跌额']),
                    'change_pct': float(row['涨跌幅']),
                    'volume': float(row['成交量']),
                    'amount': float(row['成交额']),
                    'turnover_rate': float(row['换手率']),
                    'volume_ratio': float(row.get('量比', 1)),
                    'timestamp': datetime.now()
                }

                # 缓存
                if self.cache_enabled:
                    self._cache[cache_key] = (data, datetime.now())

                return data

        except Exception as e:
            logger.error(f"Error fetching realtime data for {stock_code}: {e}")
            return None

    def get_historical_data(
        self,
        stock_code: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: str = "daily"
    ) -> Optional[pd.DataFrame]:
        """
        获取历史行情数据

        Args:
            stock_code: 股票代码
            start_date: 开始日期 (YYYYMMDD)
            end_date: 结束日期 (YYYYMMDD)
            period: 周期 (daily, weekly, monthly)

        Returns:
            历史数据DataFrame
        """
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y%m%d")
        if end_date is None:
            end_date = datetime.now().strftime("%Y%m%d")

        try:
            if self.source == "akshare":
                # 获取历史行情
                if period == "daily":
                    df = ak.stock_zh_a_hist(
                        symbol=stock_code,
                        period="daily",
                        start_date=start_date,
                        end_date=end_date,
                        adjust="qfq"  # 前复权
                    )
                else:
                    df = ak.stock_zh_a_hist(
                        symbol=stock_code,
                        period=period,
                        start_date=start_date,
                        end_date=end_date,
                        adjust="qfq"
                    )

                if df.empty:
                    logger.warning(f"No historical data for {stock_code}")
                    return None

                # 重命名列
                df.rename(columns={
                    '日期': 'date',
                    '开盘': 'open',
                    '收盘': 'close',
                    '最高': 'high',
                    '最低': 'low',
                    '成交量': 'volume',
                    '成交额': 'amount',
                    '振幅': 'amplitude',
                    '涨跌幅': 'change_pct',
                    '涨跌额': 'change',
                    '换手率': 'turnover_rate'
                }, inplace=True)

                return df

        except Exception as e:
            logger.error(f"Error fetching historical data for {stock_code}: {e}")
            return None

    def get_financial_data(self, stock_code: str) -> Optional[Dict[str, Any]]:
        """
        获取财务数据

        Args:
            stock_code: 股票代码

        Returns:
            财务数据
        """
        try:
            if self.source == "akshare":
                # 获取主要指标
                df = ak.stock_financial_analysis_indicator(symbol=stock_code)

                if df.empty:
                    return None

                # 取最新一期数据
                latest = df.iloc[-1]

                return {
                    'pe_ratio': float(latest.get('市盈率', 30)),
                    'pb_ratio': float(latest.get('市净率', 3)),
                    'roe': float(latest.get('净资产收益率', 10)) / 100,
                    'debt_ratio': float(latest.get('资产负债率', 50)) / 100,
                    'gross_margin': float(latest.get('销售毛利率', 20)) / 100,
                    'net_margin': float(latest.get('销售净利率', 10)) / 100
                }

        except Exception as e:
            logger.error(f"Error fetching financial data for {stock_code}: {e}")
            # 返回默认值
            return {
                'pe_ratio': 30,
                'pb_ratio': 3,
                'roe': 0.10,
                'debt_ratio': 0.50,
                'gross_margin': 0.20,
                'net_margin': 0.10
            }

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算技术指标

        Args:
            df: 历史数据DataFrame

        Returns:
            添加了技术指标的DataFrame
        """
        if df is None or df.empty:
            return df

        try:
            # 均线
            df['ma5'] = df['close'].rolling(window=5).mean()
            df['ma10'] = df['close'].rolling(window=10).mean()
            df['ma20'] = df['close'].rolling(window=20).mean()
            df['ma60'] = df['close'].rolling(window=60).mean()

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

            # 波动率
            df['volatility'] = df['close'].pct_change().rolling(window=20).std()

            return df

        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return df

    def get_market_sentiment(self, stock_code: str) -> float:
        """
        获取市场情绪指标(0-1)

        Args:
            stock_code: 股票代码

        Returns:
            情绪得分
        """
        try:
            # 这里可以接入真实的市场情绪数据
            # 暂时使用简单的模拟
            realtime = self.get_realtime_data(stock_code)
            if realtime is None:
                return 0.5

            change_pct = realtime.get('change_pct', 0)
            volume_ratio = realtime.get('volume_ratio', 1)

            # 简单的情绪计算
            sentiment = 0.5 + (change_pct / 100) + (volume_ratio - 1) * 0.1
            sentiment = max(0, min(1, sentiment))

            return sentiment

        except Exception as e:
            logger.error(f"Error calculating market sentiment: {e}")
            return 0.5

    def clear_cache(self):
        """清空缓存"""
        self._cache.clear()
        logger.info("Cache cleared")

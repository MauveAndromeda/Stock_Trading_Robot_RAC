"""
股票分析引擎
整合市场数据和技术分析，为Agent提供全面的分析数据
"""
import pandas as pd
from typing import Dict, Any, Optional, List
from datetime import datetime
from loguru import logger
from .market_data import MarketDataProvider


class StockAnalyzer:
    """股票分析器"""

    def __init__(self, data_provider: MarketDataProvider):
        """
        初始化

        Args:
            data_provider: 数据提供者
        """
        self.data_provider = data_provider
        logger.info("StockAnalyzer initialized")

    def analyze_stock(self, stock_code: str) -> Optional[Dict[str, Any]]:
        """
        全面分析股票

        Args:
            stock_code: 股票代码

        Returns:
            分析结果
        """
        try:
            # 获取实时数据
            realtime_data = self.data_provider.get_realtime_data(stock_code)
            if realtime_data is None:
                logger.warning(f"Cannot get realtime data for {stock_code}")
                return None

            # 获取历史数据
            hist_data = self.data_provider.get_historical_data(
                stock_code,
                period="daily"
            )

            # 获取财务数据
            financial_data = self.data_provider.get_financial_data(stock_code)

            # 计算技术指标
            if hist_data is not None and not hist_data.empty:
                hist_data = self.data_provider.calculate_technical_indicators(hist_data)
                latest_hist = hist_data.iloc[-1]

                # 从历史数据中提取技术指标
                technical_indicators = {
                    'ma5': float(latest_hist.get('ma5', realtime_data['price'])),
                    'ma10': float(latest_hist.get('ma10', realtime_data['price'])),
                    'ma20': float(latest_hist.get('ma20', realtime_data['price'])),
                    'ma60': float(latest_hist.get('ma60', realtime_data['price'])),
                    'rsi': float(latest_hist.get('rsi', 50)),
                    'macd': float(latest_hist.get('macd', 0)),
                    'macd_signal': float(latest_hist.get('macd_signal', 0)),
                    'macd_hist': float(latest_hist.get('macd_hist', 0)),
                    'bb_upper': float(latest_hist.get('bb_upper', realtime_data['price'] * 1.1)),
                    'bb_middle': float(latest_hist.get('bb_middle', realtime_data['price'])),
                    'bb_lower': float(latest_hist.get('bb_lower', realtime_data['price'] * 0.9)),
                    'volatility': float(latest_hist.get('volatility', 0.02))
                }
            else:
                # 使用默认值
                technical_indicators = {
                    'ma5': realtime_data['price'],
                    'ma10': realtime_data['price'],
                    'ma20': realtime_data['price'],
                    'ma60': realtime_data['price'],
                    'rsi': 50,
                    'macd': 0,
                    'macd_signal': 0,
                    'macd_hist': 0,
                    'bb_upper': realtime_data['price'] * 1.1,
                    'bb_middle': realtime_data['price'],
                    'bb_lower': realtime_data['price'] * 0.9,
                    'volatility': 0.02
                }

            # 获取市场情绪
            market_sentiment = self.data_provider.get_market_sentiment(stock_code)

            # 整合所有数据
            analysis = {
                **realtime_data,
                **technical_indicators,
                **financial_data,
                'market_sentiment': market_sentiment,
                'analysis_time': datetime.now().isoformat()
            }

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing stock {stock_code}: {e}")
            return None

    def batch_analyze(self, stock_codes: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        批量分析股票

        Args:
            stock_codes: 股票代码列表

        Returns:
            分析结果字典 {code: analysis}
        """
        results = {}
        for code in stock_codes:
            analysis = self.analyze_stock(code)
            if analysis:
                results[code] = analysis

        logger.info(f"Batch analyzed {len(results)}/{len(stock_codes)} stocks")
        return results

    def find_opportunities(
        self,
        min_change_pct: float = -10,
        max_change_pct: float = 10,
        min_volume_ratio: float = 1.0,
        min_turnover_rate: float = 1.0
    ) -> List[Dict[str, Any]]:
        """
        寻找交易机会

        Args:
            min_change_pct: 最小涨跌幅
            max_change_pct: 最大涨跌幅
            min_volume_ratio: 最小量比
            min_turnover_rate: 最小换手率

        Returns:
            机会列表
        """
        try:
            # 获取股票列表
            stocks = self.data_provider.get_stock_list()

            opportunities = []
            for stock in stocks:
                # 基本筛选
                if not (min_change_pct <= stock['change_pct'] <= max_change_pct):
                    continue

                # 分析股票
                analysis = self.analyze_stock(stock['code'])
                if analysis is None:
                    continue

                # 高级筛选
                if analysis.get('volume_ratio', 0) < min_volume_ratio:
                    continue
                if analysis.get('turnover_rate', 0) < min_turnover_rate:
                    continue

                # 计算机会分数
                opportunity_score = self._calculate_opportunity_score(analysis)
                analysis['opportunity_score'] = opportunity_score

                opportunities.append(analysis)

            # 按机会分数排序
            opportunities.sort(key=lambda x: x['opportunity_score'], reverse=True)

            logger.info(f"Found {len(opportunities)} opportunities")
            return opportunities

        except Exception as e:
            logger.error(f"Error finding opportunities: {e}")
            return []

    def _calculate_opportunity_score(self, analysis: Dict[str, Any]) -> float:
        """
        计算机会分数 (0-100)

        Args:
            analysis: 分析数据

        Returns:
            机会分数
        """
        score = 50  # 基准分

        # 1. 技术面得分 (30分)
        price = analysis.get('price', 0)
        ma20 = analysis.get('ma20', price)
        rsi = analysis.get('rsi', 50)
        macd_hist = analysis.get('macd_hist', 0)

        # 价格与均线关系
        if price > ma20:
            score += 10
        else:
            score -= 5

        # RSI
        if 30 < rsi < 70:
            score += 10
        elif rsi < 30:
            score += 15  # 超卖
        elif rsi > 70:
            score -= 10  # 超买

        # MACD
        if macd_hist > 0:
            score += 10
        else:
            score -= 5

        # 2. 基本面得分 (20分)
        pe_ratio = analysis.get('pe_ratio', 30)
        roe = analysis.get('roe', 0.1)

        if 10 < pe_ratio < 30:
            score += 10
        elif pe_ratio < 10:
            score += 15
        else:
            score -= 10

        if roe > 0.15:
            score += 10
        elif roe > 0.10:
            score += 5

        # 3. 市场情绪得分 (20分)
        change_pct = analysis.get('change_pct', 0)
        volume_ratio = analysis.get('volume_ratio', 1)

        # 量价配合
        if change_pct > 2 and volume_ratio > 1.5:
            score += 15
        elif change_pct < -2 and volume_ratio > 1.5:
            score -= 10

        # 波动率
        volatility = analysis.get('volatility', 0.02)
        if volatility < 0.03:
            score += 5
        elif volatility > 0.05:
            score -= 10

        # 4. 流动性得分 (10分)
        turnover_rate = analysis.get('turnover_rate', 1)
        if turnover_rate > 3:
            score += 10
        elif turnover_rate > 1:
            score += 5

        # 限制在0-100范围
        score = max(0, min(100, score))

        return score

    def get_stock_summary(self, stock_code: str) -> str:
        """
        获取股票摘要信息

        Args:
            stock_code: 股票代码

        Returns:
            摘要信息
        """
        analysis = self.analyze_stock(stock_code)
        if analysis is None:
            return f"无法获取{stock_code}的数据"

        summary = f"""
【股票分析摘要】{analysis['code']} {analysis['name']}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 基本信息
  当前价格: {analysis['price']:.2f} 元
  涨跌幅: {analysis['change_pct']:.2f}%
  成交额: {analysis['amount']/100000000:.2f} 亿
  换手率: {analysis['turnover_rate']:.2f}%
  量比: {analysis.get('volume_ratio', 1):.2f}

📈 技术指标
  MA5/MA20/MA60: {analysis['ma5']:.2f} / {analysis['ma20']:.2f} / {analysis['ma60']:.2f}
  RSI: {analysis['rsi']:.2f}
  MACD: {analysis['macd']:.4f}
  波动率: {analysis['volatility']:.2%}

💰 财务指标
  市盈率(PE): {analysis['pe_ratio']:.2f}
  市净率(PB): {analysis['pb_ratio']:.2f}
  ROE: {analysis['roe']:.2%}

😊 市场情绪: {analysis['market_sentiment']:.2f} (0-1)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        """
        return summary.strip()

    def compare_stocks(self, stock_codes: List[str]) -> pd.DataFrame:
        """
        对比多只股票

        Args:
            stock_codes: 股票代码列表

        Returns:
            对比DataFrame
        """
        data = []
        for code in stock_codes:
            analysis = self.analyze_stock(code)
            if analysis:
                data.append({
                    '代码': code,
                    '名称': analysis.get('name', ''),
                    '价格': analysis.get('price', 0),
                    '涨跌幅': analysis.get('change_pct', 0),
                    '量比': analysis.get('volume_ratio', 1),
                    'RSI': analysis.get('rsi', 50),
                    'PE': analysis.get('pe_ratio', 30),
                    'ROE': analysis.get('roe', 0.1),
                    '机会分数': self._calculate_opportunity_score(analysis)
                })

        return pd.DataFrame(data)

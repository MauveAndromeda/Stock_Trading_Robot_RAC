"""
机构Agent实现
"""
import random
from typing import Dict, Any, List
from .base_agent import BaseAgent, AgentRole, AgentDecision, ActionType


class QuantitativeAgent(BaseAgent):
    """量化对冲基金Agent"""

    def __init__(self, agent_id: str, llm_client=None):
        super().__init__(
            agent_id=agent_id,
            agent_type="quantitative",
            role=AgentRole.INSTITUTIONAL,
            risk_tolerance=0.6,
            llm_client=llm_client
        )

    def get_personality_prompt(self) -> str:
        return """
        你是一个量化对冲基金的策略系统。特点：
        - 完全基于数据和模型决策
        - 关注统计套利机会
        - 快速进出，高频交易
        - 严格的风险控制
        - 情绪中性，不受市场情绪影响
        """

    def analyze(self, stock_data: Dict[str, Any]) -> AgentDecision:
        """量化策略：多因子模型"""
        stock_code = stock_data.get('code', 'UNKNOWN')

        # 多个量化因子
        factors = {}

        # 动量因子
        change_pct = stock_data.get('change_pct', 0)
        factors['momentum'] = 0.5 + change_pct * 0.02

        # 波动率因子
        volatility = stock_data.get('volatility', 0.02)
        factors['volatility'] = 0.7 if volatility < 0.03 else 0.3

        # 流动性因子
        volume_ratio = stock_data.get('volume_ratio', 1)
        factors['liquidity'] = min(volume_ratio / 3, 1.0) * 0.5 + 0.25

        # 估值因子
        pe_ratio = stock_data.get('pe_ratio', 30)
        factors['valuation'] = 0.8 if 10 < pe_ratio < 25 else 0.4

        # 技术因子
        rsi = stock_data.get('rsi', 50)
        factors['technical'] = (100 - abs(rsi - 50)) / 100

        # 加权计算
        weights = {
            'momentum': 0.25,
            'volatility': 0.15,
            'liquidity': 0.20,
            'valuation': 0.20,
            'technical': 0.20
        }

        confidence = self._calculate_confidence(factors, weights)

        # 量化策略决策
        if confidence > 0.70:
            action = ActionType.BUY
            reason = f"多因子模型评分{confidence:.3f}，超过买入阈值0.70，信号强度高"
        elif confidence < 0.30:
            action = ActionType.SELL
            reason = f"多因子模型评分{confidence:.3f}，低于卖出阈值0.30，信号弱"
        else:
            action = ActionType.HOLD
            reason = f"多因子模型评分{confidence:.3f}，在中性区间，暂不操作"

        decision = AgentDecision(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            stock_code=stock_code,
            action=action,
            confidence=confidence,
            reason=reason,
            risk_level="medium"
        )

        self.record_decision(decision)
        return decision


class ValueInvestorAgent(BaseAgent):
    """价值投资机构Agent"""

    def __init__(self, agent_id: str, llm_client=None):
        super().__init__(
            agent_id=agent_id,
            agent_type="value_investor",
            role=AgentRole.INSTITUTIONAL,
            risk_tolerance=0.3,
            llm_client=llm_client
        )

    def get_personality_prompt(self) -> str:
        return """
        你是一个专业的价值投资机构。特点：
        - 深度基本面研究
        - 长期持有优质公司
        - 注重安全边际
        - 逆向投资，在市场恐慌时买入
        - 不受短期波动影响
        """

    def analyze(self, stock_data: Dict[str, Any]) -> AgentDecision:
        """价值投资：深度基本面分析"""
        stock_code = stock_data.get('code', 'UNKNOWN')

        signals = {}

        # 估值指标
        pe_ratio = stock_data.get('pe_ratio', 30)
        pb_ratio = stock_data.get('pb_ratio', 3)
        roe = stock_data.get('roe', 0.10)

        # 市盈率评分（更专业的判断）
        if pe_ratio < 10:
            signals['pe'] = 0.9
        elif pe_ratio < 15:
            signals['pe'] = 0.75
        elif pe_ratio < 20:
            signals['pe'] = 0.6
        elif pe_ratio < 30:
            signals['pe'] = 0.4
        else:
            signals['pe'] = 0.2

        # 市净率评分
        if pb_ratio < 1.5:
            signals['pb'] = 0.85
        elif pb_ratio < 2.5:
            signals['pb'] = 0.65
        elif pb_ratio < 4:
            signals['pb'] = 0.45
        else:
            signals['pb'] = 0.25

        # ROE评分
        if roe > 0.20:
            signals['roe'] = 0.9
        elif roe > 0.15:
            signals['roe'] = 0.75
        elif roe > 0.10:
            signals['roe'] = 0.6
        else:
            signals['roe'] = 0.3

        # 价格位置（价值投资喜欢低位）
        change_pct = stock_data.get('change_pct', 0)
        if change_pct < -8:
            signals['price_position'] = 0.85  # 大跌后更有价值
        elif change_pct < -3:
            signals['price_position'] = 0.7
        elif change_pct < 0:
            signals['price_position'] = 0.6
        elif change_pct < 5:
            signals['price_position'] = 0.4
        else:
            signals['price_position'] = 0.2  # 大涨后价值降低

        confidence = self._calculate_confidence(
            signals,
            {'pe': 0.25, 'pb': 0.20, 'roe': 0.30, 'price_position': 0.25}
        )

        if confidence > 0.70:
            action = ActionType.BUY
            reason = f"优质标的：PE={pe_ratio:.2f}, PB={pb_ratio:.2f}, ROE={roe:.2%}，估值合理，具备投资价值"
        elif confidence < 0.35:
            action = ActionType.SELL
            reason = f"估值过高：PE={pe_ratio:.2f}, PB={pb_ratio:.2f}，超出合理范围，建议减持"
        else:
            action = ActionType.HOLD
            reason = f"继续观察：基本面中等，PE={pe_ratio:.2f}，暂时持有"

        decision = AgentDecision(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            stock_code=stock_code,
            action=action,
            confidence=confidence,
            reason=reason,
            risk_level="low"
        )

        self.record_decision(decision)
        return decision


class TrendFollowerAgent(BaseAgent):
    """趋势跟踪机构Agent"""

    def __init__(self, agent_id: str, llm_client=None):
        super().__init__(
            agent_id=agent_id,
            agent_type="trend_follower",
            role=AgentRole.INSTITUTIONAL,
            risk_tolerance=0.5,
            llm_client=llm_client
        )

    def get_personality_prompt(self) -> str:
        return """
        你是一个趋势跟踪型机构投资者。特点：
        - 专注中长期趋势
        - 使用多周期技术分析
        - 顺势而为，不逆势操作
        - 严格止损，让利润奔跑
        - 系统化交易策略
        """

    def analyze(self, stock_data: Dict[str, Any]) -> AgentDecision:
        """趋势跟踪：多周期技术分析"""
        stock_code = stock_data.get('code', 'UNKNOWN')
        price = stock_data.get('price', 0)

        signals = {}

        # 多周期均线
        ma5 = stock_data.get('ma5', price)
        ma20 = stock_data.get('ma20', price)
        ma60 = stock_data.get('ma60', price)

        # 短期趋势
        if price > ma5 > ma20:
            signals['short_trend'] = 0.85
        elif price < ma5 < ma20:
            signals['short_trend'] = 0.15
        else:
            signals['short_trend'] = 0.5

        # 中期趋势
        if ma20 > ma60:
            signals['mid_trend'] = 0.75
        elif ma20 < ma60:
            signals['mid_trend'] = 0.25
        else:
            signals['mid_trend'] = 0.5

        # 趋势强度
        change_pct = stock_data.get('change_pct', 0)
        volume_ratio = stock_data.get('volume_ratio', 1)

        if change_pct > 3 and volume_ratio > 1.5:
            signals['strength'] = 0.85  # 强势上涨
        elif change_pct < -3 and volume_ratio > 1.5:
            signals['strength'] = 0.15  # 强势下跌
        else:
            signals['strength'] = 0.5

        # MACD
        macd = stock_data.get('macd', 0)
        signals['macd'] = 0.7 if macd > 0 else 0.3

        confidence = self._calculate_confidence(
            signals,
            {'short_trend': 0.30, 'mid_trend': 0.30, 'strength': 0.25, 'macd': 0.15}
        )

        if confidence > 0.68:
            action = ActionType.BUY
            reason = f"趋势向上：短期趋势{signals['short_trend']:.2f}，中期趋势{signals['mid_trend']:.2f}，顺势做多"
        elif confidence < 0.32:
            action = ActionType.SELL
            reason = f"趋势向下：趋势指标转弱，止损离场"
        else:
            action = ActionType.HOLD
            reason = "趋势不明朗，等待明确信号"

        decision = AgentDecision(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            stock_code=stock_code,
            action=action,
            confidence=confidence,
            reason=reason,
            risk_level="medium"
        )

        self.record_decision(decision)
        return decision


class HighFrequencyAgent(BaseAgent):
    """高频交易机构Agent"""

    def __init__(self, agent_id: str, llm_client=None):
        super().__init__(
            agent_id=agent_id,
            agent_type="high_frequency",
            role=AgentRole.INSTITUTIONAL,
            risk_tolerance=0.7,
            llm_client=llm_client
        )

    def get_personality_prompt(self) -> str:
        return """
        你是一个高频交易系统。特点：
        - 微秒级决策
        - 捕捉短期价格波动
        - 大量小额交易
        - 极低的持仓时间
        - 依赖流动性和订单流
        """

    def analyze(self, stock_data: Dict[str, Any]) -> AgentDecision:
        """高频交易：短期价格波动和流动性"""
        stock_code = stock_data.get('code', 'UNKNOWN')

        signals = {}

        # 短期波动
        price_volatility = stock_data.get('price_volatility_1min', 0)
        signals['volatility'] = min(price_volatility * 20, 0.9) if price_volatility > 0 else 0.3

        # 流动性
        volume_ratio = stock_data.get('volume_ratio', 1)
        turnover_rate = stock_data.get('turnover_rate', 1)
        signals['liquidity'] = min((volume_ratio * turnover_rate) / 10, 0.95)

        # 买卖盘口
        bid_ask_ratio = stock_data.get('bid_ask_ratio', 1)  # 买盘/卖盘
        if bid_ask_ratio > 1.2:
            signals['order_flow'] = 0.75
        elif bid_ask_ratio < 0.8:
            signals['order_flow'] = 0.25
        else:
            signals['order_flow'] = 0.5

        # 短期动量
        change_pct = stock_data.get('change_pct', 0)
        signals['momentum'] = 0.5 + change_pct * 0.1

        confidence = self._calculate_confidence(
            signals,
            {'volatility': 0.20, 'liquidity': 0.35, 'order_flow': 0.30, 'momentum': 0.15}
        )

        # 高频交易更激进
        if confidence > 0.60:
            action = ActionType.BUY
            reason = f"高频信号：流动性充足，买盘占优，快速进场"
        elif confidence < 0.40:
            action = ActionType.SELL
            reason = f"高频信号：卖盘压力大，快速出场"
        else:
            action = ActionType.HOLD
            reason = "暂无高频交易机会"

        decision = AgentDecision(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            stock_code=stock_code,
            action=action,
            confidence=confidence,
            reason=reason,
            risk_level="high"
        )

        self.record_decision(decision)
        return decision


class IndexFundAgent(BaseAgent):
    """指数基金Agent"""

    def __init__(self, agent_id: str, llm_client=None):
        super().__init__(
            agent_id=agent_id,
            agent_type="index_fund",
            role=AgentRole.INSTITUTIONAL,
            risk_tolerance=0.2,
            llm_client=llm_client
        )

    def get_personality_prompt(self) -> str:
        return """
        你是一个指数基金管理系统。特点：
        - 被动投资策略
        - 跟踪指数成分股
        - 低换手率
        - 长期持有
        - 主要进行再平衡操作
        """

    def analyze(self, stock_data: Dict[str, Any]) -> AgentDecision:
        """指数基金：跟踪指数，被动调整"""
        stock_code = stock_data.get('code', 'UNKNOWN')

        # 指数基金主要看是否偏离指数权重
        is_index_component = stock_data.get('is_index_component', False)
        weight_deviation = stock_data.get('weight_deviation', 0)  # 与指数权重的偏离度

        signals = {}

        if is_index_component:
            # 权重偏离
            if weight_deviation < -0.05:  # 持仓过低
                signals['rebalance'] = 0.8
            elif weight_deviation > 0.05:  # 持仓过高
                signals['rebalance'] = 0.2
            else:
                signals['rebalance'] = 0.5
        else:
            signals['rebalance'] = 0.3  # 非成分股，倾向于不持有

        # 市场波动（大幅波动时需要再平衡）
        change_pct = stock_data.get('change_pct', 0)
        if abs(change_pct) > 5:
            signals['volatility'] = 0.7
        else:
            signals['volatility'] = 0.4

        confidence = self._calculate_confidence(
            signals,
            {'rebalance': 0.80, 'volatility': 0.20}
        )

        if is_index_component:
            if confidence > 0.65:
                action = ActionType.BUY
                reason = f"指数成分股，当前持仓权重偏低{weight_deviation:.2%}，需要买入再平衡"
            elif confidence < 0.35:
                action = ActionType.SELL
                reason = f"指数成分股，当前持仓权重偏高{weight_deviation:.2%}，需要卖出再平衡"
            else:
                action = ActionType.HOLD
                reason = "持仓权重接近目标，暂不调整"
        else:
            action = ActionType.HOLD
            reason = "非指数成分股，不在投资范围"

        decision = AgentDecision(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            stock_code=stock_code,
            action=action,
            confidence=confidence,
            reason=reason,
            risk_level="low"
        )

        self.record_decision(decision)
        return decision


class InstitutionalAgentFactory:
    """机构Agent工厂类"""

    AGENT_TYPES = {
        'quantitative': QuantitativeAgent,
        'value_investor': ValueInvestorAgent,
        'trend_follower': TrendFollowerAgent,
        'high_frequency': HighFrequencyAgent,
        'index_fund': IndexFundAgent
    }

    @classmethod
    def create_agents(cls, config: Dict[str, int], llm_client=None) -> List[BaseAgent]:
        """
        批量创建机构Agent

        Args:
            config: Agent类型和数量配置
            llm_client: LLM客户端

        Returns:
            Agent列表
        """
        agents = []
        for agent_type, count in config.items():
            if agent_type not in cls.AGENT_TYPES:
                continue

            agent_class = cls.AGENT_TYPES[agent_type]
            for i in range(count):
                agent_id = f"institutional_{agent_type}_{i+1}"
                agent = agent_class(agent_id, llm_client)
                agents.append(agent)

        return agents

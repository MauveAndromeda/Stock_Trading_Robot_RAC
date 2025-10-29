"""
散户Agent实现
"""
import random
from typing import Dict, Any, List
from .base_agent import BaseAgent, AgentRole, AgentDecision, ActionType


class MomentumChaserAgent(BaseAgent):
    """追涨杀跌型散户"""

    def __init__(self, agent_id: str, llm_client=None):
        super().__init__(
            agent_id=agent_id,
            agent_type="momentum_chaser",
            role=AgentRole.RETAIL,
            risk_tolerance=0.8,
            llm_client=llm_client
        )

    def get_personality_prompt(self) -> str:
        return """
        你是一个追涨杀跌型散户投资者。特点：
        - 看到股票大涨就想买入，害怕错过机会
        - 看到股票下跌就恐慌卖出，害怕亏损扩大
        - 经常追高买入，低位卖出
        - 容易被市场情绪影响
        - 喜欢看涨幅榜，追热门股票
        """

    def analyze(self, stock_data: Dict[str, Any]) -> AgentDecision:
        """
        分析股票数据并做出决策
        追涨杀跌型主要看价格变化和成交量
        """
        stock_code = stock_data.get('code', 'UNKNOWN')
        price = stock_data.get('price', 0)
        change_pct = stock_data.get('change_pct', 0)
        volume_ratio = stock_data.get('volume_ratio', 1)

        # 追涨杀跌逻辑
        signals = {}

        # 看涨幅
        if change_pct > 5:
            signals['momentum'] = 0.9  # 大涨，想买
        elif change_pct > 2:
            signals['momentum'] = 0.7
        elif change_pct < -5:
            signals['momentum'] = 0.1  # 大跌，想卖
        elif change_pct < -2:
            signals['momentum'] = 0.3
        else:
            signals['momentum'] = 0.5

        # 看成交量
        if volume_ratio > 2:
            signals['volume'] = 0.8  # 放量
        elif volume_ratio > 1.5:
            signals['volume'] = 0.6
        else:
            signals['volume'] = 0.4

        confidence = self._calculate_confidence(signals)

        # 决定行动
        if confidence > 0.7:
            action = ActionType.BUY
            reason = f"股票涨了{change_pct:.2f}%，成交量放大{volume_ratio:.2f}倍，怕错过机会！"
        elif confidence < 0.3:
            action = ActionType.SELL
            reason = f"股票跌了{abs(change_pct):.2f}%，太可怕了，赶紧卖出止损！"
        else:
            action = ActionType.HOLD
            reason = f"涨跌幅{change_pct:.2f}%，不太明显，先观望一下"

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


class PanicSellerAgent(BaseAgent):
    """恐慌型散户"""

    def __init__(self, agent_id: str, llm_client=None):
        super().__init__(
            agent_id=agent_id,
            agent_type="panic_seller",
            role=AgentRole.RETAIL,
            risk_tolerance=0.2,
            llm_client=llm_client
        )

    def get_personality_prompt(self) -> str:
        return """
        你是一个恐慌型散户投资者。特点：
        - 极度厌恶风险和亏损
        - 市场稍有风吹草动就想卖出
        - 容易被负面新闻影响
        - 害怕套牢，宁愿小赚就跑
        - 经常在底部卖出
        """

    def analyze(self, stock_data: Dict[str, Any]) -> AgentDecision:
        """恐慌型主要关注下跌风险"""
        stock_code = stock_data.get('code', 'UNKNOWN')
        change_pct = stock_data.get('change_pct', 0)
        turnover_rate = stock_data.get('turnover_rate', 0)

        signals = {}

        # 对下跌极度敏感
        if change_pct < -3:
            signals['panic'] = 0.1  # 大跌，极度恐慌
        elif change_pct < -1:
            signals['panic'] = 0.3
        elif change_pct < 0:
            signals['panic'] = 0.4
        elif change_pct > 3:
            signals['panic'] = 0.9  # 涨太多，害怕回调
        else:
            signals['panic'] = 0.5

        # 换手率
        if turnover_rate > 10:
            signals['liquidity'] = 0.3  # 换手率太高，担心
        else:
            signals['liquidity'] = 0.6

        confidence = self._calculate_confidence(signals, {'panic': 0.7, 'liquidity': 0.3})

        if confidence > 0.7:
            action = ActionType.HOLD
            reason = f"涨了{change_pct:.2f}%，但我怕涨太高会回调，不敢买"
        elif confidence < 0.35:
            action = ActionType.SELL
            reason = f"跌了{abs(change_pct):.2f}%，太恐怖了！赶紧跑！"
        else:
            action = ActionType.HOLD
            reason = "心里很慌，不敢轻举妄动"

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


class HerdFollowerAgent(BaseAgent):
    """跟风型散户"""

    def __init__(self, agent_id: str, llm_client=None):
        super().__init__(
            agent_id=agent_id,
            agent_type="herd_follower",
            role=AgentRole.RETAIL,
            risk_tolerance=0.6,
            llm_client=llm_client
        )
        self.follow_probability = 0.8  # 跟随大众的概率

    def get_personality_prompt(self) -> str:
        return """
        你是一个跟风型散户投资者。特点：
        - 喜欢听消息、跟风买卖
        - 看到别人买什么就跟着买
        - 热衷于讨论股票论坛和群聊
        - 缺乏独立思考能力
        - 经常成为最后接盘的人
        """

    def analyze(self, stock_data: Dict[str, Any]) -> AgentDecision:
        """跟风型主要看市场情绪和热度"""
        stock_code = stock_data.get('code', 'UNKNOWN')
        change_pct = stock_data.get('change_pct', 0)
        volume_ratio = stock_data.get('volume_ratio', 1)
        market_sentiment = stock_data.get('market_sentiment', 0.5)  # 0-1

        signals = {}

        # 市场情绪
        signals['sentiment'] = market_sentiment

        # 成交量（认为成交量大=很多人在买）
        if volume_ratio > 2:
            signals['popularity'] = 0.8
        elif volume_ratio > 1.5:
            signals['popularity'] = 0.6
        else:
            signals['popularity'] = 0.4

        # 涨跌幅
        if change_pct > 0:
            signals['momentum'] = 0.6 + change_pct * 0.05
        else:
            signals['momentum'] = 0.4 + change_pct * 0.05

        confidence = self._calculate_confidence(
            signals,
            {'sentiment': 0.5, 'popularity': 0.3, 'momentum': 0.2}
        )

        # 跟风决策
        if confidence > 0.65 and random.random() < self.follow_probability:
            action = ActionType.BUY
            reason = f"好多人都在买这只股票，成交量是平时的{volume_ratio:.2f}倍，我也跟着买点"
        elif confidence < 0.35:
            action = ActionType.SELL
            reason = "看论坛上大家都在卖，我也卖了吧"
        else:
            action = ActionType.HOLD
            reason = "市场没什么热度，再看看"

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


class ValueSeekerAgent(BaseAgent):
    """价值型散户"""

    def __init__(self, agent_id: str, llm_client=None):
        super().__init__(
            agent_id=agent_id,
            agent_type="value_seeker",
            role=AgentRole.RETAIL,
            risk_tolerance=0.4,
            llm_client=llm_client
        )

    def get_personality_prompt(self) -> str:
        return """
        你是一个价值型散户投资者。特点：
        - 关注公司基本面
        - 喜欢买"便宜"的股票
        - 但缺乏专业分析能力
        - 容易被简单指标误导（如市盈率）
        - 持股时间相对较长
        """

    def analyze(self, stock_data: Dict[str, Any]) -> AgentDecision:
        """价值型关注基本面指标"""
        stock_code = stock_data.get('code', 'UNKNOWN')
        pe_ratio = stock_data.get('pe_ratio', 30)
        pb_ratio = stock_data.get('pb_ratio', 3)
        change_pct = stock_data.get('change_pct', 0)

        signals = {}

        # 市盈率（简单判断）
        if pe_ratio < 15:
            signals['pe'] = 0.8  # 认为便宜
        elif pe_ratio < 25:
            signals['pe'] = 0.6
        elif pe_ratio < 40:
            signals['pe'] = 0.4
        else:
            signals['pe'] = 0.2  # 认为贵

        # 市净率
        if pb_ratio < 2:
            signals['pb'] = 0.7
        elif pb_ratio < 4:
            signals['pb'] = 0.5
        else:
            signals['pb'] = 0.3

        # 价格变动（价值投资者更喜欢下跌买入）
        if change_pct < -5:
            signals['price'] = 0.8  # 跌了，更便宜
        elif change_pct < 0:
            signals['price'] = 0.6
        else:
            signals['price'] = 0.4

        confidence = self._calculate_confidence(
            signals,
            {'pe': 0.4, 'pb': 0.3, 'price': 0.3}
        )

        if confidence > 0.65:
            action = ActionType.BUY
            reason = f"市盈率{pe_ratio:.2f}，市净率{pb_ratio:.2f}，估值合理，可以买入"
        elif confidence < 0.35:
            action = ActionType.SELL
            reason = f"市盈率{pe_ratio:.2f}太高了，估值太贵，应该卖出"
        else:
            action = ActionType.HOLD
            reason = "估值一般，继续持有观察"

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


class TechnicalTraderAgent(BaseAgent):
    """技术型散户"""

    def __init__(self, agent_id: str, llm_client=None):
        super().__init__(
            agent_id=agent_id,
            agent_type="technical_trader",
            role=AgentRole.RETAIL,
            risk_tolerance=0.7,
            llm_client=llm_client
        )

    def get_personality_prompt(self) -> str:
        return """
        你是一个技术型散户投资者。特点：
        - 依赖技术指标（MA、MACD、RSI等）
        - 但只会用简单的指标
        - 喜欢画线、找形态
        - 经常被假突破欺骗
        - 有时会过度交易
        """

    def analyze(self, stock_data: Dict[str, Any]) -> AgentDecision:
        """技术型主要看技术指标"""
        stock_code = stock_data.get('code', 'UNKNOWN')
        ma5 = stock_data.get('ma5', 0)
        ma20 = stock_data.get('ma20', 0)
        rsi = stock_data.get('rsi', 50)
        macd = stock_data.get('macd', 0)
        price = stock_data.get('price', 0)

        signals = {}

        # 均线
        if price > ma5 > ma20:
            signals['ma'] = 0.8  # 多头排列
        elif price < ma5 < ma20:
            signals['ma'] = 0.2  # 空头排列
        else:
            signals['ma'] = 0.5

        # RSI
        if rsi < 30:
            signals['rsi'] = 0.8  # 超卖
        elif rsi > 70:
            signals['rsi'] = 0.2  # 超买
        else:
            signals['rsi'] = 0.5

        # MACD
        if macd > 0:
            signals['macd'] = 0.7
        else:
            signals['macd'] = 0.3

        confidence = self._calculate_confidence(
            signals,
            {'ma': 0.4, 'rsi': 0.3, 'macd': 0.3}
        )

        if confidence > 0.65:
            action = ActionType.BUY
            reason = f"技术指标看涨：均线多头排列，RSI={rsi:.2f}，MACD金叉"
        elif confidence < 0.35:
            action = ActionType.SELL
            reason = f"技术指标看跌：均线空头排列，RSI={rsi:.2f}，MACD死叉"
        else:
            action = ActionType.HOLD
            reason = "技术指标不明朗，等待更好的信号"

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


class RetailAgentFactory:
    """散户Agent工厂类"""

    AGENT_TYPES = {
        'momentum_chaser': MomentumChaserAgent,
        'panic_seller': PanicSellerAgent,
        'herd_follower': HerdFollowerAgent,
        'value_seeker': ValueSeekerAgent,
        'technical_trader': TechnicalTraderAgent
    }

    @classmethod
    def create_agents(cls, config: Dict[str, int], llm_client=None) -> List[BaseAgent]:
        """
        批量创建散户Agent

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
                agent_id = f"retail_{agent_type}_{i+1}"
                agent = agent_class(agent_id, llm_client)
                agents.append(agent)

        return agents

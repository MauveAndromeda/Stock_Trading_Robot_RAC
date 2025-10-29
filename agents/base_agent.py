"""
Agent基类
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Optional, List
from datetime import datetime
import random


class AgentRole(Enum):
    """Agent角色类型"""
    RETAIL = "retail"  # 散户
    INSTITUTIONAL = "institutional"  # 机构
    EXPERT = "expert"  # 专家


class ActionType(Enum):
    """行动类型"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass
class AgentDecision:
    """Agent决策结果"""
    agent_id: str
    agent_type: str
    stock_code: str
    action: ActionType
    confidence: float  # 0-1之间的置信度
    reason: str
    suggested_price: Optional[float] = None
    suggested_quantity: Optional[int] = None
    risk_level: str = "medium"  # low, medium, high
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'agent_id': self.agent_id,
            'agent_type': self.agent_type,
            'stock_code': self.stock_code,
            'action': self.action.value,
            'confidence': self.confidence,
            'reason': self.reason,
            'suggested_price': self.suggested_price,
            'suggested_quantity': self.suggested_quantity,
            'risk_level': self.risk_level,
            'timestamp': self.timestamp.isoformat()
        }


class BaseAgent(ABC):
    """Agent基类"""

    def __init__(
        self,
        agent_id: str,
        agent_type: str,
        role: AgentRole,
        risk_tolerance: float = 0.5,
        llm_client: Optional[Any] = None
    ):
        """
        初始化Agent

        Args:
            agent_id: Agent唯一标识
            agent_type: Agent类型
            role: Agent角色
            risk_tolerance: 风险承受能力 (0-1)
            llm_client: LLM客户端
        """
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.role = role
        self.risk_tolerance = risk_tolerance
        self.llm_client = llm_client
        self.decision_history: List[AgentDecision] = []

    @abstractmethod
    def analyze(self, stock_data: Dict[str, Any]) -> AgentDecision:
        """
        分析股票并做出决策

        Args:
            stock_data: 股票数据

        Returns:
            AgentDecision: 决策结果
        """
        pass

    @abstractmethod
    def get_personality_prompt(self) -> str:
        """
        获取Agent的个性化提示词

        Returns:
            提示词字符串
        """
        pass

    def _calculate_confidence(
        self,
        signals: Dict[str, float],
        weights: Optional[Dict[str, float]] = None
    ) -> float:
        """
        计算置信度

        Args:
            signals: 各种信号的得分
            weights: 信号权重

        Returns:
            置信度 (0-1)
        """
        if weights is None:
            weights = {k: 1.0 for k in signals.keys()}

        total_weight = sum(weights.values())
        if total_weight == 0:
            return 0.5

        weighted_sum = sum(signals[k] * weights.get(k, 0) for k in signals.keys())
        confidence = weighted_sum / total_weight

        # 添加一些随机性（模拟人的不确定性）
        noise = random.gauss(0, 0.05)
        confidence = max(0.0, min(1.0, confidence + noise))

        return confidence

    def _call_llm(self, prompt: str, temperature: float = 0.7) -> str:
        """
        调用LLM

        Args:
            prompt: 提示词
            temperature: 温度参数

        Returns:
            LLM响应
        """
        if self.llm_client is None:
            return "LLM client not configured"

        try:
            # 这里需要根据实际使用的LLM客户端实现
            # 示例使用OpenAI API
            response = self.llm_client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"LLM error: {str(e)}"

    def record_decision(self, decision: AgentDecision) -> None:
        """
        记录决策历史

        Args:
            decision: 决策结果
        """
        self.decision_history.append(decision)

    def get_stats(self) -> Dict[str, Any]:
        """
        获取Agent统计信息

        Returns:
            统计信息
        """
        if not self.decision_history:
            return {
                'total_decisions': 0,
                'buy_count': 0,
                'sell_count': 0,
                'hold_count': 0,
                'avg_confidence': 0
            }

        total = len(self.decision_history)
        buy_count = sum(1 for d in self.decision_history if d.action == ActionType.BUY)
        sell_count = sum(1 for d in self.decision_history if d.action == ActionType.SELL)
        hold_count = sum(1 for d in self.decision_history if d.action == ActionType.HOLD)
        avg_confidence = sum(d.confidence for d in self.decision_history) / total

        return {
            'agent_id': self.agent_id,
            'agent_type': self.agent_type,
            'role': self.role.value,
            'total_decisions': total,
            'buy_count': buy_count,
            'sell_count': sell_count,
            'hold_count': hold_count,
            'avg_confidence': round(avg_confidence, 3)
        }

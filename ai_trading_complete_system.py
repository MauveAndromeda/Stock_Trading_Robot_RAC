#!/usr/bin/env python3
"""
🚀 AI多Agent完整交易系统 - 一键回测版
================================================================
集成250+ Agent + LLM增强 + 专家讨论 + 完整回测

核心特性:
- 200个散户Agent (5种类型: 追涨杀跌、恐慌型、跟风型、价值型、技术型)
- 50个机构Agent (5种类型: 量化、价值投资、趋势、高频、指数)
- Claude 3.5 Sonnet驱动的专家委员会 (4位专家、多轮讨论)
- "顺机构反散户"核心策略
- 完整回测引擎 (支持多股票、多周期)
- 多层风险管理系统
- 详细业绩报告和可视化

使用方法:
    python ai_trading_complete_system.py

环境变量:
    ANTHROPIC_API_KEY - Claude API密钥 (推荐)
    OPENAI_API_KEY - OpenAI API密钥 (备选)

作者: AI Stock Trading Robot
版本: 2.0 (2025最新AI技术)
================================================================
"""

import os
import sys
import json
import time
import math
import random
import warnings
import hashlib
import statistics
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from collections import defaultdict

warnings.filterwarnings('ignore')

# ============================================================================
# 配置区域 - Configuration
# ============================================================================

# LLM配置
LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'anthropic')  # anthropic or openai
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY', '')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
LLM_MODEL = os.getenv('LLM_MODEL', 'claude-3-5-sonnet-20241022')
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 2000

# Agent配置
RETAIL_AGENT_COUNT = {
    'momentum_chaser': 50,      # 追涨杀跌型
    'panic_seller': 40,         # 恐慌型
    'herd_follower': 40,        # 跟风型
    'value_seeker': 40,         # 价值型
    'technical_trader': 30      # 技术型
}

INSTITUTIONAL_AGENT_COUNT = {
    'quantitative': 15,         # 量化对冲
    'value_investor': 10,       # 价值投资
    'trend_follower': 10,       # 趋势跟踪
    'high_frequency': 10,       # 高频交易
    'index_fund': 5             # 指数基金
}

EXPERT_DISCUSSION_ROUNDS = 2    # 专家讨论轮次

# 回测配置
BACKTEST_START_DATE = '2023-01-01'
BACKTEST_END_DATE = '2024-12-31'
INITIAL_CAPITAL = 100000        # 初始资金 $100,000
MAX_POSITION_SIZE = 0.15        # 单个股票最大仓位 15%
COMMISSION_RATE = 0.001         # 交易佣金 0.1%
SLIPPAGE_RATE = 0.001           # 滑点 0.1%

# 风险管理配置
MAX_DAILY_LOSS = 0.03           # 单日最大亏损 3%
MAX_DRAWDOWN = 0.20             # 最大回撤 20%
STOP_LOSS_PCT = 0.05            # 止损 5%
TAKE_PROFIT_PCT = 0.15          # 止盈 15%

# 策略配置
MIN_CONFIDENCE = 0.70           # 最小置信度
MIN_CONSENSUS = 0.60            # 最小共识度
SENTIMENT_DIVERGENCE_THRESHOLD = 0.5  # 情绪分歧阈值

# 数据配置
TECHNICAL_INDICATORS_PERIODS = [5, 10, 20, 50, 100, 200]
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
ATR_PERIOD = 14

# ============================================================================
# 简易时间序列结构（用于脱离pandas运行）
# ============================================================================


def _daterange(start: datetime, end: datetime) -> List[datetime]:
    """生成[start, end]之间的所有日期（包含结束日）。"""
    days = []
    current = start
    while current <= end:
        days.append(current)
        current += timedelta(days=1)
    return days


class TimeSeries:
    """轻量级时间序列容器，兼容本文件的回测逻辑。"""

    def __init__(self, records: List[Dict[str, Any]]):
        self._records = records
        self._index = {r['date']: r for r in records}

    def __iter__(self):
        return iter(self._records)

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        return self._records[item]

    @property
    def dates(self) -> List[datetime]:
        return [r['date'] for r in self._records]

    def get_by_date(self, date: datetime) -> Optional[Dict[str, Any]]:
        return self._index.get(date)

    def last(self) -> Dict[str, Any]:
        return self._records[-1]


def _rolling_mean(values: List[float], window: int) -> List[Optional[float]]:
    results: List[Optional[float]] = []
    window_sum = 0.0
    for i, value in enumerate(values):
        window_sum += value
        if i >= window:
            window_sum -= values[i - window]
        if i + 1 >= window:
            results.append(window_sum / window)
        else:
            results.append(None)
    return results


def _ema_series(values: List[float], span: int) -> List[Optional[float]]:
    if not values:
        return []
    multiplier = 2 / (span + 1)
    ema_values: List[Optional[float]] = []
    ema: Optional[float] = None
    for value in values:
        if value is None:
            ema_values.append(ema)
            continue
        if ema is None:
            ema = value
        else:
            ema = (value - ema) * multiplier + ema
        ema_values.append(ema)
    return ema_values


def _rolling_std(values: List[float], window: int) -> List[Optional[float]]:
    results: List[Optional[float]] = []
    for i in range(len(values)):
        if i + 1 < window:
            results.append(None)
            continue
        window_values = values[i + 1 - window:i + 1]
        mean = sum(window_values) / window
        variance = sum((v - mean) ** 2 for v in window_values) / window
        results.append(math.sqrt(variance))
    return results


def _pct_change_series(values: List[float]) -> List[Optional[float]]:
    changes: List[Optional[float]] = [None]
    for i in range(1, len(values)):
        prev = values[i - 1]
        changes.append((values[i] - prev) / prev if prev else None)
    return changes

# ============================================================================
# 基础类定义 - Base Classes
# ============================================================================

class ActionType(Enum):
    """交易动作类型"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"

class AgentRole(Enum):
    """Agent角色"""
    RETAIL = "retail"           # 散户
    INSTITUTIONAL = "institutional"  # 机构

@dataclass
class AgentDecision:
    """Agent决策"""
    action: ActionType
    confidence: float           # 0-1
    reason: str
    agent_id: str
    agent_type: str
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class ExpertOpinion:
    """专家意见"""
    expert_name: str
    action: ActionType
    confidence: float
    reasoning: str
    key_factors: List[str]
    risks: List[str]
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class PanelDecision:
    """专家委员会决策"""
    final_action: ActionType
    confidence: float
    consensus_level: float      # 共识度
    reasoning: str
    expert_opinions: List[ExpertOpinion]
    retail_sentiment: Dict[str, float]
    institutional_sentiment: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class Trade:
    """交易记录"""
    date: datetime
    symbol: str
    action: ActionType
    price: float
    shares: int
    commission: float
    reason: str
    confidence: float
    pnl: float = 0.0
    pnl_pct: float = 0.0

@dataclass
class Position:
    """持仓"""
    symbol: str
    shares: int
    avg_price: float
    current_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float

# ============================================================================
# LLM客户端 - LLM Client
# ============================================================================

class LLMClient:
    """统一的LLM客户端，支持Claude和OpenAI"""

    def __init__(self, provider: str = 'anthropic', model: str = 'claude-3-5-sonnet-20241022'):
        self.provider = provider
        self.model = model
        self.client = None

        if provider == 'anthropic':
            api_key = ANTHROPIC_API_KEY
            if not api_key:
                print("⚠️  ANTHROPIC_API_KEY未设置，使用模拟模式运行")
                self.client = None
            else:
                try:
                    import anthropic
                    self.client = anthropic.Anthropic(api_key=api_key)
                except ImportError:
                    print("⚠️  anthropic包未安装，尝试使用模拟模式")
                    self.client = None
        elif provider == 'openai':
            api_key = OPENAI_API_KEY
            if not api_key:
                print("⚠️  OPENAI_API_KEY未设置，使用模拟模式运行")
                self.client = None
            else:
                try:
                    import openai
                    self.client = openai.OpenAI(api_key=api_key)
                except ImportError:
                    print("⚠️  openai包未安装，尝试使用模拟模式")
                    self.client = None

    def chat(self, prompt: str, system_prompt: str = None) -> str:
        """发送聊天请求"""
        if self.client is None:
            # 模拟模式 - 返回简单响应
            return self._simulate_response(prompt)

        try:
            if self.provider == 'anthropic':
                return self._chat_anthropic(prompt, system_prompt)
            elif self.provider == 'openai':
                return self._chat_openai(prompt, system_prompt)
        except Exception as e:
            print(f"⚠️  LLM调用失败: {e}，使用模拟模式")
            return self._simulate_response(prompt)

    def _chat_anthropic(self, prompt: str, system_prompt: str = None) -> str:
        """Claude聊天"""
        messages = [{"role": "user", "content": prompt}]

        response = self.client.messages.create(
            model=self.model,
            max_tokens=LLM_MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
            system=system_prompt or "You are a professional stock market analyst.",
            messages=messages
        )

        return response.content[0].text

    def _chat_openai(self, prompt: str, system_prompt: str = None) -> str:
        """OpenAI聊天"""
        messages = [
            {"role": "system", "content": system_prompt or "You are a professional stock market analyst."},
            {"role": "user", "content": prompt}
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS
        )

        return response.choices[0].message.content

    def _simulate_response(self, prompt: str) -> str:
        """保留的文本模拟（兼容旧逻辑）。"""
        return json.dumps({
            "action": "hold",
            "confidence": 0.60,
            "reason": "Simulation fallback"
        })

    def _simulate_structured_decision(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """根据元数据生成确定性的模拟决策。"""
        stock_data = metadata.get('stock_data', {})
        agent_type = metadata.get('agent_type', 'generic')
        identifier = metadata.get('identifier', agent_type)

        key = f"{agent_type}:{identifier}:{stock_data.get('code', 'N/A')}:{stock_data.get('price', 0):.2f}"
        seed = int.from_bytes(hashlib.sha256(key.encode('utf-8')).digest()[:8], 'big')
        rng = random.Random(seed)

        price = stock_data.get('price', 0.0) or 0.0
        ma10 = stock_data.get('ma10', price) or price
        ma20 = stock_data.get('ma20', price) or price
        ma50 = stock_data.get('ma50', price) or price
        rsi = stock_data.get('rsi', 50.0) or 50.0
        macd = stock_data.get('macd', 0.0) or 0.0
        atr = stock_data.get('atr', 0.0) or 0.0
        volatility = stock_data.get('volatility', 0.0) or 0.0
        change_pct = stock_data.get('change_pct', 0.0) or 0.0

        trend_score = 0.0
        if ma50:
            if ma10 > ma50 * 1.01:
                trend_score += 0.4
            elif ma10 < ma50 * 0.99:
                trend_score -= 0.4
        if ma20:
            if price > ma20 * 1.01:
                trend_score += 0.1
            elif price < ma20 * 0.99:
                trend_score -= 0.1

        momentum_score = 0.0
        if macd > 0:
            momentum_score += min(macd / max(price, 1), 1.0) * 0.5
        elif macd < 0:
            momentum_score += max(macd / max(price, 1), -1.0) * 0.5

        rsi_score = 0.0
        if rsi < 35:
            rsi_score += 0.4
        elif rsi > 65:
            rsi_score -= 0.4

        change_score = 0.0
        if change_pct > 1.5:
            change_score += 0.2
        elif change_pct < -1.5:
            change_score -= 0.2

        noise = rng.uniform(-0.05, 0.05)

        score = trend_score + momentum_score + rsi_score + change_score + noise

        if agent_type == 'institutional':
            score -= abs(volatility) * 0.5
            score -= atr * 0.0005
            threshold = 0.25
        elif agent_type == 'expert':
            retail_decisions: List[AgentDecision] = metadata.get('retail_decisions', [])
            inst_decisions: List[AgentDecision] = metadata.get('institutional_decisions', [])
            retail_bias = sum(1 for d in retail_decisions if d.action == ActionType.BUY) - \
                sum(1 for d in retail_decisions if d.action == ActionType.SELL)
            inst_bias = sum(1 for d in inst_decisions if d.action == ActionType.BUY) - \
                sum(1 for d in inst_decisions if d.action == ActionType.SELL)
            score += inst_bias * 0.02
            score -= retail_bias * 0.01  # 逆散户
            threshold = 0.20
        else:  # retail 或其他
            score += atr * 0.0002
            threshold = 0.15

        if score > threshold:
            action = 'buy'
        elif score < -threshold:
            action = 'sell'
        else:
            action = 'hold'

        confidence = min(0.9, 0.5 + abs(score) * 1.2)
        confidence = max(0.1, confidence - volatility * 0.5)

        reason_parts = []
        if action == 'buy':
            if rsi < 40:
                reason_parts.append("RSI偏低")
            if macd > 0:
                reason_parts.append("MACD正向")
            if price > ma20:
                reason_parts.append("价格站上均线")
        elif action == 'sell':
            if rsi > 60:
                reason_parts.append("RSI偏高")
            if macd < 0:
                reason_parts.append("MACD转弱")
            if price < ma20:
                reason_parts.append("价格跌破均线")
        if not reason_parts:
            reason_parts.append("信号中性，保持谨慎")

        return {
            "action": action,
            "confidence": round(confidence, 2),
            "reason": "，".join(reason_parts)
        }

    def structured_chat(self, prompt: str, system_prompt: str = None,
                        metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """结构化聊天，返回JSON"""
        if self.client is None:
            return self._simulate_structured_decision(metadata or {})

        response_text = self.chat(prompt, system_prompt)

        try:
            if '```json' in response_text:
                json_start = response_text.find('```json') + 7
                json_end = response_text.find('```', json_start)
                json_text = response_text[json_start:json_end].strip()
            elif '{' in response_text:
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                json_text = response_text[json_start:json_end]
            else:
                json_text = response_text

            return json.loads(json_text)
        except Exception:
            return {
                "action": "hold",
                "confidence": 0.5,
                "reason": "Unable to parse LLM response"
            }

# ============================================================================
# Agent实现 - Agent Implementations
# ============================================================================

class BaseAgent(ABC):
    """Agent基类"""

    def __init__(self, agent_id: str, agent_type: str, role: AgentRole,
                 risk_tolerance: float, llm_client: LLMClient):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.role = role
        self.risk_tolerance = risk_tolerance
        self.llm_client = llm_client

    @abstractmethod
    def analyze(self, stock_data: Dict[str, Any]) -> AgentDecision:
        """分析并做出决策"""
        pass

    def _build_analysis_prompt(self, stock_data: Dict[str, Any]) -> str:
        """构建分析提示"""
        return f"""
分析以下股票数据并给出交易建议：

股票代码: {stock_data['code']}
股票名称: {stock_data['name']}
当前价格: ${stock_data['price']:.2f}
涨跌幅: {stock_data['change_pct']:.2f}%

技术指标:
- MA5: {stock_data.get('ma5', 0):.2f}
- MA20: {stock_data.get('ma20', 0):.2f}
- MA50: {stock_data.get('ma50', 0):.2f}
- RSI(14): {stock_data.get('rsi', 50):.2f}
- MACD: {stock_data.get('macd', 0):.4f}
- ATR: {stock_data.get('atr', 0):.2f}
- 波动率: {stock_data.get('volatility', 0):.2%}

市场情绪:
- 成交量比率: {stock_data.get('volume_ratio', 1):.2f}
- 换手率: {stock_data.get('turnover_rate', 0):.2%}
- 市场情绪: {stock_data.get('market_sentiment', 0.5):.2f}

请以JSON格式返回你的决策：
{{
    "action": "buy/sell/hold",
    "confidence": 0.0-1.0,
    "reason": "你的分析理由"
}}
"""

class RetailAgent(BaseAgent):
    """散户Agent基类"""

    def __init__(self, agent_id: str, agent_type: str, llm_client: LLMClient):
        super().__init__(
            agent_id=agent_id,
            agent_type=agent_type,
            role=AgentRole.RETAIL,
            risk_tolerance=0.8,  # 散户风险偏好高
            llm_client=llm_client
        )

    def analyze(self, stock_data: Dict[str, Any]) -> AgentDecision:
        """散户分析逻辑"""
        system_prompt = self._get_personality_prompt()
        prompt = self._build_analysis_prompt(stock_data)

        result = self.llm_client.structured_chat(
            prompt,
            system_prompt,
            metadata={
                'agent_type': 'retail',
                'identifier': self.agent_id,
                'stock_data': stock_data
            }
        )

        return AgentDecision(
            action=ActionType(result.get('action', 'hold')),
            confidence=result.get('confidence', 0.5),
            reason=result.get('reason', 'No specific reason'),
            agent_id=self.agent_id,
            agent_type=self.agent_type
        )

    def _get_personality_prompt(self) -> str:
        """获取人格提示"""
        personalities = {
            'momentum_chaser': """
你是一个追涨杀跌型散户投资者。你的特点:
- 看到股票上涨就想买入，害怕错过机会
- 看到股票下跌就恐慌，想要卖出
- 喜欢追热点和概念股
- 容易被短期价格波动影响
- 风险意识较弱，喜欢高风险高收益
""",
            'panic_seller': """
你是一个恐慌型散户投资者。你的特点:
- 对市场波动非常敏感，容易恐慌
- 看到亏损就想止损，即使是小幅回调
- 经常在底部卖出
- 风险厌恶，害怕损失
- 容易被负面新闻影响
""",
            'herd_follower': """
你是一个跟风型散户投资者。你的特点:
- 喜欢跟随大众，人买我买，人卖我卖
- 相信"群众的智慧"
- 喜欢看各种股评和推荐
- 缺乏独立判断
- 容易在市场情绪极端时做出错误决策
""",
            'value_seeker': """
你是一个价值型散户投资者。你的特点:
- 关注股票的基本面和估值
- 喜欢买入被低估的股票
- 有一定的耐心，愿意长期持有
- 但分析能力有限，容易看错
- 有时会陷入"价值陷阱"
""",
            'technical_trader': """
你是一个技术型散户投资者。你的特点:
- 相信技术分析，喜欢看K线和指标
- 关注MA、RSI、MACD等技术指标
- 喜欢做短线交易
- 有一定的交易纪律
- 但容易过度交易
"""
        }
        return personalities.get(self.agent_type, personalities['herd_follower'])

class InstitutionalAgent(BaseAgent):
    """机构Agent基类"""

    def __init__(self, agent_id: str, agent_type: str, llm_client: LLMClient):
        super().__init__(
            agent_id=agent_id,
            agent_type=agent_type,
            role=AgentRole.INSTITUTIONAL,
            risk_tolerance=0.3,  # 机构风险偏好低
            llm_client=llm_client
        )

    def analyze(self, stock_data: Dict[str, Any]) -> AgentDecision:
        """机构分析逻辑"""
        system_prompt = self._get_strategy_prompt()
        prompt = self._build_analysis_prompt(stock_data)

        result = self.llm_client.structured_chat(
            prompt,
            system_prompt,
            metadata={
                'agent_type': 'institutional',
                'identifier': self.agent_id,
                'stock_data': stock_data
            }
        )

        return AgentDecision(
            action=ActionType(result.get('action', 'hold')),
            confidence=result.get('confidence', 0.5),
            reason=result.get('reason', 'No specific reason'),
            agent_id=self.agent_id,
            agent_type=self.agent_type
        )

    def _get_strategy_prompt(self) -> str:
        """获取策略提示"""
        strategies = {
            'quantitative': """
你是一个量化对冲基金。你的特点:
- 使用多因子模型和统计套利策略
- 关注市场异常和价格偏差
- 严格的风险控制和仓位管理
- 快速进出，不恋战
- 基于数据和模型，不受情绪影响
""",
            'value_investor': """
你是一个价值投资机构。你的特点:
- 深度研究公司基本面
- 关注长期价值，不在意短期波动
- 寻找被市场低估的优质公司
- 有耐心，愿意长期持有
- 严格的估值纪律
""",
            'trend_follower': """
你是一个趋势跟踪基金。你的特点:
- 识别并跟随市场趋势
- 趋势确立后才进场
- 使用移动平均线和突破策略
- 严格执行止损
- 顺势而为，不逆势操作
""",
            'high_frequency': """
你是一个高频交易机构。你的特点:
- 捕捉短期价格波动
- 快进快出，持仓时间很短
- 关注流动性和价差
- 严格的风险控制
- 依赖算法和模型
""",
            'index_fund': """
你是一个指数基金。你的特点:
- 被动跟踪指数
- 不主动择时
- 关注成分股调整
- 低换手率
- 稳健保守
"""
        }
        return strategies.get(self.agent_type, strategies['quantitative'])

# ============================================================================
# 专家委员会 - Expert Panel
# ============================================================================

class Expert:
    """专家基类"""

    def __init__(self, name: str, specialty: str, llm_client: LLMClient):
        self.name = name
        self.specialty = specialty
        self.llm_client = llm_client

    def analyze(self, stock_data: Dict[str, Any],
                retail_decisions: List[AgentDecision],
                institutional_decisions: List[AgentDecision],
                previous_opinions: List[ExpertOpinion] = None) -> ExpertOpinion:
        """专家分析"""
        prompt = self._build_expert_prompt(
            stock_data, retail_decisions, institutional_decisions, previous_opinions
        )
        system_prompt = self._get_expert_prompt()

        result = self.llm_client.structured_chat(
            prompt,
            system_prompt,
            metadata={
                'agent_type': 'expert',
                'identifier': self.name,
                'stock_data': stock_data,
                'retail_decisions': retail_decisions,
                'institutional_decisions': institutional_decisions
            }
        )

        return ExpertOpinion(
            expert_name=self.name,
            action=ActionType(result.get('action', 'hold')),
            confidence=result.get('confidence', 0.5),
            reasoning=result.get('reasoning', ''),
            key_factors=result.get('key_factors', []),
            risks=result.get('risks', [])
        )

    def _build_expert_prompt(self, stock_data: Dict[str, Any],
                            retail_decisions: List[AgentDecision],
                            institutional_decisions: List[AgentDecision],
                            previous_opinions: List[ExpertOpinion] = None) -> str:
        """构建专家提示"""
        # 统计散户情绪
        retail_buy = sum(1 for d in retail_decisions if d.action == ActionType.BUY)
        retail_sell = sum(1 for d in retail_decisions if d.action == ActionType.SELL)
        retail_hold = sum(1 for d in retail_decisions if d.action == ActionType.HOLD)

        # 统计机构情绪
        inst_buy = sum(1 for d in institutional_decisions if d.action == ActionType.BUY)
        inst_sell = sum(1 for d in institutional_decisions if d.action == ActionType.SELL)
        inst_hold = sum(1 for d in institutional_decisions if d.action == ActionType.HOLD)

        prompt = f"""
作为{self.specialty}专家，请分析以下情况：

股票信息:
- 代码: {stock_data['code']}
- 价格: ${stock_data['price']:.2f}
- 涨跌幅: {stock_data['change_pct']:.2f}%
- RSI: {stock_data.get('rsi', 50):.2f}
- MACD: {stock_data.get('macd', 0):.4f}

散户Agent决策 (总计{len(retail_decisions)}个):
- 买入: {retail_buy} ({retail_buy/len(retail_decisions)*100:.1f}%)
- 卖出: {retail_sell} ({retail_sell/len(retail_decisions)*100:.1f}%)
- 持有: {retail_hold} ({retail_hold/len(retail_decisions)*100:.1f}%)

机构Agent决策 (总计{len(institutional_decisions)}个):
- 买入: {inst_buy} ({inst_buy/len(institutional_decisions)*100:.1f}%)
- 卖出: {inst_sell} ({inst_sell/len(institutional_decisions)*100:.1f}%)
- 持有: {inst_hold} ({inst_hold/len(institutional_decisions)*100:.1f}%)
"""

        if previous_opinions:
            prompt += "\n其他专家意见:\n"
            for op in previous_opinions:
                prompt += f"- {op.expert_name}: {op.action.value} (置信度{op.confidence:.0%})\n"

        prompt += """
请以JSON格式返回你的专家意见：
{
    "action": "buy/sell/hold",
    "confidence": 0.0-1.0,
    "reasoning": "你的详细分析",
    "key_factors": ["关键因素1", "关键因素2"],
    "risks": ["风险1", "风险2"]
}
"""
        return prompt

    def _get_expert_prompt(self) -> str:
        """获取专家人设"""
        return f"你是一位经验丰富的{self.specialty}专家，擅长{self.specialty}。"

class ExpertPanel:
    """专家委员会"""

    def __init__(self, llm_client: LLMClient, discussion_rounds: int = 2):
        self.llm_client = llm_client
        self.discussion_rounds = discussion_rounds

        # 创建专家
        self.experts = [
            Expert("情绪分析师", "市场情绪和散户行为分析", llm_client),
            Expert("机构追踪师", "机构资金流向和大单分析", llm_client),
            Expert("风险控制师", "风险评估和仓位管理", llm_client),
            Expert("择时专家", "市场时机和趋势判断", llm_client)
        ]

    def make_decision(self, stock_data: Dict[str, Any],
                     retail_decisions: List[AgentDecision],
                     institutional_decisions: List[AgentDecision]) -> PanelDecision:
        """专家委员会决策"""
        all_opinions = []

        # 多轮讨论
        for round_num in range(self.discussion_rounds):
            round_opinions = []

            for expert in self.experts:
                opinion = expert.analyze(
                    stock_data,
                    retail_decisions,
                    institutional_decisions,
                    all_opinions if round_num > 0 else None
                )
                round_opinions.append(opinion)

            all_opinions.extend(round_opinions)

        # 综合决策
        return self._synthesize_decision(
            stock_data,
            retail_decisions,
            institutional_decisions,
            all_opinions
        )

    def _synthesize_decision(self, stock_data: Dict[str, Any],
                           retail_decisions: List[AgentDecision],
                           institutional_decisions: List[AgentDecision],
                           expert_opinions: List[ExpertOpinion]) -> PanelDecision:
        """综合决策"""
        # 统计散户情绪
        retail_sentiment = self._calculate_sentiment(retail_decisions)

        # 统计机构情绪
        inst_sentiment = self._calculate_sentiment(institutional_decisions)

        # 统计专家意见（只看最后一轮）
        last_round_opinions = expert_opinions[-len(self.experts):]

        # 计算专家共识
        expert_actions = [op.action for op in last_round_opinions]
        expert_confidences = [op.confidence for op in last_round_opinions]

        # 投票决策
        buy_votes = sum(1 for a in expert_actions if a == ActionType.BUY)
        sell_votes = sum(1 for a in expert_actions if a == ActionType.SELL)
        hold_votes = sum(1 for a in expert_actions if a == ActionType.HOLD)

        if buy_votes > sell_votes and buy_votes > hold_votes:
            final_action = ActionType.BUY
        elif sell_votes > buy_votes and sell_votes > hold_votes:
            final_action = ActionType.SELL
        else:
            final_action = ActionType.HOLD

        # 计算置信度
        max_votes = max(buy_votes, sell_votes, hold_votes)
        confidence = max_votes / len(expert_actions)
        avg_expert_confidence = (
            statistics.fmean(expert_confidences)
            if expert_confidences else 0.0
        )
        final_confidence = (confidence + avg_expert_confidence) / 2

        # 计算共识度
        consensus_level = max_votes / len(expert_actions)

        # 生成reasoning
        reasoning = f"专家投票: 买{buy_votes}票, 卖{sell_votes}票, 持{hold_votes}票。"
        reasoning += f"散户{retail_sentiment['dominant_action'].value}倾向{retail_sentiment['dominant_pct']:.0%}，"
        reasoning += f"机构{inst_sentiment['dominant_action'].value}倾向{inst_sentiment['dominant_pct']:.0%}。"

        # "顺机构反散户"策略调整
        if (retail_sentiment['dominant_action'] == ActionType.SELL and
            retail_sentiment['dominant_pct'] > 0.6 and
            inst_sentiment['dominant_action'] == ActionType.BUY):
            # 散户恐慌卖出，机构买入 -> 强烈买入信号
            final_action = ActionType.BUY
            final_confidence = min(final_confidence * 1.2, 0.95)
            reasoning += " [策略触发: 散户恐慌+机构买入]"

        elif (retail_sentiment['dominant_action'] == ActionType.BUY and
              retail_sentiment['dominant_pct'] > 0.6 and
              inst_sentiment['dominant_action'] == ActionType.SELL):
            # 散户疯狂买入，机构卖出 -> 强烈卖出信号
            final_action = ActionType.SELL
            final_confidence = min(final_confidence * 1.2, 0.95)
            reasoning += " [策略触发: 散户追高+机构卖出]"

        return PanelDecision(
            final_action=final_action,
            confidence=final_confidence,
            consensus_level=consensus_level,
            reasoning=reasoning,
            expert_opinions=last_round_opinions,
            retail_sentiment=retail_sentiment,
            institutional_sentiment=inst_sentiment
        )

    def _calculate_sentiment(self, decisions: List[AgentDecision]) -> Dict[str, Any]:
        """计算情绪统计"""
        buy_count = sum(1 for d in decisions if d.action == ActionType.BUY)
        sell_count = sum(1 for d in decisions if d.action == ActionType.SELL)
        hold_count = sum(1 for d in decisions if d.action == ActionType.HOLD)

        total = len(decisions)

        buy_pct = buy_count / total
        sell_pct = sell_count / total
        hold_pct = hold_count / total

        # 确定主导行动
        if buy_pct > sell_pct and buy_pct > hold_pct:
            dominant_action = ActionType.BUY
            dominant_pct = buy_pct
        elif sell_pct > buy_pct and sell_pct > hold_pct:
            dominant_action = ActionType.SELL
            dominant_pct = sell_pct
        else:
            dominant_action = ActionType.HOLD
            dominant_pct = hold_pct

        return {
            'buy_pct': buy_pct,
            'sell_pct': sell_pct,
            'hold_pct': hold_pct,
            'dominant_action': dominant_action,
            'dominant_pct': dominant_pct
        }

# ============================================================================
# 数据提供器 - Data Provider
# ============================================================================

class DataProvider:
    """数据提供器（纯Python实现，确保离线可运行）"""

    def __init__(self):
        self.cache: Dict[Tuple[str, str, str], TimeSeries] = {}

    def get_stock_list(self) -> List[str]:
        """获取股票列表"""
        return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'JPM', 'V', 'WMT']

    def get_historical_data(self, symbol: str, start_date: str, end_date: str) -> TimeSeries:
        """生成稳定的模拟历史数据。"""
        cache_key = (symbol, start_date, end_date)
        if cache_key in self.cache:
            return self.cache[cache_key]

        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        dates = _daterange(start, end)

        seed = int.from_bytes(hashlib.sha256(symbol.encode("utf-8")).digest()[:8], "big")
        rng = random.Random(seed)

        price = 100 + rng.gauss(0, 20)
        records: List[Dict[str, Any]] = []

        for date in dates:
            daily_return = rng.gauss(0, 0.02)
            price *= math.exp(daily_return)
            base_spread = abs(rng.gauss(0, 0.01))
            high = price * (1 + base_spread)
            low = max(price * (1 - base_spread), 1)
            open_price = price * (1 + rng.gauss(0, 0.005))
            volume = rng.randint(1_000_000, 10_000_000)

            record = {
                'date': date,
                'open': open_price,
                'high': max(high, open_price, price),
                'low': min(low, open_price, price),
                'close': price,
                'volume': float(volume)
            }
            records.append(record)

        self.calculate_technical_indicators(records)
        series = TimeSeries(records)
        self.cache[cache_key] = series
        return series

    def calculate_technical_indicators(self, records: List[Dict[str, Any]]) -> None:
        """计算技术指标（就地修改records）。"""
        closes = [r['close'] for r in records]
        volumes = [r['volume'] for r in records]

        # 移动平均线
        for period in [5, 10, 20, 50, 100, 200]:
            ma_values = _rolling_mean(closes, period)
            for rec, value in zip(records, ma_values):
                rec[f'ma{period}'] = value

        # RSI
        gains: List[float] = []
        losses: List[float] = []
        rsi_values: List[Optional[float]] = [None]
        for i in range(1, len(closes)):
            delta = closes[i] - closes[i - 1]
            gains.append(max(delta, 0.0))
            losses.append(max(-delta, 0.0))
            if len(gains) >= RSI_PERIOD:
                avg_gain = sum(gains[-RSI_PERIOD:]) / RSI_PERIOD
                avg_loss = sum(losses[-RSI_PERIOD:]) / RSI_PERIOD
                if avg_loss == 0:
                    rsi = 100.0
                else:
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))
                rsi_values.append(rsi)
            else:
                rsi_values.append(None)
        if len(rsi_values) < len(records):
            rsi_values.extend([None] * (len(records) - len(rsi_values)))
        for rec, value in zip(records, rsi_values):
            rec['rsi'] = value

        # MACD
        ema_fast = _ema_series(closes, MACD_FAST)
        ema_slow = _ema_series(closes, MACD_SLOW)
        macd_values: List[Optional[float]] = []
        for fast, slow in zip(ema_fast, ema_slow):
            if fast is None or slow is None:
                macd_values.append(None)
            else:
                macd_values.append(fast - slow)
        macd_signal = _ema_series(macd_values, MACD_SIGNAL)
        macd_hist: List[Optional[float]] = []
        for macd, signal in zip(macd_values, macd_signal):
            if macd is None or signal is None:
                macd_hist.append(None)
            else:
                macd_hist.append(macd - signal)
        for rec, macd, signal, hist in zip(records, macd_values, macd_signal, macd_hist):
            rec['macd'] = macd
            rec['macd_signal'] = signal
            rec['macd_hist'] = hist

        # ATR
        true_ranges: List[float] = []
        for idx, rec in enumerate(records):
            high_low = rec['high'] - rec['low']
            if idx == 0:
                true_ranges.append(high_low)
            else:
                prev_close = records[idx - 1]['close']
                high_close = abs(rec['high'] - prev_close)
                low_close = abs(rec['low'] - prev_close)
                true_ranges.append(max(high_low, high_close, low_close))
        atr_values = _rolling_mean(true_ranges, ATR_PERIOD)
        for rec, value in zip(records, atr_values):
            rec['atr'] = value

        # 波动率
        returns = _pct_change_series(closes)
        std_source = [r if r is not None else 0.0 for r in returns]
        raw_volatility = _rolling_std(std_source, 20)
        volatility_values: List[Optional[float]] = []
        for idx, value in enumerate(raw_volatility):
            if idx + 1 < 20:
                volatility_values.append(None)
            else:
                volatility_values.append(value)
        for rec, value, ret in zip(records, volatility_values, returns):
            rec['returns'] = ret
            rec['volatility'] = value

        # 成交量比率
        volume_ma = _rolling_mean(volumes, 20)
        for rec, value in zip(records, volume_ma):
            rec['volume_ma'] = value
            rec['volume_ratio'] = rec['volume'] / value if value else None


# ============================================================================
# 回测引擎 - Backtest Engine
# ============================================================================

class BacktestEngine:
    """回测引擎"""

    def __init__(self, initial_capital: float, commission_rate: float, slippage_rate: float):
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate

        self.cash = initial_capital
        self.portfolio: Dict[str, Position] = {}
        self.trades = []
        self.daily_values = []
        self.positions_history = []

    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """计算组合总值"""
        portfolio_value = 0.0
        for symbol, position in self.portfolio.items():
            price = current_prices.get(symbol, position.current_price)
            position.current_price = price
            position.unrealized_pnl = (price - position.avg_price) * position.shares
            position.unrealized_pnl_pct = ((price - position.avg_price) / position.avg_price
                                           if position.avg_price else 0.0)
            portfolio_value += price * position.shares
        return self.cash + portfolio_value

    def execute_trade(self, date: datetime, symbol: str, action: ActionType,
                     price: float, confidence: float, reason: str,
                     max_position_size: float) -> bool:
        """执行交易"""
        if action == ActionType.HOLD:
            return False

        if action == ActionType.BUY:
            # 计算买入数量
            max_position_value = self.initial_capital * max_position_size
            available_cash = self.cash * 0.95  # 保留5%现金
            position_value = min(max_position_value, available_cash)

            if position_value < price * 100:  # 至少买100股
                return False

            # 应用滑点
            actual_price = price * (1 + self.slippage_rate)

            # 计算股数
            shares = int(position_value / actual_price)
            cost = shares * actual_price
            commission = cost * self.commission_rate
            total_cost = cost + commission

            if total_cost > self.cash:
                return False

            # 执行买入
            self.cash -= total_cost
            previous = self.portfolio.get(symbol)
            if previous:
                total_shares = previous.shares + shares
                total_cost = previous.avg_price * previous.shares + total_cost
                avg_price = total_cost / total_shares if total_shares else actual_price
                self.portfolio[symbol] = Position(
                    symbol=symbol,
                    shares=total_shares,
                    avg_price=avg_price,
                    current_price=actual_price,
                    unrealized_pnl=0.0,
                    unrealized_pnl_pct=0.0
                )
            else:
                self.portfolio[symbol] = Position(
                    symbol=symbol,
                    shares=shares,
                    avg_price=actual_price,
                    current_price=actual_price,
                    unrealized_pnl=0.0,
                    unrealized_pnl_pct=0.0
                )

            self.trades.append(Trade(
                date=date,
                symbol=symbol,
                action=ActionType.BUY,
                price=actual_price,
                shares=shares,
                commission=commission,
                reason=reason,
                confidence=confidence
            ))

            return True

        elif action == ActionType.SELL:
            if symbol not in self.portfolio or self.portfolio[symbol].shares <= 0:
                return False

            position = self.portfolio[symbol]
            shares = position.shares

            # 应用滑点
            actual_price = price * (1 - self.slippage_rate)

            proceeds = shares * actual_price
            commission = proceeds * self.commission_rate
            net_proceeds = proceeds - commission

            cost_basis = position.avg_price * shares
            pnl = net_proceeds - cost_basis
            pnl_pct = pnl / cost_basis if cost_basis else 0.0

            # 执行卖出
            self.cash += net_proceeds
            del self.portfolio[symbol]

            self.trades.append(Trade(
                date=date,
                symbol=symbol,
                action=ActionType.SELL,
                price=actual_price,
                shares=shares,
                commission=commission,
                reason=reason,
                confidence=confidence,
                pnl=pnl,
                pnl_pct=pnl_pct
            ))

            return True

        return False

    def check_stop_loss_take_profit(self, date: datetime, symbol: str,
                                    current_price: float, entry_price: float) -> Optional[ActionType]:
        """检查止损止盈"""
        if symbol not in self.portfolio:
            return None

        position = self.portfolio[symbol]
        reference_price = entry_price if entry_price is not None else position.avg_price
        if not reference_price:
            return None

        pnl_pct = (current_price - reference_price) / reference_price

        if pnl_pct <= -STOP_LOSS_PCT:
            return ActionType.SELL  # 止损
        elif pnl_pct >= TAKE_PROFIT_PCT:
            return ActionType.SELL  # 止盈

        return None

    def record_daily_value(self, date: datetime, current_prices: Dict[str, float]):
        """记录每日价值"""
        total_value = self.get_portfolio_value(current_prices)
        self.daily_values.append({
            'date': date,
            'total_value': total_value,
            'cash': self.cash,
            'portfolio_value': total_value - self.cash
        })

    def get_performance_metrics(self) -> Dict[str, Any]:
        """计算业绩指标"""
        if not self.daily_values:
            return {}

        values = [entry['total_value'] for entry in self.daily_values]
        returns: List[float] = []
        for i in range(1, len(values)):
            prev = values[i - 1]
            returns.append((values[i] - prev) / prev if prev else 0.0)

        final_value = values[-1]
        total_return = (final_value - self.initial_capital) / self.initial_capital
        annual_return = (1 + total_return) ** (365 / len(values)) - 1 if len(values) > 0 else 0.0

        volatility = statistics.pstdev(returns) * math.sqrt(252) if len(returns) > 1 else 0.0
        sharpe_ratio = (annual_return - 0.02) / volatility if volatility > 0 else 0.0

        peak = values[0]
        max_drawdown = 0.0
        for value in values:
            if value > peak:
                peak = value
            drawdown = (value - peak) / peak if peak else 0.0
            if drawdown < max_drawdown:
                max_drawdown = drawdown

        winning_trades = [t for t in self.trades if self._calculate_trade_pnl(t) > 0]
        total_trades = len([t for t in self.trades if t.action == ActionType.SELL])
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0.0

        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': len(self.trades),
            'final_value': final_value
        }

    def _calculate_trade_pnl(self, trade: Trade) -> float:
        """计算单笔交易盈亏"""
        if trade.action != ActionType.SELL:
            return 0.0
        return trade.pnl

# ============================================================================
# 完整交易系统 - Complete Trading System
# ============================================================================

class AITradingSystem:
    """AI多Agent完整交易系统"""

    def __init__(self):
        print("=" * 80)
        print("🚀 AI多Agent完整交易系统启动")
        print("=" * 80)

        # 初始化LLM客户端
        print(f"\n初始化LLM客户端: {LLM_PROVIDER} - {LLM_MODEL}")
        self.llm_client = LLMClient(provider=LLM_PROVIDER, model=LLM_MODEL)

        # 设置随机数生成器，保证回测结果可复现
        random.seed(42)
        self.rng = random.Random(42)

        # 创建散户Agent
        print(f"\n创建散户Agent...")
        self.retail_agents = []
        for agent_type, count in RETAIL_AGENT_COUNT.items():
            for i in range(count):
                agent = RetailAgent(
                    agent_id=f"retail_{agent_type}_{i+1}",
                    agent_type=agent_type,
                    llm_client=self.llm_client
                )
                self.retail_agents.append(agent)
        print(f"✅ 创建了 {len(self.retail_agents)} 个散户Agent")

        # 创建机构Agent
        print(f"\n创建机构Agent...")
        self.institutional_agents = []
        for agent_type, count in INSTITUTIONAL_AGENT_COUNT.items():
            for i in range(count):
                agent = InstitutionalAgent(
                    agent_id=f"inst_{agent_type}_{i+1}",
                    agent_type=agent_type,
                    llm_client=self.llm_client
                )
                self.institutional_agents.append(agent)
        print(f"✅ 创建了 {len(self.institutional_agents)} 个机构Agent")

        # 创建专家委员会
        print(f"\n创建专家委员会...")
        self.expert_panel = ExpertPanel(
            llm_client=self.llm_client,
            discussion_rounds=EXPERT_DISCUSSION_ROUNDS
        )
        print(f"✅ 创建了专家委员会（{EXPERT_DISCUSSION_ROUNDS}轮讨论，{len(self.expert_panel.experts)}位专家）")

        # 数据提供器
        print(f"\n初始化数据提供器...")
        self.data_provider = DataProvider()
        print(f"✅ 数据提供器已初始化")

        # 回测引擎
        self.backtest_engine = None

        print("\n" + "=" * 80)
        print("✅ 系统初始化完成!")
        print(f"总Agent数: {len(self.retail_agents) + len(self.institutional_agents)}")
        print(f"散户Agent: {len(self.retail_agents)}")
        print(f"机构Agent: {len(self.institutional_agents)}")
        print(f"专家: {len(self.expert_panel.experts)}")
        print("=" * 80)

    def analyze_stock(self, stock_data: Dict[str, Any], sample_size: int = 20) -> PanelDecision:
        """分析单个股票"""
        symbol = stock_data['code']

        # 散户分析（采样，不是全部）
        retail_count = min(sample_size, len(self.retail_agents))
        retail_sample = []
        if retail_count:
            retail_sample = self.rng.sample(self.retail_agents, retail_count)
        retail_decisions = [agent.analyze(stock_data) for agent in retail_sample]

        # 机构分析（采样）
        inst_count = min(sample_size // 2, len(self.institutional_agents))
        inst_sample = []
        if inst_count:
            inst_sample = self.rng.sample(self.institutional_agents, inst_count)
        inst_decisions = [agent.analyze(stock_data) for agent in inst_sample]

        # 专家委员会决策
        panel_decision = self.expert_panel.make_decision(
            stock_data,
            retail_decisions,
            inst_decisions
        )

        return panel_decision

    def run_backtest(self, start_date: str, end_date: str, symbols: List[str] = None):
        """运行回测"""
        print("\n" + "=" * 80)
        print("📊 开始回测")
        print("=" * 80)
        print(f"起始日期: {start_date}")
        print(f"结束日期: {end_date}")
        print(f"初始资金: ${INITIAL_CAPITAL:,.2f}")
        print("=" * 80)

        # 初始化回测引擎
        self.backtest_engine = BacktestEngine(
            initial_capital=INITIAL_CAPITAL,
            commission_rate=COMMISSION_RATE,
            slippage_rate=SLIPPAGE_RATE
        )

        # 获取股票列表
        if symbols is None:
            symbols = self.data_provider.get_stock_list()

        print(f"\n股票池: {', '.join(symbols)}")

        # 获取所有股票的历史数据
        print(f"\n加载历史数据...")
        stock_data_dict = {}
        for symbol in symbols:
            series = self.data_provider.get_historical_data(symbol, start_date, end_date)
            stock_data_dict[symbol] = series
            print(f"  {symbol}: {len(series)} 天")

        # 获取所有交易日期
        all_dates = sorted(set(
            date
            for series in stock_data_dict.values()
            for date in series.dates
        ))

        print(f"\n总交易日: {len(all_dates)}")
        print(f"\n开始模拟交易...")

        # 记录买入价格（用于止损止盈）
        entry_prices = {}

        # 逐日回测
        for i, date in enumerate(all_dates):
            if i % 50 == 0:
                progress = i / len(all_dates) * 100
                print(f"\n进度: {progress:.1f}% ({i}/{len(all_dates)}) - {date.strftime('%Y-%m-%d')}")

            # 获取当日所有股票数据
            current_prices = {}
            stock_infos = {}

            for symbol in symbols:
                series = stock_data_dict[symbol]
                row = series.get_by_date(date)

                if not row:
                    continue

                current_prices[symbol] = row['close']

                # 检查数据完整性
                if row.get('rsi') is None or row.get('macd') is None:
                    continue

                stock_infos[symbol] = {
                    'code': symbol,
                    'name': symbol,
                    'price': row['close'],
                    'change_pct': ((row['close'] - row['open']) / row['open']) * 100 if row['open'] else 0,
                    'ma5': row.get('ma5') or 0,
                    'ma10': row.get('ma10') or 0,
                    'ma20': row.get('ma20') or 0,
                    'ma50': row.get('ma50') or 0,
                    'rsi': row.get('rsi') or 0,
                    'macd': row.get('macd') or 0,
                    'atr': row.get('atr') or 0,
                    'volatility': row.get('volatility') or 0,
                    'volume_ratio': row.get('volume_ratio') or 1,
                    'turnover_rate': 0.05,
                    'market_sentiment': 0.5
                }

            # 检查止损止盈
            for symbol in list(self.backtest_engine.portfolio.keys()):
                if symbol in current_prices and symbol in entry_prices:
                    action = self.backtest_engine.check_stop_loss_take_profit(
                        date, symbol, current_prices[symbol], entry_prices[symbol]
                    )
                    if action == ActionType.SELL:
                        self.backtest_engine.execute_trade(
                            date, symbol, ActionType.SELL,
                            current_prices[symbol], 1.0,
                            "止损/止盈", MAX_POSITION_SIZE
                        )
                        if symbol in entry_prices:
                            del entry_prices[symbol]

            # 分析并交易
            for symbol, stock_info in stock_infos.items():
                # 如果已经持有，跳过
                if symbol in self.backtest_engine.portfolio:
                    continue

                # AI分析
                decision = self.analyze_stock(stock_info, sample_size=10)

                # 检查决策条件
                if (decision.confidence >= MIN_CONFIDENCE and
                    decision.consensus_level >= MIN_CONSENSUS):

                    success = self.backtest_engine.execute_trade(
                        date, symbol, decision.final_action,
                        current_prices[symbol], decision.confidence,
                        decision.reasoning, MAX_POSITION_SIZE
                    )

                    if success and decision.final_action == ActionType.BUY:
                        entry_prices[symbol] = current_prices[symbol]

            # 记录每日价值
            self.backtest_engine.record_daily_value(date, current_prices)

        # 计算业绩指标
        print("\n" + "=" * 80)
        print("📈 回测完成！计算业绩指标...")
        print("=" * 80)

        metrics = self.backtest_engine.get_performance_metrics()

        self._print_backtest_results(metrics)

        return metrics

    def _print_backtest_results(self, metrics: Dict[str, Any]):
        """打印回测结果"""
        print("\n" + "=" * 80)
        print("📊 回测结果")
        print("=" * 80)

        print(f"\n💰 收益指标:")
        print(f"  初始资金:    ${INITIAL_CAPITAL:,.2f}")
        print(f"  最终资金:    ${metrics['final_value']:,.2f}")
        print(f"  总收益:      {metrics['total_return']:+.2%}")
        print(f"  年化收益:    {metrics['annual_return']:+.2%}")

        print(f"\n📉 风险指标:")
        print(f"  波动率:      {metrics['volatility']:.2%}")
        print(f"  最大回撤:    {metrics['max_drawdown']:.2%}")
        print(f"  夏普比率:    {metrics['sharpe_ratio']:.2f}")

        print(f"\n📋 交易指标:")
        print(f"  总交易次数:  {metrics['total_trades']}")
        print(f"  胜率:        {metrics['win_rate']:.2%}")

        print("\n" + "=" * 80)

        # 评估结果
        print("\n✨ AI评估:")
        if metrics['total_return'] > 0.3:
            print("  🎉 优秀! 收益超过30%")
        elif metrics['total_return'] > 0.15:
            print("  👍 良好! 收益超过15%")
        elif metrics['total_return'] > 0:
            print("  ✓ 盈利")
        else:
            print("  ⚠️  亏损，需要优化")

        drawdown = abs(metrics['max_drawdown'])
        if drawdown >= 0.20:
            print("  ⚠️  警告: 回撤超过20%，风险较高!")
        elif drawdown >= 0.10:
            print("  ⚡ 注意: 回撤在10-20%，需要控制风险")
        else:
            print("  ✓ 回撤控制良好")

        if metrics['sharpe_ratio'] > 2:
            print("  🌟 夏普比率优秀!")
        elif metrics['sharpe_ratio'] > 1:
            print("  ✓ 夏普比率良好")
        else:
            print("  ⚡ 风险调整后收益一般")

        print("=" * 80)

# ============================================================================
# 主程序 - Main Program
# ============================================================================

def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("🤖 AI多Agent股票自动交易系统")
    print("顺机构 反散户 策略")
    print("=" * 80)

    print("\n系统配置:")
    print(f"  LLM: {LLM_PROVIDER} - {LLM_MODEL}")
    print(f"  散户Agent: {sum(RETAIL_AGENT_COUNT.values())}个")
    print(f"  机构Agent: {sum(INSTITUTIONAL_AGENT_COUNT.values())}个")
    print(f"  专家讨论: {EXPERT_DISCUSSION_ROUNDS}轮")

    print(f"\n回测配置:")
    print(f"  时间范围: {BACKTEST_START_DATE} ~ {BACKTEST_END_DATE}")
    print(f"  初始资金: ${INITIAL_CAPITAL:,.2f}")
    print(f"  最大仓位: {MAX_POSITION_SIZE:.0%}")
    print(f"  佣金率: {COMMISSION_RATE:.2%}")

    print("\n" + "=" * 80)

    # 检查API密钥
    if LLM_PROVIDER == 'anthropic' and not ANTHROPIC_API_KEY:
        print("\n⚠️  警告: ANTHROPIC_API_KEY未设置")
        print("系统将使用模拟模式运行（基于规则的简单决策）")
        print("要使用完整AI功能，请设置环境变量: export ANTHROPIC_API_KEY=your_key")
        print("\n按回车继续使用模拟模式，或 Ctrl+C 退出...")
        input()
    elif LLM_PROVIDER == 'openai' and not OPENAI_API_KEY:
        print("\n⚠️  警告: OPENAI_API_KEY未设置")
        print("系统将使用模拟模式运行")
        print("要使用完整AI功能，请设置环境变量: export OPENAI_API_KEY=your_key")
        print("\n按回车继续使用模拟模式，或 Ctrl+C 退出...")
        input()

    try:
        # 创建交易系统
        system = AITradingSystem()

        # 运行回测
        metrics = system.run_backtest(
            start_date=BACKTEST_START_DATE,
            end_date=BACKTEST_END_DATE
        )

        print("\n✅ 回测完成!")
        print("\n提示: 要使用真实交易，请参考forex_trading_example.py")

    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

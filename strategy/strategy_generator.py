"""
策略生成器
整合所有Agent的分析，生成交易策略
"""
from typing import Dict, Any, List, Optional
from loguru import logger
from agents import BaseAgent, AgentDecision
from agents.expert_panel import ExpertPanel, PanelDecision
from agents.base_agent import ActionType


class StrategyGenerator:
    """策略生成器"""

    def __init__(
        self,
        retail_agents: List[BaseAgent],
        institutional_agents: List[BaseAgent],
        expert_panel: ExpertPanel,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        初始化

        Args:
            retail_agents: 散户Agent列表
            institutional_agents: 机构Agent列表
            expert_panel: 专家委员会
            config: 配置
        """
        self.retail_agents = retail_agents
        self.institutional_agents = institutional_agents
        self.expert_panel = expert_panel

        self.config = config or {}
        self.min_confidence = self.config.get('min_confidence', 0.75)
        self.sentiment_divergence_threshold = self.config.get('sentiment_divergence', 0.6)
        self.institutional_agreement_threshold = self.config.get('institutional_agreement', 0.7)

        logger.info(
            f"StrategyGenerator initialized with "
            f"{len(retail_agents)} retail agents, "
            f"{len(institutional_agents)} institutional agents"
        )

    def generate_strategy(self, stock_data: Dict[str, Any]) -> Optional[PanelDecision]:
        """
        生成交易策略

        Args:
            stock_data: 股票分析数据

        Returns:
            专家委员会决策
        """
        stock_code = stock_data.get('code', 'UNKNOWN')
        logger.info(f"Generating strategy for {stock_code}...")

        # 1. 收集散户意见
        logger.info("Collecting retail agent decisions...")
        retail_decisions = self._collect_agent_decisions(
            self.retail_agents,
            stock_data
        )

        # 2. 收集机构意见
        logger.info("Collecting institutional agent decisions...")
        institutional_decisions = self._collect_agent_decisions(
            self.institutional_agents,
            stock_data
        )

        # 3. 分析情绪分歧
        divergence = self._analyze_sentiment_divergence(
            retail_decisions,
            institutional_decisions
        )

        logger.info(
            f"Agent analysis complete: "
            f"{len(retail_decisions)} retail, "
            f"{len(institutional_decisions)} institutional, "
            f"divergence={divergence:.2f}"
        )

        # 4. 专家委员会讨论
        logger.info("Expert panel deliberation...")
        panel_decision = self.expert_panel.make_decision(
            stock_data,
            retail_decisions,
            institutional_decisions
        )

        # 5. 应用策略过滤器
        if self._should_execute(panel_decision, divergence):
            logger.info(
                f"Strategy approved: {panel_decision.final_action.value} "
                f"confidence={panel_decision.confidence:.2%}"
            )
            return panel_decision
        else:
            logger.info("Strategy filtered out")
            return None

    def _collect_agent_decisions(
        self,
        agents: List[BaseAgent],
        stock_data: Dict[str, Any]
    ) -> List[AgentDecision]:
        """收集Agent决策"""
        decisions = []
        for agent in agents:
            try:
                decision = agent.analyze(stock_data)
                decisions.append(decision)
            except Exception as e:
                logger.error(f"Error collecting decision from {agent.agent_id}: {e}")

        return decisions

    def _analyze_sentiment_divergence(
        self,
        retail_decisions: List[AgentDecision],
        institutional_decisions: List[AgentDecision]
    ) -> float:
        """
        分析散户与机构的情绪分歧度

        Returns:
            分歧度 (0-1)
        """
        if not retail_decisions or not institutional_decisions:
            return 0.0

        # 计算散户情绪
        retail_buy_ratio = sum(
            1 for d in retail_decisions if d.action == ActionType.BUY
        ) / len(retail_decisions)

        # 计算机构情绪
        inst_buy_ratio = sum(
            1 for d in institutional_decisions if d.action == ActionType.BUY
        ) / len(institutional_decisions)

        # 分歧度
        divergence = abs(retail_buy_ratio - inst_buy_ratio)

        return divergence

    def _should_execute(
        self,
        decision: PanelDecision,
        divergence: float
    ) -> bool:
        """
        判断是否应该执行策略

        Args:
            decision: 专家决策
            divergence: 情绪分歧度

        Returns:
            是否执行
        """
        # 1. 置信度检查
        if decision.confidence < self.min_confidence:
            logger.info(
                f"Confidence too low: {decision.confidence:.2%} < {self.min_confidence:.2%}"
            )
            return False

        # 2. 共识度检查
        if decision.consensus_level < 0.5:
            logger.info(
                f"Consensus too low: {decision.consensus_level:.2%}"
            )
            return False

        # 3. 只在观望时不执行
        if decision.final_action == ActionType.HOLD:
            logger.info("Action is HOLD, no execution needed")
            return False

        # 4. "顺机构反散户"策略检查
        # 当散户和机构分歧大时，才是好机会
        if divergence < self.sentiment_divergence_threshold:
            logger.info(
                f"Sentiment divergence too low: {divergence:.2%} < {self.sentiment_divergence_threshold:.2%}"
            )
            # 但如果机构高度一致且置信度高，也可以执行
            inst_sentiment = decision.institutional_sentiment
            if inst_sentiment['buy_ratio'] < self.institutional_agreement_threshold and \
               inst_sentiment['sell_ratio'] < self.institutional_agreement_threshold:
                return False

        return True

    def get_signal_summary(self, decision: PanelDecision) -> str:
        """
        获取信号摘要

        Args:
            decision: 专家决策

        Returns:
            摘要字符串
        """
        action_str = {
            ActionType.BUY: "买入",
            ActionType.SELL: "卖出",
            ActionType.HOLD: "观望"
        }[decision.final_action]

        summary = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【交易信号】{decision.stock_code}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

最终决策: {action_str}
置信度: {decision.confidence:.1%}
共识度: {decision.consensus_level:.1%}

散户情绪:
  买入: {decision.retail_sentiment['buy_ratio']:.1%}
  卖出: {decision.retail_sentiment['sell_ratio']:.1%}
  观望: {decision.retail_sentiment['hold_ratio']:.1%}

机构态度:
  买入: {decision.institutional_sentiment['buy_ratio']:.1%}
  卖出: {decision.institutional_sentiment['sell_ratio']:.1%}
  观望: {decision.institutional_sentiment['hold_ratio']:.1%}

{decision.reasoning}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        """
        return summary.strip()

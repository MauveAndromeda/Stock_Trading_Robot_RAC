"""
LLM增强的Agent - 真正的AI驱动决策
"""
from typing import Dict, Any, Optional, List
from loguru import logger
import json

from agents.base_agent import BaseAgent, AgentRole, AgentDecision, ActionType
from agents.expert_panel import ExpertOpinion, PanelDecision
from .llm_client import LLMClient


class LLMEnhancedRetailAgent(BaseAgent):
    """LLM增强的散户Agent"""

    def __init__(
        self,
        agent_id: str,
        agent_type: str,
        llm_client: LLMClient,
        risk_tolerance: float = 0.5
    ):
        super().__init__(
            agent_id=agent_id,
            agent_type=agent_type,
            role=AgentRole.RETAIL,
            risk_tolerance=risk_tolerance,
            llm_client=llm_client
        )

    def get_personality_prompt(self) -> str:
        """获取Agent的个性提示词"""
        personalities = {
            "momentum_chaser": """
你是一个追涨杀跌型的散户投资者，特点：
- 看到股票大涨就想买入，害怕错过机会（FOMO心理）
- 看到股票下跌就恐慌卖出，害怕亏损扩大
- 经常追高买入，低位卖出
- 容易被市场情绪和热点影响
- 喜欢看涨幅榜，追热门股票
- 缺乏耐心，喜欢频繁交易
            """,
            "panic_seller": """
你是一个恐慌型的散户投资者，特点：
- 极度厌恶风险和亏损
- 市场稍有风吹草动就想卖出
- 容易被负面新闻影响
- 害怕套牢，宁愿小赚就跑
- 经常在底部卖出
- 持仓时间很短，一有利润就想落袋为安
            """,
            "herd_follower": """
你是一个跟风型的散户投资者，特点：
- 喜欢听消息、跟风买卖
- 看到别人买什么就跟着买
- 热衷于讨论股票论坛和群聊
- 缺乏独立思考能力
- 经常成为最后接盘的人
- 相信"大家都在买，肯定不会错"
            """,
            "value_seeker": """
你是一个价值型散户投资者，特点：
- 关注公司基本面
- 喜欢买"便宜"的股票
- 但缺乏专业分析能力
- 容易被简单指标误导（如只看市盈率）
- 持股时间相对较长
- 相信"好公司迟早会涨"
            """,
            "technical_trader": """
你是一个技术型散户投资者，特点：
- 依赖技术指标（MA、MACD、RSI等）
- 但只会用简单的指标，不够专业
- 喜欢画线、找形态
- 经常被假突破欺骗
- 有时会过度交易
- 相信"技术分析可以预测未来"
            """
        }

        return personalities.get(self.agent_type, personalities["momentum_chaser"])

    def analyze(self, stock_data: Dict[str, Any]) -> AgentDecision:
        """
        使用LLM分析股票并做出决策

        Args:
            stock_data: 股票数据

        Returns:
            AgentDecision
        """
        stock_code = stock_data.get('code', 'UNKNOWN')
        stock_name = stock_data.get('name', '')

        # 构建提示词
        prompt = self._build_analysis_prompt(stock_data)
        system_prompt = self.get_personality_prompt()

        try:
            # 调用LLM
            response = self.llm_client.structured_chat(
                prompt=prompt,
                system_prompt=system_prompt
            )

            # 解析响应
            action_str = response.get('action', '观望')
            if action_str in ['买入', 'buy', 'BUY']:
                action = ActionType.BUY
            elif action_str in ['卖出', 'sell', 'SELL']:
                action = ActionType.SELL
            else:
                action = ActionType.HOLD

            confidence = float(response.get('confidence', 0.5))
            reason = response.get('reason', '基于LLM分析')

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

        except Exception as e:
            logger.error(f"LLM analysis error for {self.agent_id}: {e}")
            # 降级到规则系统
            return self._fallback_analysis(stock_data)

    def _build_analysis_prompt(self, stock_data: Dict[str, Any]) -> str:
        """构建分析提示词"""
        prompt = f"""
请根据以下信息，作为一个{self.agent_type}投资者，分析这只股票并给出你的投资决策：

【股票基本信息】
股票代码：{stock_data.get('code', 'N/A')}
股票名称：{stock_data.get('name', 'N/A')}
当前价格：{stock_data.get('price', 0):.2f} 元

【市场表现】
涨跌幅：{stock_data.get('change_pct', 0):.2f}%
成交额：{stock_data.get('amount', 0)/100000000:.2f} 亿元
换手率：{stock_data.get('turnover_rate', 0):.2f}%
量比：{stock_data.get('volume_ratio', 1):.2f}

【技术指标】
MA5：{stock_data.get('ma5', 0):.2f}
MA20：{stock_data.get('ma20', 0):.2f}
MA60：{stock_data.get('ma60', 0):.2f}
RSI：{stock_data.get('rsi', 50):.2f}
MACD：{stock_data.get('macd', 0):.4f}

【财务指标】
市盈率(PE)：{stock_data.get('pe_ratio', 30):.2f}
市净率(PB)：{stock_data.get('pb_ratio', 3):.2f}
ROE：{stock_data.get('roe', 0.1):.2%}

【市场情绪】
市场情绪指数：{stock_data.get('market_sentiment', 0.5):.2f} (0-1，越高越乐观)

请基于你的投资风格和以上数据，回答：
1. 你会采取什么操作？（买入/卖出/观望）
2. 你的理由是什么？
3. 你对这个决策的置信度是多少？（0-1之间）

请保持你作为{self.agent_type}投资者的性格特点，做出符合你风格的决策。
        """
        return prompt.strip()

    def _fallback_analysis(self, stock_data: Dict[str, Any]) -> AgentDecision:
        """降级到规则系统（当LLM失败时）"""
        from agents.retail_agents import MomentumChaserAgent, PanicSellerAgent

        # 根据类型选择规则Agent
        if self.agent_type == "momentum_chaser":
            fallback_agent = MomentumChaserAgent(self.agent_id)
        else:
            fallback_agent = PanicSellerAgent(self.agent_id)

        return fallback_agent.analyze(stock_data)


class LLMEnhancedInstitutionalAgent(BaseAgent):
    """LLM增强的机构Agent"""

    def __init__(
        self,
        agent_id: str,
        agent_type: str,
        llm_client: LLMClient,
        risk_tolerance: float = 0.5
    ):
        super().__init__(
            agent_id=agent_id,
            agent_type=agent_type,
            role=AgentRole.INSTITUTIONAL,
            risk_tolerance=risk_tolerance,
            llm_client=llm_client
        )

    def get_personality_prompt(self) -> str:
        """获取机构Agent的个性"""
        personalities = {
            "quantitative": """
你是一个量化对冲基金的AI交易系统，特点：
- 完全基于数据和模型决策，情绪中性
- 使用多因子模型分析（动量、价值、质量、流动性等）
- 关注统计套利机会和市场异常
- 快速进出，严格的风险控制
- 追求稳定的Alpha收益
- 不受市场情绪影响，纯理性决策
            """,
            "value_investor": """
你是一个专业的价值投资机构分析师，特点：
- 深度基本面研究，关注企业内在价值
- 长期持有优质公司（3-5年视野）
- 注重安全边际，不追热点
- 逆向投资，在市场恐慌时买入
- 关注护城河、管理层质量、现金流
- 不受短期波动影响，坚持长期主义
            """,
            "trend_follower": """
你是一个趋势跟踪型CTA基金经理，特点：
- 专注中长期趋势（周、月级别）
- 使用多周期技术分析确认趋势
- 顺势而为，绝不逆势操作
- 严格止损，让利润奔跑
- 系统化交易，纪律性强
- 关注趋势强度和持续性
            """,
            "high_frequency": """
你是一个高频交易算法系统，特点：
- 微秒到秒级的交易决策
- 捕捉短期价格波动和订单流不平衡
- 大量小额交易，单笔利润微薄但胜率高
- 极低的持仓时间（秒到分钟）
- 依赖流动性和做市商策略
- 对交易成本极其敏感
            """,
            "index_fund": """
你是一个指数基金的被动投资系统，特点：
- 跟踪指数成分股，追求贴近指数收益
- 被动投资策略，低换手率
- 定期再平衡，保持与指数权重一致
- 长期持有，不主动择时
- 关注跟踪误差最小化
- 成本优先，避免不必要的交易
            """
        }

        return personalities.get(self.agent_type, personalities["quantitative"])

    def analyze(self, stock_data: Dict[str, Any]) -> AgentDecision:
        """机构分析"""
        stock_code = stock_data.get('code', 'UNKNOWN')

        prompt = self._build_institutional_prompt(stock_data)
        system_prompt = self.get_personality_prompt()

        try:
            response = self.llm_client.structured_chat(
                prompt=prompt,
                system_prompt=system_prompt
            )

            action_str = response.get('action', '观望')
            if action_str in ['买入', 'buy', 'BUY']:
                action = ActionType.BUY
            elif action_str in ['卖出', 'sell', 'SELL']:
                action = ActionType.SELL
            else:
                action = ActionType.HOLD

            confidence = float(response.get('confidence', 0.5))
            reason = response.get('reason', '基于机构分析模型')

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

        except Exception as e:
            logger.error(f"LLM institutional analysis error: {e}")
            return self._fallback_analysis(stock_data)

    def _build_institutional_prompt(self, stock_data: Dict[str, Any]) -> str:
        """构建机构分析提示词"""
        prompt = f"""
作为一个专业的{self.agent_type}机构，请分析以下股票：

【标的信息】
{stock_data.get('code', 'N/A')} - {stock_data.get('name', 'N/A')}
价格：{stock_data.get('price', 0):.2f} 元（{stock_data.get('change_pct', 0):+.2f}%）

【量价数据】
成交额：{stock_data.get('amount', 0)/100000000:.2f} 亿
换手率：{stock_data.get('turnover_rate', 0):.2f}%
量比：{stock_data.get('volume_ratio', 1):.2f}
波动率：{stock_data.get('volatility', 0.02):.2%}

【技术面】
均线：MA5={stock_data.get('ma5', 0):.2f}, MA20={stock_data.get('ma20', 0):.2f}, MA60={stock_data.get('ma60', 0):.2f}
RSI：{stock_data.get('rsi', 50):.2f}
MACD：{stock_data.get('macd', 0):.4f}
当前价格位置：{"均线上方多头排列" if stock_data.get('price', 0) > stock_data.get('ma20', 0) else "均线下方空头排列"}

【基本面】
估值：PE={stock_data.get('pe_ratio', 30):.2f}, PB={stock_data.get('pb_ratio', 3):.2f}
盈利能力：ROE={stock_data.get('roe', 0.1):.2%}

请从你的机构视角出发，进行专业分析并给出：
1. 投资建议（买入/卖出/观望）
2. 详细的分析逻辑和理由
3. 决策置信度（0-1）
4. 关键风险点

保持你作为{self.agent_type}的专业特征和投资风格。
        """
        return prompt.strip()

    def _fallback_analysis(self, stock_data: Dict[str, Any]) -> AgentDecision:
        """降级分析"""
        from agents.institutional_agents import QuantitativeAgent

        fallback_agent = QuantitativeAgent(self.agent_id)
        return fallback_agent.analyze(stock_data)


class LLMEnhancedExpertPanel:
    """LLM增强的专家决策委员会 - 真正的多轮讨论"""

    def __init__(
        self,
        llm_client: LLMClient,
        discussion_rounds: int = 3
    ):
        self.llm_client = llm_client
        self.discussion_rounds = discussion_rounds
        self.experts = [
            "情绪分析专家",
            "机构行为专家",
            "风险控制专家",
            "市场时机专家"
        ]

        logger.info(f"LLMEnhancedExpertPanel initialized with {discussion_rounds} rounds")

    def make_decision(
        self,
        stock_data: Dict[str, Any],
        retail_decisions: List[AgentDecision],
        institutional_decisions: List[AgentDecision]
    ) -> PanelDecision:
        """
        通过LLM进行多轮真实讨论

        Args:
            stock_data: 股票数据
            retail_decisions: 散户决策
            institutional_decisions: 机构决策

        Returns:
            PanelDecision
        """
        stock_code = stock_data.get('code', 'UNKNOWN')

        logger.info(f"Expert panel deliberating on {stock_code}...")

        # 准备讨论上下文
        context = self._prepare_context(stock_data, retail_decisions, institutional_decisions)

        # 多轮讨论
        discussion_history = []

        for round_num in range(self.discussion_rounds):
            logger.info(f"  Round {round_num + 1}/{self.discussion_rounds}")

            round_opinions = []

            for expert in self.experts:
                opinion = self._expert_discuss(
                    expert=expert,
                    context=context,
                    previous_discussions=discussion_history
                )
                round_opinions.append(opinion)
                discussion_history.append({
                    'round': round_num + 1,
                    'expert': expert,
                    'opinion': opinion
                })

        # 最终决策
        final_decision = self._synthesize_decision(
            stock_code,
            context,
            discussion_history,
            retail_decisions,
            institutional_decisions
        )

        return final_decision

    def _prepare_context(
        self,
        stock_data: Dict[str, Any],
        retail_decisions: List[AgentDecision],
        institutional_decisions: List[AgentDecision]
    ) -> str:
        """准备讨论上下文"""
        # 统计散户情绪
        retail_buy = sum(1 for d in retail_decisions if d.action == ActionType.BUY)
        retail_sell = sum(1 for d in retail_decisions if d.action == ActionType.SELL)
        retail_total = len(retail_decisions)

        # 统计机构态度
        inst_buy = sum(1 for d in institutional_decisions if d.action == ActionType.BUY)
        inst_sell = sum(1 for d in institutional_decisions if d.action == ActionType.SELL)
        inst_total = len(institutional_decisions)

        context = f"""
【股票信息】
代码：{stock_data.get('code')} {stock_data.get('name')}
价格：{stock_data.get('price'):.2f}元（{stock_data.get('change_pct'):+.2f}%）
成交额：{stock_data.get('amount', 0)/100000000:.2f}亿  换手率：{stock_data.get('turnover_rate', 0):.2f}%

【散户情绪】（共{retail_total}人）
买入：{retail_buy}人（{retail_buy/retail_total*100:.1f}%）
卖出：{retail_sell}人（{retail_sell/retail_total*100:.1f}%）
观望：{retail_total-retail_buy-retail_sell}人

【机构态度】（共{inst_total}家）
买入：{inst_buy}家（{inst_buy/inst_total*100:.1f}%）
卖出：{inst_sell}家（{inst_sell/inst_total*100:.1f}%）
观望：{inst_total-inst_buy-inst_sell}家

【技术面】
MA20：{stock_data.get('ma20', 0):.2f}  RSI：{stock_data.get('rsi', 50):.1f}
MACD：{stock_data.get('macd', 0):.4f}  波动率：{stock_data.get('volatility', 0.02):.2%}

【基本面】
PE：{stock_data.get('pe_ratio', 30):.2f}  PB：{stock_data.get('pb_ratio', 3):.2f}  ROE：{stock_data.get('roe', 0.1):.2%}
        """
        return context.strip()

    def _expert_discuss(
        self,
        expert: str,
        context: str,
        previous_discussions: List[Dict[str, Any]]
    ) -> str:
        """专家讨论"""
        # 构建讨论历史
        history_text = ""
        if previous_discussions:
            history_text = "\n【之前的讨论】\n"
            for disc in previous_discussions[-4:]:  # 只看最近4条
                history_text += f"{disc['expert']}（第{disc['round']}轮）：{disc['opinion'][:200]}...\n"

        expert_prompts = {
            "情绪分析专家": "你是情绪分析专家，专门分析散户群体的心理状态和情绪倾向。请重点关注散户的贪婪/恐慌程度，以及是否存在过度一致的情绪（极端信号）。",
            "机构行为专家": "你是机构行为专家，专门解读大型机构的资金流向和真实意图。请分析机构的操作模式，判断是建仓、加仓、还是出货阶段。",
            "风险控制专家": "你是风险控制专家，负责识别和评估各种交易风险。请评估波动性风险、流动性风险、估值风险等，给出风险等级。",
            "市场时机专家": "你是市场时机专家，专门判断买卖时机。请分析当前是否是好的入场时机，考虑技术面、情绪面和基本面的综合时机。"
        }

        prompt = f"""
{expert_prompts[expert]}

{context}

{history_text}

请从你的专业角度分析，给出：
1. 你的核心观点（50字以内）
2. 支持理由
3. 你建议的操作方向（买入/卖出/观望）
        """

        try:
            response = self.llm_client.chat(
                prompt=prompt,
                system_prompt=expert_prompts[expert]
            )
            return response
        except Exception as e:
            logger.error(f"Expert {expert} discussion error: {e}")
            return f"{expert}：分析遇到问题，建议谨慎"

    def _synthesize_decision(
        self,
        stock_code: str,
        context: str,
        discussion_history: List[Dict[str, Any]],
        retail_decisions: List[AgentDecision],
        institutional_decisions: List[AgentDecision]
    ) -> PanelDecision:
        """综合所有讨论，做出最终决策"""
        # 整理所有专家意见
        all_opinions_text = "\n\n".join([
            f"{d['expert']}（第{d['round']}轮）：\n{d['opinion']}"
            for d in discussion_history
        ])

        synthesis_prompt = f"""
作为专家委员会主席，你需要综合所有专家的意见，做出最终决策。

{context}

【专家讨论内容】
{all_opinions_text}

请综合所有信息，给出最终决策：
1. 最终行动（买入/卖出/观望）
2. 决策理由（综合各专家观点）
3. 决策置信度（0-1）
4. 专家共识度评估（0-1，专家意见的一致程度）

请以JSON格式返回：
{{
    "action": "买入/卖出/观望",
    "confidence": 0.75,
    "consensus": 0.8,
    "reasoning": "综合理由"
}}
        """

        try:
            response = self.llm_client.structured_chat(synthesis_prompt)

            action_str = response.get('action', '观望')
            if action_str in ['买入', 'buy', 'BUY']:
                final_action = ActionType.BUY
            elif action_str in ['卖出', 'sell', 'SELL']:
                final_action = ActionType.SELL
            else:
                final_action = ActionType.HOLD

            confidence = float(response.get('confidence', 0.5))
            consensus = float(response.get('consensus', 0.5))
            reasoning = response.get('reasoning', '专家委员会综合决策')

            # 创建专家意见对象
            expert_opinions = []
            for disc in discussion_history[-len(self.experts):]:  # 最后一轮
                expert_opinions.append(ExpertOpinion(
                    expert_name=disc['expert'],
                    opinion=disc['opinion'],
                    confidence=confidence,
                    vote=final_action,
                    key_factors=[reasoning[:100]]
                ))

            # 计算情绪
            retail_sentiment = self._calculate_sentiment(retail_decisions)
            inst_sentiment = self._calculate_sentiment(institutional_decisions)

            return PanelDecision(
                stock_code=stock_code,
                final_action=final_action,
                confidence=confidence,
                consensus_level=consensus,
                reasoning=reasoning,
                expert_opinions=expert_opinions,
                retail_sentiment=retail_sentiment,
                institutional_sentiment=inst_sentiment,
                risk_assessment="LLM分析"
            )

        except Exception as e:
            logger.error(f"Decision synthesis error: {e}")
            # 降级到简单投票
            return self._simple_vote_decision(
                stock_code,
                retail_decisions,
                institutional_decisions
            )

    def _calculate_sentiment(self, decisions: List[AgentDecision]) -> Dict[str, Any]:
        """计算情绪统计"""
        if not decisions:
            return {'total': 0, 'buy_ratio': 0, 'sell_ratio': 0, 'hold_ratio': 0, 'avg_confidence': 0}

        total = len(decisions)
        buy = sum(1 for d in decisions if d.action == ActionType.BUY)
        sell = sum(1 for d in decisions if d.action == ActionType.SELL)
        hold = total - buy - sell
        avg_conf = sum(d.confidence for d in decisions) / total

        return {
            'total': total,
            'buy_ratio': buy / total,
            'sell_ratio': sell / total,
            'hold_ratio': hold / total,
            'avg_confidence': avg_conf
        }

    def _simple_vote_decision(
        self,
        stock_code: str,
        retail_decisions: List[AgentDecision],
        inst_decisions: List[AgentDecision]
    ) -> PanelDecision:
        """简单投票决策（降级方案）"""
        from agents.expert_panel import ExpertPanel

        fallback_panel = ExpertPanel(llm_client=None, discussion_rounds=1)
        return fallback_panel.make_decision({}, retail_decisions, inst_decisions)

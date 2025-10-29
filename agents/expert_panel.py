"""
专家决策委员会
通过多轮讨论，综合所有Agent的意见，做出最终交易决策
"""
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
from .base_agent import BaseAgent, AgentRole, AgentDecision, ActionType


@dataclass
class ExpertOpinion:
    """专家意见"""
    expert_name: str
    opinion: str
    confidence: float
    vote: ActionType
    key_factors: List[str]
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class PanelDecision:
    """专家委员会决策"""
    stock_code: str
    final_action: ActionType
    confidence: float
    consensus_level: float  # 共识度 0-1
    reasoning: str
    expert_opinions: List[ExpertOpinion]
    retail_sentiment: Dict[str, Any]
    institutional_sentiment: Dict[str, Any]
    risk_assessment: str
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        return {
            'stock_code': self.stock_code,
            'final_action': self.final_action.value,
            'confidence': self.confidence,
            'consensus_level': self.consensus_level,
            'reasoning': self.reasoning,
            'expert_opinions': [
                {
                    'expert_name': op.expert_name,
                    'opinion': op.opinion,
                    'confidence': op.confidence,
                    'vote': op.vote.value
                }
                for op in self.expert_opinions
            ],
            'retail_sentiment': self.retail_sentiment,
            'institutional_sentiment': self.institutional_sentiment,
            'risk_assessment': self.risk_assessment,
            'timestamp': self.timestamp.isoformat()
        }


class Expert:
    """专家基类"""

    def __init__(self, name: str, expertise: str, llm_client=None):
        self.name = name
        self.expertise = expertise
        self.llm_client = llm_client

    def analyze(
        self,
        stock_data: Dict[str, Any],
        retail_decisions: List[AgentDecision],
        institutional_decisions: List[AgentDecision],
        previous_opinions: Optional[List[ExpertOpinion]] = None
    ) -> ExpertOpinion:
        """
        分析并给出专家意见

        Args:
            stock_data: 股票数据
            retail_decisions: 散户决策列表
            institutional_decisions: 机构决策列表
            previous_opinions: 之前轮次的专家意见

        Returns:
            ExpertOpinion
        """
        raise NotImplementedError


class SentimentAnalystExpert(Expert):
    """情绪分析专家"""

    def __init__(self, llm_client=None):
        super().__init__("情绪分析专家", "散户情绪分析", llm_client)

    def analyze(
        self,
        stock_data: Dict[str, Any],
        retail_decisions: List[AgentDecision],
        institutional_decisions: List[AgentDecision],
        previous_opinions: Optional[List[ExpertOpinion]] = None
    ) -> ExpertOpinion:
        """分析散户情绪"""
        stock_code = stock_data.get('code', 'UNKNOWN')

        # 统计散户决策分布
        total_retail = len(retail_decisions)
        if total_retail == 0:
            return ExpertOpinion(
                expert_name=self.name,
                opinion="无散户数据",
                confidence=0.5,
                vote=ActionType.HOLD,
                key_factors=[]
            )

        buy_count = sum(1 for d in retail_decisions if d.action == ActionType.BUY)
        sell_count = sum(1 for d in retail_decisions if d.action == ActionType.SELL)
        hold_count = sum(1 for d in retail_decisions if d.action == ActionType.HOLD)

        buy_ratio = buy_count / total_retail
        sell_ratio = sell_count / total_retail

        # 计算情绪指标
        sentiment_score = buy_ratio - sell_ratio  # -1到1

        # 分析情绪类型
        key_factors = []
        if buy_ratio > 0.7:
            emotion_type = "极度贪婪"
            key_factors.append("散户狂热追涨，可能是顶部信号")
            # 反向操作：散户太贪婪时，专家建议卖出
            vote = ActionType.SELL
            confidence = 0.75
        elif buy_ratio > 0.55:
            emotion_type = "偏向贪婪"
            key_factors.append("散户买入意愿较强，需警惕")
            vote = ActionType.HOLD
            confidence = 0.6
        elif sell_ratio > 0.7:
            emotion_type = "极度恐慌"
            key_factors.append("散户恐慌性抛售，可能是底部机会")
            # 反向操作：散户太恐慌时，专家建议买入
            vote = ActionType.BUY
            confidence = 0.75
        elif sell_ratio > 0.55:
            emotion_type = "偏向恐慌"
            key_factors.append("散户卖出意愿较强，可能有机会")
            vote = ActionType.BUY
            confidence = 0.6
        else:
            emotion_type = "相对平衡"
            key_factors.append("散户情绪较为平衡")
            vote = ActionType.HOLD
            confidence = 0.5

        # 分析追涨杀跌型和恐慌型的占比
        momentum_chasers = sum(1 for d in retail_decisions if d.agent_type == "momentum_chaser")
        panic_sellers = sum(1 for d in retail_decisions if d.agent_type == "panic_seller")

        if momentum_chasers > total_retail * 0.3 and buy_ratio > 0.6:
            key_factors.append("大量追涨盘，警惕追高风险")

        if panic_sellers > total_retail * 0.3 and sell_ratio > 0.6:
            key_factors.append("恐慌盘集中出逃，可能是抄底机会")

        opinion = f"""
        【散户情绪分析】{stock_code}
        - 情绪类型：{emotion_type}
        - 买入比例：{buy_ratio:.1%} ({buy_count}人)
        - 卖出比例：{sell_ratio:.1%} ({sell_count}人)
        - 观望比例：{hold_count/total_retail:.1%} ({hold_count}人)
        - 情绪得分：{sentiment_score:.2f}

        结论：{key_factors[0] if key_factors else '情绪平稳'}
        建议：{"反向操作，考虑卖出" if vote == ActionType.SELL else "反向操作，考虑买入" if vote == ActionType.BUY else "观望"}
        """

        return ExpertOpinion(
            expert_name=self.name,
            opinion=opinion,
            confidence=confidence,
            vote=vote,
            key_factors=key_factors
        )


class InstitutionalFlowExpert(Expert):
    """机构行为专家"""

    def __init__(self, llm_client=None):
        super().__init__("机构行为专家", "机构资金流向分析", llm_client)

    def analyze(
        self,
        stock_data: Dict[str, Any],
        retail_decisions: List[AgentDecision],
        institutional_decisions: List[AgentDecision],
        previous_opinions: Optional[List[ExpertOpinion]] = None
    ) -> ExpertOpinion:
        """分析机构资金流向"""
        stock_code = stock_data.get('code', 'UNKNOWN')

        total_inst = len(institutional_decisions)
        if total_inst == 0:
            return ExpertOpinion(
                expert_name=self.name,
                opinion="无机构数据",
                confidence=0.5,
                vote=ActionType.HOLD,
                key_factors=[]
            )

        # 统计机构决策
        buy_count = sum(1 for d in institutional_decisions if d.action == ActionType.BUY)
        sell_count = sum(1 for d in institutional_decisions if d.action == ActionType.SELL)
        hold_count = sum(1 for d in institutional_decisions if d.action == ActionType.HOLD)

        buy_ratio = buy_count / total_inst
        sell_ratio = sell_count / total_inst

        # 计算机构一致性
        agreement_score = max(buy_ratio, sell_ratio, hold_count / total_inst)

        # 分析不同类型机构的态度
        key_factors = []
        inst_by_type = {}

        for decision in institutional_decisions:
            agent_type = decision.agent_type
            if agent_type not in inst_by_type:
                inst_by_type[agent_type] = {'buy': 0, 'sell': 0, 'hold': 0}
            inst_by_type[agent_type][decision.action.value] += 1

        # 重点关注价值投资和量化机构
        if 'value_investor' in inst_by_type:
            value_buy = inst_by_type['value_investor']['buy']
            if value_buy > 0:
                key_factors.append(f"价值投资机构看好，有{value_buy}家机构买入")

        if 'quantitative' in inst_by_type:
            quant_buy = inst_by_type['quantitative']['buy']
            quant_sell = inst_by_type['quantitative']['sell']
            if quant_buy > quant_sell:
                key_factors.append(f"量化模型偏多，{quant_buy}买/{quant_sell}卖")
            elif quant_sell > quant_buy:
                key_factors.append(f"量化模型偏空，{quant_sell}卖/{quant_buy}买")

        # 决策逻辑：顺应机构方向
        if buy_ratio > 0.65 and agreement_score > 0.65:
            vote = ActionType.BUY
            confidence = 0.80
            key_factors.insert(0, f"机构高度一致看多({buy_ratio:.1%})")
        elif sell_ratio > 0.65 and agreement_score > 0.65:
            vote = ActionType.SELL
            confidence = 0.80
            key_factors.insert(0, f"机构高度一致看空({sell_ratio:.1%})")
        elif buy_ratio > 0.55:
            vote = ActionType.BUY
            confidence = 0.65
            key_factors.insert(0, f"机构偏向看多({buy_ratio:.1%})")
        elif sell_ratio > 0.55:
            vote = ActionType.SELL
            confidence = 0.65
            key_factors.insert(0, f"机构偏向看空({sell_ratio:.1%})")
        else:
            vote = ActionType.HOLD
            confidence = 0.5
            key_factors.insert(0, "机构意见分歧，方向不明")

        opinion = f"""
        【机构资金流向分析】{stock_code}
        - 买入比例：{buy_ratio:.1%} ({buy_count}家)
        - 卖出比例：{sell_ratio:.1%} ({sell_count}家)
        - 观望比例：{hold_count/total_inst:.1%} ({hold_count}家)
        - 一致性得分：{agreement_score:.2f}

        机构态度：
        {chr(10).join(f'  • {factor}' for factor in key_factors)}

        建议：{"跟随机构买入" if vote == ActionType.BUY else "跟随机构卖出" if vote == ActionType.SELL else "机构分歧，等待明确信号"}
        """

        return ExpertOpinion(
            expert_name=self.name,
            opinion=opinion,
            confidence=confidence,
            vote=vote,
            key_factors=key_factors
        )


class RiskControlExpert(Expert):
    """风险控制专家"""

    def __init__(self, llm_client=None):
        super().__init__("风险控制专家", "风险评估与控制", llm_client)

    def analyze(
        self,
        stock_data: Dict[str, Any],
        retail_decisions: List[AgentDecision],
        institutional_decisions: List[AgentDecision],
        previous_opinions: Optional[List[ExpertOpinion]] = None
    ) -> ExpertOpinion:
        """风险评估"""
        stock_code = stock_data.get('code', 'UNKNOWN')

        risk_factors = []
        risk_score = 0  # 0-100，越高越危险

        # 1. 波动率风险
        volatility = stock_data.get('volatility', 0.02)
        if volatility > 0.05:
            risk_factors.append(f"高波动率{volatility:.2%}，风险较大")
            risk_score += 25
        elif volatility > 0.03:
            risk_factors.append(f"中等波动率{volatility:.2%}")
            risk_score += 15

        # 2. 流动性风险
        volume_ratio = stock_data.get('volume_ratio', 1)
        turnover_rate = stock_data.get('turnover_rate', 1)
        if volume_ratio < 0.5 or turnover_rate < 0.5:
            risk_factors.append("流动性不足，可能难以成交")
            risk_score += 20
        elif volume_ratio > 5:
            risk_factors.append("成交量异常放大，需警惕")
            risk_score += 15

        # 3. 价格位置风险
        change_pct = stock_data.get('change_pct', 0)
        if change_pct > 8:
            risk_factors.append(f"短期涨幅过大({change_pct:.1f}%)，警惕回调")
            risk_score += 20
        elif change_pct < -8:
            risk_factors.append(f"短期跌幅过大({abs(change_pct):.1f}%)，可能继续下跌")
            risk_score += 15

        # 4. 估值风险
        pe_ratio = stock_data.get('pe_ratio', 30)
        if pe_ratio > 50:
            risk_factors.append(f"市盈率过高({pe_ratio:.1f})，估值风险大")
            risk_score += 20
        elif pe_ratio < 0:
            risk_factors.append("公司亏损，基本面风险")
            risk_score += 25

        # 5. 市场情绪风险
        total_agents = len(retail_decisions) + len(institutional_decisions)
        if total_agents > 0:
            buy_count = sum(1 for d in retail_decisions + institutional_decisions if d.action == ActionType.BUY)
            buy_ratio = buy_count / total_agents

            if buy_ratio > 0.8:
                risk_factors.append("市场过度一致看多，可能是反转信号")
                risk_score += 15
            elif buy_ratio < 0.2:
                risk_factors.append("市场过度悲观，虽有机会但风险也大")
                risk_score += 10

        # 综合风险评估
        if risk_score >= 70:
            risk_level = "高风险"
            vote = ActionType.SELL  # 高风险时建议规避
            confidence = 0.75
        elif risk_score >= 40:
            risk_level = "中等风险"
            vote = ActionType.HOLD
            confidence = 0.6
        else:
            risk_level = "低风险"
            vote = ActionType.HOLD
            confidence = 0.5

        if not risk_factors:
            risk_factors.append("风险指标正常")

        opinion = f"""
        【风险评估】{stock_code}
        - 风险等级：{risk_level}
        - 风险得分：{risk_score}/100

        风险因素：
        {chr(10).join(f'  • {factor}' for factor in risk_factors)}

        风控建议：{"风险过高，建议规避" if risk_score >= 70 else "风险可控，可以参与" if risk_score < 40 else "中等风险，谨慎参与"}
        """

        return ExpertOpinion(
            expert_name=self.name,
            opinion=opinion,
            confidence=confidence,
            vote=vote,
            key_factors=risk_factors
        )


class TimingExpert(Expert):
    """市场时机专家"""

    def __init__(self, llm_client=None):
        super().__init__("市场时机专家", "入场时机判断", llm_client)

    def analyze(
        self,
        stock_data: Dict[str, Any],
        retail_decisions: List[AgentDecision],
        institutional_decisions: List[AgentDecision],
        previous_opinions: Optional[List[ExpertOpinion]] = None
    ) -> ExpertOpinion:
        """判断入场时机"""
        stock_code = stock_data.get('code', 'UNKNOWN')

        timing_factors = []
        timing_score = 50  # 50为中性

        # 1. 技术面时机
        ma5 = stock_data.get('ma5', 0)
        ma20 = stock_data.get('ma20', 0)
        price = stock_data.get('price', 0)

        if price > ma5 > ma20:
            timing_factors.append("均线多头排列，技术面良好")
            timing_score += 15
        elif price < ma5 < ma20:
            timing_factors.append("均线空头排列，技术面较弱")
            timing_score -= 15

        # 2. RSI超买超卖
        rsi = stock_data.get('rsi', 50)
        if rsi < 30:
            timing_factors.append(f"RSI={rsi:.1f}，超卖区域，可能反弹")
            timing_score += 20
        elif rsi > 70:
            timing_factors.append(f"RSI={rsi:.1f}，超买区域，可能回调")
            timing_score -= 20

        # 3. 成交量配合
        volume_ratio = stock_data.get('volume_ratio', 1)
        change_pct = stock_data.get('change_pct', 0)

        if change_pct > 2 and volume_ratio > 1.5:
            timing_factors.append("放量上涨，多头力量强")
            timing_score += 15
        elif change_pct < -2 and volume_ratio > 1.5:
            timing_factors.append("放量下跌，空头力量强")
            timing_score -= 15
        elif change_pct > 2 and volume_ratio < 0.8:
            timing_factors.append("缩量上涨，上涨动能不足")
            timing_score -= 10

        # 4. 机构与散户的分歧度
        if len(retail_decisions) > 0 and len(institutional_decisions) > 0:
            retail_buy_ratio = sum(1 for d in retail_decisions if d.action == ActionType.BUY) / len(retail_decisions)
            inst_buy_ratio = sum(1 for d in institutional_decisions if d.action == ActionType.BUY) / len(institutional_decisions)

            divergence = abs(retail_buy_ratio - inst_buy_ratio)
            if divergence > 0.4:
                if inst_buy_ratio > retail_buy_ratio:
                    timing_factors.append("机构买入而散户卖出，是好的入场时机")
                    timing_score += 20
                else:
                    timing_factors.append("散户买入而机构卖出，需谨慎")
                    timing_score -= 15

        # 综合判断时机
        if timing_score >= 70:
            timing = "绝佳时机"
            vote = ActionType.BUY
            confidence = 0.80
        elif timing_score >= 55:
            timing = "较好时机"
            vote = ActionType.BUY
            confidence = 0.65
        elif timing_score <= 30:
            timing = "糟糕时机"
            vote = ActionType.SELL
            confidence = 0.75
        elif timing_score <= 45:
            timing = "一般时机"
            vote = ActionType.HOLD
            confidence = 0.55
        else:
            timing = "中性时机"
            vote = ActionType.HOLD
            confidence = 0.5

        if not timing_factors:
            timing_factors.append("时机指标中性")

        opinion = f"""
        【市场时机分析】{stock_code}
        - 时机评级：{timing}
        - 时机得分：{timing_score}/100

        时机因素：
        {chr(10).join(f'  • {factor}' for factor in timing_factors)}

        时机建议：{"当前是买入的好时机" if timing_score >= 55 else "当前不是好的入场时机" if timing_score <= 40 else "时机一般，可等待更好机会"}
        """

        return ExpertOpinion(
            expert_name=self.name,
            opinion=opinion,
            confidence=confidence,
            vote=vote,
            key_factors=timing_factors
        )


class ExpertPanel:
    """专家决策委员会"""

    def __init__(self, llm_client=None, discussion_rounds: int = 3):
        """
        初始化专家委员会

        Args:
            llm_client: LLM客户端
            discussion_rounds: 讨论轮次
        """
        self.llm_client = llm_client
        self.discussion_rounds = discussion_rounds

        # 创建专家
        self.experts = [
            SentimentAnalystExpert(llm_client),
            InstitutionalFlowExpert(llm_client),
            RiskControlExpert(llm_client),
            TimingExpert(llm_client)
        ]

    def make_decision(
        self,
        stock_data: Dict[str, Any],
        retail_decisions: List[AgentDecision],
        institutional_decisions: List[AgentDecision]
    ) -> PanelDecision:
        """
        通过多轮讨论做出最终决策

        Args:
            stock_data: 股票数据
            retail_decisions: 散户决策列表
            institutional_decisions: 机构决策列表

        Returns:
            PanelDecision
        """
        stock_code = stock_data.get('code', 'UNKNOWN')

        # 多轮讨论
        all_opinions = []
        previous_opinions = None

        for round_num in range(self.discussion_rounds):
            round_opinions = []
            for expert in self.experts:
                opinion = expert.analyze(
                    stock_data,
                    retail_decisions,
                    institutional_decisions,
                    previous_opinions
                )
                round_opinions.append(opinion)

            all_opinions.extend(round_opinions)
            previous_opinions = round_opinions

        # 最后一轮的意见作为最终意见
        final_opinions = all_opinions[-len(self.experts):]

        # 统计专家投票
        votes = [op.vote for op in final_opinions]
        buy_votes = sum(1 for v in votes if v == ActionType.BUY)
        sell_votes = sum(1 for v in votes if v == ActionType.SELL)
        hold_votes = sum(1 for v in votes if v == ActionType.HOLD)

        # 决定最终行动
        if buy_votes > sell_votes and buy_votes > hold_votes:
            final_action = ActionType.BUY
        elif sell_votes > buy_votes and sell_votes > hold_votes:
            final_action = ActionType.SELL
        else:
            final_action = ActionType.HOLD

        # 计算共识度
        max_votes = max(buy_votes, sell_votes, hold_votes)
        consensus_level = max_votes / len(self.experts)

        # 计算综合置信度
        action_opinions = [op for op in final_opinions if op.vote == final_action]
        if action_opinions:
            confidence = sum(op.confidence for op in action_opinions) / len(action_opinions)
        else:
            confidence = 0.5

        # 生成综合推理
        reasoning = self._generate_reasoning(
            final_action,
            final_opinions,
            consensus_level,
            confidence
        )

        # 统计散户和机构情绪
        retail_sentiment = self._calculate_sentiment(retail_decisions)
        institutional_sentiment = self._calculate_sentiment(institutional_decisions)

        # 风险评估
        risk_expert_opinion = next((op for op in final_opinions if op.expert_name == "风险控制专家"), None)
        risk_assessment = risk_expert_opinion.opinion if risk_expert_opinion else "风险评估缺失"

        return PanelDecision(
            stock_code=stock_code,
            final_action=final_action,
            confidence=confidence,
            consensus_level=consensus_level,
            reasoning=reasoning,
            expert_opinions=final_opinions,
            retail_sentiment=retail_sentiment,
            institutional_sentiment=institutional_sentiment,
            risk_assessment=risk_assessment
        )

    def _generate_reasoning(
        self,
        action: ActionType,
        opinions: List[ExpertOpinion],
        consensus: float,
        confidence: float
    ) -> str:
        """生成综合推理"""
        action_str = "买入" if action == ActionType.BUY else "卖出" if action == ActionType.SELL else "观望"

        reasoning_parts = [
            f"【专家委员会最终决策：{action_str}】",
            f"置信度：{confidence:.2%} | 共识度：{consensus:.2%}",
            "",
            "各专家意见汇总："
        ]

        for opinion in opinions:
            vote_str = "买入" if opinion.vote == ActionType.BUY else "卖出" if opinion.vote == ActionType.SELL else "观望"
            reasoning_parts.append(
                f"  • {opinion.expert_name}：{vote_str}（置信度{opinion.confidence:.2%}）"
            )
            if opinion.key_factors:
                reasoning_parts.append(f"    - {opinion.key_factors[0]}")

        reasoning_parts.append("")
        reasoning_parts.append(f"综合判断：专家委员会经过{len(opinions)}位专家的分析讨论，")

        if consensus >= 0.75:
            reasoning_parts.append(f"高度一致（{consensus:.0%}）认为应该{action_str}。")
        elif consensus >= 0.5:
            reasoning_parts.append(f"多数专家（{consensus:.0%}）倾向于{action_str}。")
        else:
            reasoning_parts.append(f"专家意见存在分歧，但综合判断建议{action_str}。")

        return "\n".join(reasoning_parts)

    def _calculate_sentiment(self, decisions: List[AgentDecision]) -> Dict[str, Any]:
        """计算情绪统计"""
        if not decisions:
            return {
                'total': 0,
                'buy_ratio': 0,
                'sell_ratio': 0,
                'hold_ratio': 0,
                'avg_confidence': 0
            }

        total = len(decisions)
        buy_count = sum(1 for d in decisions if d.action == ActionType.BUY)
        sell_count = sum(1 for d in decisions if d.action == ActionType.SELL)
        hold_count = sum(1 for d in decisions if d.action == ActionType.HOLD)
        avg_confidence = sum(d.confidence for d in decisions) / total

        return {
            'total': total,
            'buy_ratio': buy_count / total,
            'sell_ratio': sell_count / total,
            'hold_ratio': hold_count / total,
            'avg_confidence': avg_confidence
        }

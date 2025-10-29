"""
信号聚合器
聚合多个股票的策略信号，排序和筛选
"""
from typing import List, Dict, Any, Optional
from loguru import logger
from agents.expert_panel import PanelDecision
from agents.base_agent import ActionType


class SignalAggregator:
    """信号聚合器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化

        Args:
            config: 配置
        """
        self.config = config or {}
        self.min_score = self.config.get('min_score', 70)
        self.max_opportunities = self.config.get('max_opportunities', 10)

        logger.info("SignalAggregator initialized")

    def aggregate_signals(
        self,
        decisions: List[PanelDecision],
        stock_analyses: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        聚合信号

        Args:
            decisions: 专家决策列表
            stock_analyses: 股票分析数据

        Returns:
            聚合后的交易机会列表
        """
        opportunities = []

        for decision in decisions:
            stock_code = decision.stock_code

            # 获取股票分析数据
            analysis = stock_analyses.get(stock_code, {})

            # 计算综合评分
            score = self._calculate_composite_score(decision, analysis)

            if score < self.min_score:
                continue

            # 构建机会对象
            opportunity = {
                'stock_code': stock_code,
                'stock_name': analysis.get('name', ''),
                'action': decision.final_action.value,
                'score': score,
                'confidence': decision.confidence,
                'consensus': decision.consensus_level,
                'price': analysis.get('price', 0),
                'change_pct': analysis.get('change_pct', 0),
                'volume_ratio': analysis.get('volume_ratio', 1),
                'decision': decision,
                'analysis': analysis
            }

            opportunities.append(opportunity)

        # 按评分排序
        opportunities.sort(key=lambda x: x['score'], reverse=True)

        # 限制数量
        opportunities = opportunities[:self.max_opportunities]

        logger.info(f"Aggregated {len(opportunities)} opportunities")

        return opportunities

    def _calculate_composite_score(
        self,
        decision: PanelDecision,
        analysis: Dict[str, Any]
    ) -> float:
        """
        计算综合评分 (0-100)

        Args:
            decision: 专家决策
            analysis: 股票分析

        Returns:
            综合评分
        """
        score = 50  # 基准分

        # 1. 专家决策得分 (40分)
        # 置信度
        score += decision.confidence * 20

        # 共识度
        score += decision.consensus_level * 20

        # 2. 情绪分歧得分 (20分)
        retail_buy = decision.retail_sentiment['buy_ratio']
        inst_buy = decision.institutional_sentiment['buy_ratio']
        divergence = abs(retail_buy - inst_buy)

        # 分歧越大，机会越好
        score += divergence * 20

        # 3. 技术面得分 (20分)
        opportunity_score = analysis.get('opportunity_score', 50)
        score += (opportunity_score - 50) * 0.4

        # 4. 风险调整 (20分)
        volatility = analysis.get('volatility', 0.02)
        if volatility < 0.02:
            score += 20
        elif volatility < 0.03:
            score += 10
        elif volatility > 0.05:
            score -= 10

        # 5. 流动性得分 (10分)
        volume_ratio = analysis.get('volume_ratio', 1)
        turnover_rate = analysis.get('turnover_rate', 1)
        if volume_ratio > 1.5 and turnover_rate > 2:
            score += 10
        elif volume_ratio > 1.0:
            score += 5

        # 限制在0-100范围
        score = max(0, min(100, score))

        return round(score, 2)

    def filter_by_industry(
        self,
        opportunities: List[Dict[str, Any]],
        max_per_industry: int = 2
    ) -> List[Dict[str, Any]]:
        """
        按行业分散

        Args:
            opportunities: 机会列表
            max_per_industry: 每个行业最大数量

        Returns:
            过滤后的列表
        """
        # 简化实现：这里需要实际的行业数据
        # 暂时返回原列表
        return opportunities

    def get_opportunities_summary(
        self,
        opportunities: List[Dict[str, Any]]
    ) -> str:
        """
        获取机会摘要

        Args:
            opportunities: 机会列表

        Returns:
            摘要字符串
        """
        if not opportunities:
            return "当前没有发现交易机会"

        buy_opps = [o for o in opportunities if o['action'] == 'buy']
        sell_opps = [o for o in opportunities if o['action'] == 'sell']

        summary_parts = [
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"【交易机会汇总】共{len(opportunities)}个机会",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            "",
            f"买入机会: {len(buy_opps)}个",
            f"卖出机会: {len(sell_opps)}个",
            ""
        ]

        if buy_opps:
            summary_parts.append("【买入机会】")
            for i, opp in enumerate(buy_opps[:5], 1):
                summary_parts.append(
                    f"{i}. {opp['stock_code']} {opp['stock_name']} "
                    f"评分:{opp['score']:.1f} "
                    f"置信度:{opp['confidence']:.1%} "
                    f"价格:{opp['price']:.2f}"
                )
            summary_parts.append("")

        if sell_opps:
            summary_parts.append("【卖出机会】")
            for i, opp in enumerate(sell_opps[:5], 1):
                summary_parts.append(
                    f"{i}. {opp['stock_code']} {opp['stock_name']} "
                    f"评分:{opp['score']:.1f} "
                    f"置信度:{opp['confidence']:.1%} "
                    f"价格:{opp['price']:.2f}"
                )
            summary_parts.append("")

        summary_parts.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        return "\n".join(summary_parts)

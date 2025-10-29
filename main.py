"""
AI股票多Agent自动交易系统 - 主程序
"""
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from loguru import logger

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils import setup_logger, load_config
from agents import (
    RetailAgentFactory,
    InstitutionalAgentFactory,
    ExpertPanel
)
from data import MarketDataProvider, StockAnalyzer
from trading import TradeExecutor, RiskManager, PositionManager, OrderSide, OrderType
from strategy import StrategyGenerator, SignalAggregator


class TradingSystem:
    """交易系统主类"""

    def __init__(self, config_path: str = "config.yaml"):
        """
        初始化交易系统

        Args:
            config_path: 配置文件路径
        """
        logger.info("=" * 60)
        logger.info("AI股票多Agent自动交易系统启动中...")
        logger.info("=" * 60)

        # 加载配置
        self.config = load_config(config_path)

        # 初始化日志
        log_config = self.config.get('logging', {})
        setup_logger(
            log_level=log_config.get('level', 'INFO'),
            log_file=log_config.get('output.file_path'),
            rotation=log_config.get('rotation', '1 day'),
            retention=log_config.get('retention', '30 days'),
            format_type=log_config.get('format', 'text')
        )

        # 初始化LLM客户端（可选）
        self.llm_client = None  # 如果需要使用LLM增强Agent决策，在这里初始化

        # 初始化Agent
        logger.info("初始化Agent系统...")
        self._init_agents()

        # 初始化数据模块
        logger.info("初始化数据模块...")
        self._init_data_modules()

        # 初始化交易模块
        logger.info("初始化交易模块...")
        self._init_trading_modules()

        # 初始化策略模块
        logger.info("初始化策略模块...")
        self._init_strategy_modules()

        logger.info("系统初始化完成！")
        logger.info("=" * 60)

    def _init_agents(self):
        """初始化Agent"""
        # 创建散户Agent
        retail_config = self.config.get('agents.retail.types', {})
        self.retail_agents = RetailAgentFactory.create_agents(
            retail_config,
            self.llm_client
        )
        logger.info(f"创建了 {len(self.retail_agents)} 个散户Agent")

        # 创建机构Agent
        inst_config = self.config.get('agents.institutional.types', {})
        self.institutional_agents = InstitutionalAgentFactory.create_agents(
            inst_config,
            self.llm_client
        )
        logger.info(f"创建了 {len(self.institutional_agents)} 个机构Agent")

        # 创建专家委员会
        discussion_rounds = self.config.get('agents.expert_panel.discussion_rounds', 3)
        self.expert_panel = ExpertPanel(
            llm_client=self.llm_client,
            discussion_rounds=discussion_rounds
        )
        logger.info(f"创建了专家决策委员会（{discussion_rounds}轮讨论）")

    def _init_data_modules(self):
        """初始化数据模块"""
        data_source = self.config.get('market_data.source', 'akshare')
        self.data_provider = MarketDataProvider(source=data_source)

        self.stock_analyzer = StockAnalyzer(self.data_provider)

    def _init_trading_modules(self):
        """初始化交易模块"""
        trading_config = self.config.get('trading', {})

        # 交易执行器
        self.trade_executor = TradeExecutor(
            mode=trading_config.get('mode', 'simulation'),
            broker_config=trading_config.get('broker', {}),
            commission_rate=0.0003
        )

        # 风险管理器
        risk_config = self.config.get('risk_management', {})
        self.risk_manager = RiskManager(
            initial_capital=1000000,  # 默认100万
            max_total_position=risk_config.get('limits.max_total_position', 0.8),
            max_single_position=risk_config.get('limits.max_single_position', 0.1),
            stop_loss_pct=risk_config.get('stop_loss.percentage', 0.05),
            stop_profit_pct=risk_config.get('stop_profit.percentage', 0.15),
            max_daily_trades=risk_config.get('limits.max_daily_trades', 20)
        )

        # 仓位管理器
        self.position_manager = PositionManager()

    def _init_strategy_modules(self):
        """初始化策略模块"""
        strategy_config = self.config.get('strategy', {})

        # 策略生成器
        self.strategy_generator = StrategyGenerator(
            retail_agents=self.retail_agents,
            institutional_agents=self.institutional_agents,
            expert_panel=self.expert_panel,
            config=strategy_config
        )

        # 信号聚合器
        self.signal_aggregator = SignalAggregator(
            config=strategy_config.get('opportunity_filter', {})
        )

    def run(self):
        """运行交易系统"""
        logger.info("开始运行交易系统...")

        try:
            # 1. 扫描市场机会
            logger.info("\n" + "=" * 60)
            logger.info("第一阶段：市场扫描")
            logger.info("=" * 60)

            opportunities = self._scan_market()

            if not opportunities:
                logger.info("未发现交易机会")
                return

            logger.info(f"发现 {len(opportunities)} 个潜在机会")

            # 2. 生成交易策略
            logger.info("\n" + "=" * 60)
            logger.info("第二阶段：策略生成")
            logger.info("=" * 60)

            decisions = self._generate_strategies(opportunities)

            if not decisions:
                logger.info("没有通过策略筛选的机会")
                return

            # 3. 聚合和排序信号
            logger.info("\n" + "=" * 60)
            logger.info("第三阶段：信号聚合")
            logger.info("=" * 60)

            stock_analyses = {opp['code']: opp for opp in opportunities}
            trade_opportunities = self.signal_aggregator.aggregate_signals(
                decisions,
                stock_analyses
            )

            # 显示机会摘要
            summary = self.signal_aggregator.get_opportunities_summary(trade_opportunities)
            logger.info(f"\n{summary}")

            # 4. 执行交易
            logger.info("\n" + "=" * 60)
            logger.info("第四阶段：交易执行")
            logger.info("=" * 60)

            self._execute_trades(trade_opportunities)

            # 5. 生成报告
            logger.info("\n" + "=" * 60)
            logger.info("第五阶段：生成报告")
            logger.info("=" * 60)

            self._generate_report()

        except Exception as e:
            logger.error(f"系统运行出错: {e}", exc_info=True)

        logger.info("\n" + "=" * 60)
        logger.info("交易系统运行完成")
        logger.info("=" * 60)

    def _scan_market(self) -> List[Dict[str, Any]]:
        """扫描市场机会"""
        logger.info("正在扫描全市场股票...")

        # 使用股票分析器寻找机会
        opportunities = self.stock_analyzer.find_opportunities(
            min_change_pct=-10,
            max_change_pct=10,
            min_volume_ratio=1.0,
            min_turnover_rate=1.0
        )

        return opportunities

    def _generate_strategies(
        self,
        opportunities: List[Dict[str, Any]]
    ) -> List[Any]:
        """生成交易策略"""
        decisions = []

        for i, stock_data in enumerate(opportunities[:20], 1):  # 限制分析数量
            stock_code = stock_data['code']
            logger.info(f"\n[{i}/{min(20, len(opportunities))}] 分析 {stock_code} {stock_data.get('name', '')}...")

            try:
                # 生成策略
                decision = self.strategy_generator.generate_strategy(stock_data)

                if decision:
                    decisions.append(decision)

                    # 显示策略摘要
                    summary = self.strategy_generator.get_signal_summary(decision)
                    logger.info(f"\n{summary}")

            except Exception as e:
                logger.error(f"分析 {stock_code} 时出错: {e}")

        return decisions

    def _execute_trades(self, opportunities: List[Dict[str, Any]]):
        """执行交易"""
        if not opportunities:
            logger.info("没有需要执行的交易")
            return

        # 检查是否允许交易
        allowed, reason = self.risk_manager.is_trading_allowed()
        if not allowed:
            logger.warning(f"交易被禁止: {reason}")
            return

        for opp in opportunities:
            stock_code = opp['stock_code']
            stock_name = opp['stock_name']
            action = opp['action']
            price = opp['price']

            logger.info(f"\n准备交易: {stock_code} {stock_name} {action.upper()}")

            try:
                # 计算交易数量
                quantity = self._calculate_quantity(opp)

                if quantity <= 0:
                    logger.info("计算出的交易数量为0，跳过")
                    continue

                # 创建订单
                side = OrderSide.BUY if action == 'buy' else OrderSide.SELL
                order = self.trade_executor.create_order(
                    stock_code=stock_code,
                    stock_name=stock_name,
                    side=side,
                    quantity=quantity,
                    price=price,
                    order_type=OrderType.LIMIT,
                    notes=f"AI策略：评分{opp['score']:.1f}，置信度{opp['confidence']:.1%}"
                )

                # 风险检查
                positions = self._get_current_positions()
                risk_ok, risk_reason = self.risk_manager.check_order_risk(
                    order,
                    positions,
                    price
                )

                if not risk_ok:
                    logger.warning(f"风险检查未通过: {risk_reason}")
                    continue

                # 提交订单
                success = self.trade_executor.submit_order(order)

                if success:
                    logger.info(f"订单已提交: {order.order_id}")

                    # 模拟成交
                    if self.trade_executor.mode == "simulation":
                        self.trade_executor.simulate_fill(order, price)
                        logger.info(f"订单已成交: {quantity}股@{price:.2f}元")

                        # 更新持仓
                        for trade in self.trade_executor.trades:
                            if trade.order_id == order.order_id:
                                self.position_manager.update_from_trade(trade, stock_name)

                else:
                    logger.error(f"订单提交失败")

            except Exception as e:
                logger.error(f"执行交易 {stock_code} 时出错: {e}")

    def _calculate_quantity(self, opportunity: Dict[str, Any]) -> int:
        """计算交易数量"""
        price = opportunity['price']
        confidence = opportunity['confidence']

        # 基于资金和仓位限制计算
        available_cash = self.risk_manager._calculate_available_cash(
            self._get_current_positions()
        )

        # 根据置信度分配资金
        allocation = available_cash * confidence * 0.1  # 最多10%的可用资金

        # 计算股数（100股的整数倍）
        quantity = int(allocation / price / 100) * 100

        return quantity

    def _get_current_positions(self) -> Dict[str, Dict[str, Any]]:
        """获取当前持仓（转换为风险管理器需要的格式）"""
        positions = {}
        for stock_code, position in self.position_manager.get_all_positions().items():
            positions[stock_code] = {
                'quantity': position.quantity,
                'cost_price': position.cost_price,
                'current_price': position.current_price
            }
        return positions

    def _generate_report(self):
        """生成交易报告"""
        logger.info("\n" + "=" * 60)
        logger.info("交易统计报告")
        logger.info("=" * 60)

        # 交易统计
        trade_stats = self.trade_executor.get_statistics()
        logger.info(f"\n【交易统计】")
        logger.info(f"  总订单数: {trade_stats['total_orders']}")
        logger.info(f"  成交订单: {trade_stats['filled_orders']}")
        logger.info(f"  成交率: {trade_stats['fill_rate']:.1%}")
        logger.info(f"  买入金额: {trade_stats['total_buy_amount']:.2f}元")
        logger.info(f"  卖出金额: {trade_stats['total_sell_amount']:.2f}元")
        logger.info(f"  总佣金: {trade_stats['total_commission']:.2f}元")

        # 持仓统计
        position_stats = self.position_manager.get_statistics()
        logger.info(f"\n【持仓统计】")
        logger.info(f"  持仓数量: {position_stats['total_positions']}")
        logger.info(f"  总市值: {position_stats['total_market_value']:.2f}元")
        logger.info(f"  总盈亏: {position_stats['total_pnl']:.2f}元")
        logger.info(f"  盈利持仓: {position_stats['winning_count']}")
        logger.info(f"  亏损持仓: {position_stats['losing_count']}")
        logger.info(f"  胜率: {position_stats['win_rate']:.1%}")

        # 风险指标
        risk_metrics = self.risk_manager.get_risk_metrics()
        logger.info(f"\n【风险指标】")
        logger.info(f"  当前资金: {risk_metrics['current_capital']:.2f}元")
        logger.info(f"  峰值资金: {risk_metrics['peak_capital']:.2f}元")
        logger.info(f"  回撤: {risk_metrics['drawdown_pct']:.2%}")
        logger.info(f"  今日交易: {risk_metrics['daily_trades_count']}次")
        logger.info(f"  风险状态: {risk_metrics['risk_status']}")

        logger.info("\n" + "=" * 60)


def main():
    """主函数"""
    try:
        # 创建交易系统
        system = TradingSystem()

        # 运行系统
        system.run()

    except KeyboardInterrupt:
        logger.info("\n用户中断程序")
    except Exception as e:
        logger.error(f"程序异常退出: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

"""
外汇交易示例 - Forex.com MT5
使用AI多Agent系统进行外汇自动交易

配置：
- 平台：Forex.com
- 杠杆：50倍
- 初始资金：$1000
- 品种：EURUSD, GBPUSD等主要货币对
"""
import sys
from pathlib import Path
from datetime import datetime
from loguru import logger

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils import load_config, setup_logger
from data.mt5_provider import MT5DataProvider, test_mt5_connection
from trading.mt5_executor import MT5Executor
from ai.llm_client import LLMClient, create_llm_client
from ai.llm_enhanced_agents import (
    LLMEnhancedRetailAgent,
    LLMEnhancedInstitutionalAgent,
    LLMEnhancedExpertPanel
)


class ForexTradingSystem:
    """外汇交易系统"""

    def __init__(self, config_path: str = "config_forex.yaml"):
        """初始化外汇交易系统"""
        logger.info("=" * 80)
        logger.info("AI外汇多Agent自动交易系统启动")
        logger.info("Platform: Forex.com MT5")
        logger.info("Leverage: 50x | Initial Capital: $1,000")
        logger.info("=" * 80)

        # 加载配置
        self.config = load_config(config_path)

        # 初始化日志
        log_config = self.config.get('logging', {})
        setup_logger(
            log_level=log_config.get('level', 'INFO'),
            log_file=log_config.get('output.file_path', 'logs/forex_trading.log')
        )

        # 初始化组件
        self._init_mt5()
        self._init_llm()
        self._init_agents()
        self._init_trading()

        logger.info("系统初始化完成！")
        logger.info("=" * 80)

    def _init_mt5(self):
        """初始化MT5连接"""
        logger.info("初始化MT5连接...")

        mt5_config = self.config.get('mt5', {})

        self.data_provider = MT5DataProvider(
            login=mt5_config.get('login'),
            password=mt5_config.get('password'),
            server=mt5_config.get('server'),
            path=mt5_config.get('path')
        )

        if not self.data_provider.connect():
            raise ConnectionError("Failed to connect to MT5")

        # 显示账户信息
        account = self.data_provider.get_account_info()
        if account:
            logger.info(f"✅ MT5连接成功")
            logger.info(f"  账户: {account['login']}")
            logger.info(f"  余额: ${account['balance']:.2f}")
            logger.info(f"  净值: ${account['equity']:.2f}")
            logger.info(f"  杠杆: 1:{account['leverage']}")
            logger.info(f"  服务器: {account['server']}")

    def _init_llm(self):
        """初始化LLM"""
        logger.info("初始化LLM客户端...")

        llm_config = self.config.get('llm', {})
        self.llm_client = create_llm_client(llm_config)

        logger.info(f"✅ LLM已初始化: {llm_config.get('provider')} - {llm_config.get('model')}")

    def _init_agents(self):
        """初始化Agent"""
        logger.info("初始化AI Agent系统...")

        # 创建散户Agent（减少数量，外汇市场快）
        retail_config = self.config.get('agents.retail.types', {})
        self.retail_agents = []

        for agent_type, count in retail_config.items():
            for i in range(count):
                agent_id = f"retail_{agent_type}_{i+1}"
                agent = LLMEnhancedRetailAgent(
                    agent_id=agent_id,
                    agent_type=agent_type,
                    llm_client=self.llm_client
                )
                self.retail_agents.append(agent)

        logger.info(f"✅ 创建了 {len(self.retail_agents)} 个散户Agent")

        # 创建机构Agent
        inst_config = self.config.get('agents.institutional.types', {})
        self.institutional_agents = []

        for agent_type, count in inst_config.items():
            for i in range(count):
                agent_id = f"inst_{agent_type}_{i+1}"
                agent = LLMEnhancedInstitutionalAgent(
                    agent_id=agent_id,
                    agent_type=agent_type,
                    llm_client=self.llm_client
                )
                self.institutional_agents.append(agent)

        logger.info(f"✅ 创建了 {len(self.institutional_agents)} 个机构Agent")

        # 创建专家委员会
        discussion_rounds = self.config.get('agents.expert_panel.discussion_rounds', 2)
        self.expert_panel = LLMEnhancedExpertPanel(
            llm_client=self.llm_client,
            discussion_rounds=discussion_rounds
        )

        logger.info(f"✅ 创建了专家决策委员会（{discussion_rounds}轮讨论）")

    def _init_trading(self):
        """初始化交易执行器"""
        logger.info("初始化交易执行器...")

        leverage = self.config.get('leverage_risk.leverage', 50)

        self.executor = MT5Executor(
            leverage=leverage,
            max_slippage_points=20,
            magic_number=234000
        )

        if not self.executor.connect():
            raise ConnectionError("Failed to connect MT5 executor")

        logger.info(f"✅ 交易执行器已初始化（{leverage}倍杠杆）")

    def scan_forex_market(self):
        """扫描外汇市场"""
        logger.info("\n" + "=" * 80)
        logger.info("扫描外汇市场...")
        logger.info("=" * 80)

        # 获取配置的交易品种
        focus_pairs = self.config.get('forex_trading.symbols.focus_pairs', ['EURUSD'])

        opportunities = []

        for symbol in focus_pairs:
            try:
                logger.info(f"\n分析 {symbol}...")

                # 获取历史数据
                timeframe = self.config.get('forex_trading.timeframes.analysis', 'H1')
                df = self.data_provider.get_historical_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    count=500
                )

                if df is None or df.empty:
                    logger.warning(f"无法获取 {symbol} 数据")
                    continue

                # 计算技术指标
                df = self.data_provider.calculate_technical_indicators(df)

                # 获取最新数据
                latest = df.iloc[-1]
                realtime = self.data_provider.get_realtime_data(symbol)

                # 构建分析数据
                stock_data = {
                    'code': symbol,
                    'name': symbol,
                    'price': realtime['bid'],
                    'spread': realtime['spread'],
                    'change_pct': ((latest['close'] - df.iloc[-20]['close']) / df.iloc[-20]['close']) * 100,
                    'ma5': latest['ma5'],
                    'ma10': latest['ma10'],
                    'ma20': latest['ma20'],
                    'ma50': latest['ma50'],
                    'rsi': latest['rsi'],
                    'macd': latest['macd'],
                    'atr': latest['atr'],
                    'volatility': latest['volatility'],
                    'volume_ratio': 1.0,  # 外汇没有明确成交量
                    'turnover_rate': 1.0,
                    'market_sentiment': 0.5
                }

                logger.info(f"  价格: {stock_data['price']:.5f}")
                logger.info(f"  点差: {realtime['spread']:.5f}")
                logger.info(f"  RSI: {stock_data['rsi']:.2f}")
                logger.info(f"  ATR: {stock_data['atr']:.5f}")

                opportunities.append(stock_data)

            except Exception as e:
                logger.error(f"分析 {symbol} 时出错: {e}")

        return opportunities

    def analyze_with_agents(self, forex_data: dict):
        """使用Agent分析"""
        symbol = forex_data['code']
        logger.info(f"\n{'='*60}")
        logger.info(f"Agent分析: {symbol}")
        logger.info(f"{'='*60}")

        # 散户分析（采样，不是全部）
        retail_sample = self.retail_agents[:10]  # 只用10个散户代表
        retail_decisions = []

        logger.info("\n散户Agent分析...")
        for agent in retail_sample:
            decision = agent.analyze(forex_data)
            retail_decisions.append(decision)
            logger.info(f"  {agent.agent_type}: {decision.action.value} ({decision.confidence:.1%})")

        # 机构分析
        inst_decisions = []

        logger.info("\n机构Agent分析...")
        for agent in self.institutional_agents[:5]:  # 采样5个
            decision = agent.analyze(forex_data)
            inst_decisions.append(decision)
            logger.info(f"  {agent.agent_type}: {decision.action.value} ({decision.confidence:.1%})")

        # 专家委员会决策
        logger.info("\n专家委员会讨论...")
        panel_decision = self.expert_panel.make_decision(
            forex_data,
            retail_decisions,
            inst_decisions
        )

        logger.info(f"\n{'='*60}")
        logger.info(f"最终决策: {panel_decision.final_action.value.upper()}")
        logger.info(f"置信度: {panel_decision.confidence:.1%}")
        logger.info(f"共识度: {panel_decision.consensus_level:.1%}")
        logger.info(f"{'='*60}")

        return panel_decision

    def execute_trade(self, symbol: str, decision):
        """执行交易"""
        from agents.base_agent import ActionType

        if decision.final_action == ActionType.HOLD:
            logger.info("决策为观望，不执行交易")
            return

        # 计算手数
        account = self.data_provider.get_account_info()
        balance = account['balance']

        # 每笔交易风险2%
        risk_amount = balance * 0.02  # $1000 * 2% = $20

        # 止损点数
        sl_pips = self.config.get('stop_loss_take_profit.fixed.stop_loss_pips', 20)
        tp_pips = self.config.get('stop_loss_take_profit.fixed.take_profit_pips', 40)

        # 计算手数
        lot_size = self.executor.calculate_lot_size(
            symbol=symbol,
            risk_amount=risk_amount,
            stop_loss_pips=sl_pips
        )

        logger.info(f"\n准备执行交易:")
        logger.info(f"  品种: {symbol}")
        logger.info(f"  方向: {decision.final_action.value.upper()}")
        logger.info(f"  手数: {lot_size:.2f}")
        logger.info(f"  止损: {sl_pips} pips")
        logger.info(f"  止盈: {tp_pips} pips")
        logger.info(f"  风险: ${risk_amount:.2f}")

        # 发送订单
        if decision.final_action == ActionType.BUY:
            order_type = "buy"
        else:
            order_type = "sell"

        result = self.executor.send_market_order(
            symbol=symbol,
            volume=lot_size,
            order_type=order_type,
            sl_pips=sl_pips,
            tp_pips=tp_pips,
            comment="AI_Trading"
        )

        if result:
            logger.info(f"✅ 订单执行成功! Ticket: {result['ticket']}")
        else:
            logger.error("❌ 订单执行失败!")

    def run(self):
        """运行交易系统"""
        try:
            # 1. 扫描市场
            opportunities = self.scan_forex_market()

            if not opportunities:
                logger.info("未发现交易机会")
                return

            # 2. 分析第一个机会
            forex_data = opportunities[0]
            decision = self.analyze_with_agents(forex_data)

            # 3. 执行交易（如果置信度足够）
            if decision.confidence >= 0.70:
                # 生产环境取消下面的注释来执行真实交易
                # self.execute_trade(forex_data['code'], decision)
                logger.info("\n⚠️  模拟模式：不执行真实交易")
                logger.info("生产环境请取消 execute_trade 的注释")
            else:
                logger.info(f"\n置信度不足 ({decision.confidence:.1%} < 70%)，不交易")

        except Exception as e:
            logger.error(f"系统运行出错: {e}", exc_info=True)

        finally:
            # 清理
            self.data_provider.disconnect()
            self.executor.disconnect()


def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("AI外汇多Agent自动交易系统")
    print("Platform: Forex.com MT5")
    print("=" * 80)

    print("\n⚠️  重要提示:")
    print("1. 确保MT5终端已经启动并登录")
    print("2. 建议先在模拟账户测试")
    print("3. 配置文件: config_forex.yaml")
    print("4. 需要配置ANTHROPIC_API_KEY或OPENAI_API_KEY")

    input("\n按回车继续...")

    try:
        # 先测试MT5连接
        print("\n正在测试MT5连接...")
        if test_mt5_connection():
            print("✅ MT5连接测试成功!\n")
        else:
            print("❌ MT5连接失败，请检查MT5终端是否启动")
            return

        # 创建交易系统
        system = ForexTradingSystem(config_path="config_forex.yaml")

        # 运行
        system.run()

    except KeyboardInterrupt:
        print("\n\n用户中断程序")
    except Exception as e:
        print(f"\n\n程序异常: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

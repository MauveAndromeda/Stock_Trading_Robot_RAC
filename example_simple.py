"""
简单示例：演示系统基本功能
"""
from utils import load_config
from agents import (
    MomentumChaserAgent,
    ValueInvestorAgent,
    ExpertPanel
)
from data import MarketDataProvider, StockAnalyzer

# 简化的示例
def simple_example():
    """简单示例"""
    print("=" * 60)
    print("AI股票多Agent自动交易系统 - 简单示例")
    print("=" * 60)

    # 1. 创建几个Agent
    print("\n1. 创建Agent...")
    retail_agent = MomentumChaserAgent("retail_1")
    inst_agent = ValueInvestorAgent("inst_1")
    expert_panel = ExpertPanel(discussion_rounds=2)

    # 2. 获取股票数据
    print("\n2. 获取股票数据...")
    data_provider = MarketDataProvider(source="akshare")
    analyzer = StockAnalyzer(data_provider)

    # 模拟股票数据（如果无法获取真实数据）
    stock_data = {
        'code': '000001',
        'name': '平安银行',
        'price': 10.50,
        'change_pct': 2.5,
        'volume_ratio': 1.5,
        'turnover_rate': 3.2,
        'ma5': 10.30,
        'ma20': 10.00,
        'ma60': 9.80,
        'rsi': 55,
        'macd': 0.05,
        'pe_ratio': 5.5,
        'pb_ratio': 0.8,
        'roe': 0.12,
        'volatility': 0.025,
        'market_sentiment': 0.6
    }

    print(f"股票: {stock_data['code']} {stock_data['name']}")
    print(f"价格: {stock_data['price']} ({stock_data['change_pct']:+.2f}%)")

    # 3. Agent分析
    print("\n3. Agent分析...")
    print("\n散户Agent意见:")
    retail_decision = retail_agent.analyze(stock_data)
    print(f"  {retail_agent.agent_type}: {retail_decision.action.value}")
    print(f"  理由: {retail_decision.reason}")
    print(f"  置信度: {retail_decision.confidence:.2%}")

    print("\n机构Agent意见:")
    inst_decision = inst_agent.analyze(stock_data)
    print(f"  {inst_agent.agent_type}: {inst_decision.action.value}")
    print(f"  理由: {inst_decision.reason}")
    print(f"  置信度: {inst_decision.confidence:.2%}")

    # 4. 专家委员会决策
    print("\n4. 专家委员会讨论...")
    panel_decision = expert_panel.make_decision(
        stock_data,
        [retail_decision],
        [inst_decision]
    )

    print("\n专家委员会最终决策:")
    print(f"  行动: {panel_decision.final_action.value}")
    print(f"  置信度: {panel_decision.confidence:.2%}")
    print(f"  共识度: {panel_decision.consensus_level:.2%}")
    print("\n" + panel_decision.reasoning)

    print("\n" + "=" * 60)
    print("示例完成！")
    print("=" * 60)


if __name__ == "__main__":
    simple_example()

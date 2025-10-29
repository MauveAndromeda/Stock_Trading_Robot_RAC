"""
MT5连接测试脚本
用于验证MT5是否正确安装和配置
"""
import MetaTrader5 as mt5
from datetime import datetime
import pandas as pd


def test_mt5_basic():
    """测试MT5基本连接"""
    print("=" * 60)
    print("MT5连接测试")
    print("=" * 60)

    # 1. 初始化MT5
    print("\n1. 初始化MT5...")
    if not mt5.initialize():
        print(f"❌ MT5初始化失败: {mt5.last_error()}")
        print("\n请检查:")
        print("  - MT5终端是否已安装")
        print("  - MT5终端是否正在运行")
        print("  - MetaTrader5 Python包是否已安装")
        return False

    print("✅ MT5初始化成功")

    # 2. 获取MT5版本
    print("\n2. MT5版本信息...")
    version = mt5.version()
    print(f"  版本: {version}")

    # 3. 获取账户信息
    print("\n3. 账户信息...")
    account_info = mt5.account_info()
    if account_info is None:
        print("❌ 无法获取账户信息")
        mt5.shutdown()
        return False

    print(f"  账户号: {account_info.login}")
    print(f"  服务器: {account_info.server}")
    print(f"  账户类型: {'模拟' if account_info.trade_mode == 0 else '真实'}")
    print(f"  余额: ${account_info.balance:.2f}")
    print(f"  净值: ${account_info.equity:.2f}")
    print(f"  杠杆: 1:{account_info.leverage}")
    print(f"  货币: {account_info.currency}")
    print(f"  保证金: ${account_info.margin:.2f}")
    print(f"  可用保证金: ${account_info.margin_free:.2f}")

    # 4. 获取可用交易品种
    print("\n4. 可用交易品种...")
    symbols = mt5.symbols_get()
    if symbols is None:
        print("❌ 无法获取交易品种")
    else:
        print(f"  总数: {len(symbols)}")

        # 显示外汇货币对
        forex_symbols = [s for s in symbols if 'USD' in s.name or 'EUR' in s.name or 'GBP' in s.name]
        print(f"  外汇货币对: {len(forex_symbols)}")

        if forex_symbols:
            print("\n  常见货币对:")
            common_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD']
            for pair in common_pairs:
                symbol = mt5.symbol_info(pair)
                if symbol:
                    print(f"    {pair}: ✅ 可用")
                    print(f"      点差: {symbol.spread} 点")
                    print(f"      最小手数: {symbol.volume_min}")
                    print(f"      最大手数: {symbol.volume_max}")
                else:
                    print(f"    {pair}: ❌ 不可用")

    # 5. 获取实时行情
    print("\n5. 实时行情测试...")
    test_symbol = "EURUSD"
    tick = mt5.symbol_info_tick(test_symbol)

    if tick is None:
        print(f"❌ 无法获取 {test_symbol} 行情")
        print(f"   尝试启用品种...")
        if mt5.symbol_select(test_symbol, True):
            tick = mt5.symbol_info_tick(test_symbol)

    if tick:
        print(f"  {test_symbol}:")
        print(f"    买价: {tick.ask:.5f}")
        print(f"    卖价: {tick.bid:.5f}")
        print(f"    点差: {tick.ask - tick.bid:.5f}")
        print(f"    时间: {datetime.fromtimestamp(tick.time)}")
    else:
        print(f"  无法获取 {test_symbol} 行情")

    # 6. 获取历史数据
    print("\n6. 历史数据测试...")
    rates = mt5.copy_rates_from_pos(test_symbol, mt5.TIMEFRAME_H1, 0, 10)

    if rates is None:
        print(f"❌ 无法获取历史数据")
    else:
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        print(f"  获取了 {len(df)} 根K线")
        print("\n  最新5根K线:")
        print(df[['time', 'open', 'high', 'low', 'close', 'tick_volume']].tail())

    # 7. 获取持仓
    print("\n7. 当前持仓...")
    positions = mt5.positions_get()
    if positions is None:
        print("  无持仓")
    else:
        print(f"  持仓数: {len(positions)}")
        for pos in positions:
            print(f"    {pos.symbol} {pos.type} {pos.volume} 手")
            print(f"      盈亏: ${pos.profit:.2f}")

    # 8. 断开连接
    print("\n8. 断开连接...")
    mt5.shutdown()
    print("✅ 已断开")

    print("\n" + "=" * 60)
    print("✅ MT5连接测试完成!")
    print("=" * 60)

    return True


def quick_test():
    """快速测试"""
    print("\n快速测试MT5连接...\n")

    if not mt5.initialize():
        print("❌ MT5连接失败")
        print(f"错误: {mt5.last_error()}")
        print("\n请检查:")
        print("1. MT5终端是否已安装")
        print("2. MT5终端是否正在运行")
        print("3. 是否已安装: pip install MetaTrader5")
        return False

    account = mt5.account_info()
    if account:
        print("✅ MT5连接成功!")
        print(f"\n账户: {account.login}")
        print(f"余额: ${account.balance:.2f}")
        print(f"杠杆: 1:{account.leverage}")
        print(f"服务器: {account.server}")

    mt5.shutdown()
    return True


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        quick_test()
    else:
        test_mt5_basic()

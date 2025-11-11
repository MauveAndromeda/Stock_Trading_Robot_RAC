#!/usr/bin/env python3
"""一键回测脚本
==================
直接运行本脚本即可在纯离线环境下复现仓库默认的AI多Agent完整回测。
"""
from __future__ import annotations

import sys
from pathlib import Path

import random


def _prepare_environment() -> Path:
    """将仓库根目录加入Python路径并返回根目录路径。"""
    repo_root = Path(__file__).resolve().parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


def run_backtest():
    """加载完整系统并执行默认区间的回测。"""
    _prepare_environment()

    from ai_trading_complete_system import (
        AITradingSystem,
        BACKTEST_START_DATE,
        BACKTEST_END_DATE,
    )

    system = AITradingSystem()
    metrics = system.run_backtest(BACKTEST_START_DATE, BACKTEST_END_DATE)

    print("\n✅ 一键回测执行完成！")
    print("最终资金: ${:,.2f}".format(metrics.get("final_value", 0.0)))
    print("总收益率: {:+.2%}".format(metrics.get("total_return", 0.0)))
    print("最大回撤: {:+.2%}".format(metrics.get("max_drawdown", 0.0)))
    print("夏普比率: {:.2f}".format(metrics.get("sharpe_ratio", 0.0)))

    return metrics


if __name__ == "__main__":
    # 固定随机种子，确保下载即用也能复现完全相同的结果
    random.seed(42)
    run_backtest()

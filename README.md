# 🤖 AI Multi-Agent Stock Trading System | AI多Agent股票自动交易系统

[English](#english) | [中文](#chinese)

---

<a name="english"></a>
## 🇬🇧 English Version

### Overview

An advanced AI-powered stock trading system featuring **250+ intelligent agents** (200 retail + 50 institutional) powered by **GPT-5** and **Claude 4.5 Sonnet**, plus a **12-expert consensus panel** conducting multi-round deliberations to achieve **30%+ annual returns**.

### 🌟 Key Features

- **🤖 Dual AI Engine**: GPT-5 + Claude 4.5 Sonnet working together
- **👥 250+ Independent Agents**: Each makes autonomous decisions
  - 200 Retail Agents (5 types: momentum chasers, panic sellers, herd followers, value seekers, technical traders)
  - 50 Institutional Agents (5 types: quantitative, statistical arbitrage, machine learning, high frequency, market makers)
- **🎓 12 Elite Experts**: Multi-round consensus deliberation
- **📊 Focused Assets**: Index ETFs, Gold, Precious Metals, Blue Chips
- **🏛️ Institutional Strategy**: Follow smart money, counter retail sentiment
- **🎯 Target**: 30%+ annual returns with controlled risk

### 🚀 Quick Start

#### 1. Install Dependencies
```bash
pip install numpy pandas anthropic openai yfinance
```

#### 2. Set API Keys
```bash
# GPT-5 (OpenAI)
export OPENAI_API_KEY="sk-xxxxx"

# Claude 4.5 Sonnet (Anthropic)
export ANTHROPIC_API_KEY="sk-ant-xxxxx"
```

#### 3. Run System
```bash
# Full system with agents + experts
python ultimate_trading_system.py

# Expert-only version
python expert_consensus_system.py

# Elite version (institutional-grade)
python elite_trading_system.py
```

### 📂 System Versions

| File | Description | Target | Best For |
|------|-------------|--------|----------|
| `ultimate_trading_system.py` | **250+ Agents + 12 Experts** | 30%+ | Maximum intelligence |
| `expert_consensus_system.py` | 12 Experts only | 30%+ | Focused decision-making |
| `elite_trading_system.py` | Full position, multi-factor | 15-30% | Institutional-grade |
| `ai_trading_complete_system.py` | Original complete system | 15-25% | Learning & research |
| `forex_trading_example.py` | Forex with MT5 | High risk | Forex traders |

### 🎯 System Architecture

```
Market Data (yfinance)
    ↓
┌─────────────────────────────────────────────┐
│  250+ Independent Agents                    │
│  ┌──────────────┐  ┌──────────────────┐   │
│  │ 200 Retail   │  │ 50 Institutional │   │
│  │ - Momentum   │  │ - Quant          │   │
│  │ - Panic      │  │ - Stat Arb       │   │
│  │ - Herd       │  │ - ML             │   │
│  │ - Value      │  │ - HFT            │   │
│  │ - Technical  │  │ - Market Maker   │   │
│  └──────────────┘  └──────────────────┘   │
│         ↓                    ↓              │
│    Each agent makes independent decision   │
│    Powered by GPT-5 or Claude 4.5         │
└─────────────────────────────────────────────┘
                  ↓
        ┌─────────────────────┐
        │  12 Expert Panel    │
        │  6-Round Discussion │
        └─────────────────────┘
                  ↓
         Final Consensus Decision
                  ↓
         Trade Execution (100% position)
```

### 🏛️ Institutional Strategy

**"Follow Institutions, Counter Retail"**

- **When retail panics** (extreme fear) + institutions buy → **BUY**
- **When retail euphoric** (extreme greed) + institutions sell → **SELL**
- Track institutional flow via market microstructure analysis
- Exploit retail behavioral biases

### 📊 Asset Universe (16 Focused Assets)

**Index ETFs**: SPY, QQQ, DIA, IWM
**Sector ETFs**: XLF, XLK, XLE
**Gold & Precious Metals**: GLD, GDX, GDXJ, SLV
**Blue Chips**: AAPL, MSFT, NVDA, TSLA, AMZN

### 🛡️ Risk Management

- **Stop Loss**: 4%
- **Take Profit**: 12%
- **Trailing Stop**: 3%
- **Max Drawdown**: -15%
- **Target Sharpe**: > 2.0

### 📖 Documentation

- **English**:
  - `EXPERT_CONSENSUS_GUIDE.md` - Expert system guide
  - `FINAL_SUMMARY.md` - Project summary
  - `FOREX_TRADING_GUIDE.md` - Forex trading guide

- **Chinese**:
  - `快速开始.md` - Quick start guide
  - `一键回测使用指南.md` - Backtest guide
  - `精英系统使用指南.md` - Elite system guide
  - `系统版本对比.md` - Version comparison

### 🎓 Performance Targets

| Metric | Target | Description |
|--------|--------|-------------|
| Annual Return | 30%+ | Aggressive returns |
| Sharpe Ratio | > 2.0 | Risk-adjusted performance |
| Max Drawdown | < -15% | Capital preservation |
| Win Rate | > 65% | Trading accuracy |

### 🔬 Technology Stack

- **AI Models**: GPT-5, Claude 4.5 Sonnet
- **Data**: yfinance (real US stock data)
- **Analytics**: NumPy, Pandas
- **Factors**: Multi-factor quantitative models
- **Risk**: VaR, CVaR, Kelly Criterion

### 📜 License

MIT License - See LICENSE file

### ⚠️ Disclaimer

This system is for educational and research purposes only. Trading carries significant risk. Past performance does not guarantee future results. Never trade with money you cannot afford to lose.

---

<a name="chinese"></a>
## 🇨🇳 中文版本

### 系统概述

先进的AI驱动股票交易系统，配备**250+智能Agent**（200散户 + 50机构），由**GPT-5**和**Claude 4.5 Sonnet**双引擎驱动，加上**12位顶级专家**进行多轮讨论，目标实现**年化30%+收益**。

### 🌟 核心特性

- **🤖 双AI引擎**: GPT-5 + Claude 4.5 Sonnet协同工作
- **👥 250+独立Agent**: 每个都自主决策
  - 200个散户Agent（5类：追涨杀跌、恐慌型、跟风型、价值型、技术型）
  - 50个机构Agent（5类：量化、统计套利、机器学习、高频、做市商）
- **🎓 12位精英专家**: 多轮共识讨论
- **📊 专注资产**: 指数ETF、黄金、贵金属、蓝筹股
- **🏛️ 机构策略**: 顺机构、反散户
- **🎯 目标**: 控制风险下实现30%+年化收益

### 🚀 快速开始

#### 1. 安装依赖
```bash
pip install numpy pandas anthropic openai yfinance
```

#### 2. 设置API密钥
```bash
# GPT-5 (OpenAI)
export OPENAI_API_KEY="sk-xxxxx"

# Claude 4.5 Sonnet (Anthropic)
export ANTHROPIC_API_KEY="sk-ant-xxxxx"
```

#### 3. 运行系统
```bash
# 完整系统（agents + 专家）
python ultimate_trading_system.py

# 纯专家版本
python expert_consensus_system.py

# 精英版（机构级别）
python elite_trading_system.py
```

### 📂 系统版本

| 文件 | 说明 | 目标收益 | 适合人群 |
|------|------|---------|----------|
| `ultimate_trading_system.py` | **250+ Agent + 12专家** | 30%+ | 追求最大智能 |
| `expert_consensus_system.py` | 纯12专家系统 | 30%+ | 专注决策质量 |
| `elite_trading_system.py` | 全仓轮动、多因子 | 15-30% | 机构级别 |
| `ai_trading_complete_system.py` | 原始完整系统 | 15-25% | 学习研究 |
| `forex_trading_example.py` | 外汇MT5交易 | 高风险 | 外汇交易者 |

### 🎯 系统架构

```
市场数据 (yfinance)
    ↓
┌─────────────────────────────────────────────┐
│  250+独立Agent                              │
│  ┌──────────────┐  ┌──────────────────┐   │
│  │ 200散户Agent │  │ 50机构Agent      │   │
│  │ - 追涨杀跌   │  │ - 量化           │   │
│  │ - 恐慌型     │  │ - 统计套利       │   │
│  │ - 跟风型     │  │ - 机器学习       │   │
│  │ - 价值型     │  │ - 高频交易       │   │
│  │ - 技术型     │  │ - 做市商         │   │
│  └──────────────┘  └──────────────────┘   │
│         ↓                    ↓              │
│    每个Agent独立决策                        │
│    由GPT-5或Claude 4.5驱动                 │
└─────────────────────────────────────────────┘
                  ↓
        ┌─────────────────────┐
        │  12专家委员会        │
        │  6轮深度讨论         │
        └─────────────────────┘
                  ↓
         最终共识决策
                  ↓
         交易执行（全仓）
```

### 🏛️ 机构策略

**"顺机构，反散户"**

- **散户恐慌**（极度恐惧）+ 机构买入 → **买入**
- **散户狂热**（极度贪婪）+ 机构卖出 → **卖出**
- 通过市场微观结构分析追踪机构流向
- 利用散户行为偏差

### 📊 资产池（16个专注资产）

**指数ETF**: SPY, QQQ, DIA, IWM
**行业ETF**: XLF, XLK, XLE
**黄金及贵金属**: GLD, GDX, GDXJ, SLV
**蓝筹股**: AAPL, MSFT, NVDA, TSLA, AMZN

### 🛡️ 风险管理

- **止损**: 4%
- **止盈**: 12%
- **移动止损**: 3%
- **最大回撤**: -15%
- **目标夏普比率**: > 2.0

### 📖 文档

- **英文文档**:
  - `EXPERT_CONSENSUS_GUIDE.md` - 专家系统指南
  - `FINAL_SUMMARY.md` - 项目总结
  - `FOREX_TRADING_GUIDE.md` - 外汇交易指南

- **中文文档**:
  - `快速开始.md` - 快速入门
  - `一键回测使用指南.md` - 回测使用指南
  - `精英系统使用指南.md` - 精英系统指南
  - `系统版本对比.md` - 版本对比

### 🎓 性能目标

| 指标 | 目标 | 说明 |
|------|------|------|
| 年化收益 | 30%+ | 激进收益 |
| 夏普比率 | > 2.0 | 风险调整后收益 |
| 最大回撤 | < -15% | 资本保护 |
| 胜率 | > 65% | 交易准确率 |

### 🔬 技术栈

- **AI模型**: GPT-5, Claude 4.5 Sonnet
- **数据**: yfinance（真实美股数据）
- **分析**: NumPy, Pandas
- **因子**: 多因子量化模型
- **风险**: VaR, CVaR, Kelly公式

### 📜 许可证

MIT License - 查看LICENSE文件

### ⚠️ 免责声明

本系统仅供教育和研究用途。交易存在重大风险。历史业绩不代表未来表现。切勿用无法承受损失的资金进行交易。

---

## 🤝 Contributing | 贡献

Contributions are welcome! Please feel free to submit a Pull Request.

欢迎贡献！请随时提交Pull Request。

## 📧 Contact | 联系

For questions or support, please open an issue on GitHub.

如有问题或需要支持，请在GitHub上开issue。

---

**Made with ❤️ by AI Trading Team**

**由AI交易团队用❤️制作**

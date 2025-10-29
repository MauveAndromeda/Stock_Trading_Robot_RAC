# 使用指南

## 快速开始

### 1. 环境准备

```bash
# 克隆项目
cd Stock_Trading_Robot_RAC

# 安装依赖
pip install -r requirements.txt

# 复制环境变量配置
cp .env.example .env

# 编辑.env文件，填入你的API密钥
nano .env
```

### 2. 配置系统

编辑 `config.yaml`，根据需要调整配置：

```yaml
# 关键配置项
llm:
  api_key: "${OPENAI_API_KEY}"  # LLM API密钥

agents:
  retail:
    total_count: 200  # 散户Agent数量
  institutional:
    total_count: 50   # 机构Agent数量

trading:
  mode: "simulation"  # simulation（模拟）或 live（实盘）

risk_management:
  stop_loss:
    percentage: 0.05  # 止损比例5%
  stop_profit:
    percentage: 0.15  # 止盈比例15%
```

### 3. 运行方式

#### 方式一：运行一次

```bash
# 运行主程序（执行一次完整的市场扫描和交易）
python main.py
```

#### 方式二：定时运行

```bash
# 启动调度器（按配置的时间表自动运行）
python scheduler.py
```

#### 方式三：运行简单示例

```bash
# 运行简化示例，快速了解系统
python example_simple.py
```

## 系统工作流程

### 完整流程

```
1. 市场扫描
   ↓
2. 筛选潜在机会（基于基本面和技术面）
   ↓
3. 200+散户Agent分析 + 50+机构Agent分析
   ↓
4. 专家委员会3轮讨论
   ↓
5. 生成交易策略
   ↓
6. 风险检查
   ↓
7. 执行交易
   ↓
8. 生成报告
```

### 核心策略："顺机构，反散户"

- **当散户恐慌抛售 + 机构逆向买入** → 系统买入
- **当散户狂热追涨 + 机构获利了结** → 系统卖出
- **情绪分歧度高** → 机会更好

## Agent类型说明

### 散户Agent（5种类型，共200个）

1. **追涨杀跌型** (50个)
   - 看到涨就追，看到跌就杀
   - 情绪化交易
   - 常常买在高点，卖在低点

2. **恐慌型** (40个)
   - 极度厌恶风险
   - 市场稍有风吹草动就卖出
   - 容易被负面消息影响

3. **跟风型** (40个)
   - 听消息，跟大众
   - 看别人买就跟着买
   - 缺乏独立判断

4. **价值型散户** (40个)
   - 关注基本面但不专业
   - 喜欢"便宜"的股票
   - 容易被简单指标误导

5. **技术型散户** (30个)
   - 依赖技术指标
   - 但只会用简单指标
   - 容易被假突破欺骗

### 机构Agent（5种类型，共50个）

1. **量化对冲基金** (15个)
   - 多因子模型
   - 数据驱动决策
   - 情绪中性

2. **价值投资机构** (10个)
   - 深度基本面研究
   - 注重安全边际
   - 长期持有

3. **趋势跟踪机构** (10个)
   - 多周期技术分析
   - 顺势而为
   - 系统化交易

4. **高频交易机构** (10个)
   - 短期价格波动
   - 依赖流动性
   - 快速进出

5. **指数基金** (5个)
   - 被动投资
   - 跟踪指数
   - 定期再平衡

### 专家委员会（4位专家）

1. **情绪分析专家**
   - 分析散户群体情绪
   - 识别贪婪/恐慌信号

2. **机构行为专家**
   - 解读机构资金流向
   - 判断真实意图

3. **风险控制专家**
   - 评估交易风险
   - 控制回撤

4. **市场时机专家**
   - 判断入场时机
   - 识别买卖点

## 风险管理

系统内置多重风险控制：

### 仓位控制
- 最大总仓位：80%
- 单只股票最大仓位：10%
- 最小现金储备：20%

### 止损止盈
- 止损：-5%
- 止盈：+15%
- 支持跟踪止损

### 交易限制
- 单日最大交易次数：20次
- 单日最大亏损：3%
- 最大回撤限制：10%

### 异常处理
- 错误时自动停止
- 大额亏损实时通知
- 异常情况自动暂停交易

## 实盘交易配置

⚠️ **重要提示**：实盘交易有风险，请确保充分测试！

### 切换到实盘模式

1. 修改 `config.yaml`:

```yaml
trading:
  mode: "live"  # 从simulation改为live

  broker:
    name: "your_broker"
    account_id: "${BROKER_ACCOUNT}"
    api_key: "${BROKER_API_KEY}"
    api_secret: "${BROKER_API_SECRET}"
```

2. 在 `.env` 文件中配置券商信息:

```bash
BROKER_ACCOUNT=your_account
BROKER_API_KEY=your_api_key
BROKER_API_SECRET=your_api_secret
```

3. 实现券商API接口（需要根据你的券商文档）：

编辑 `trading/executor.py`，实现 `_submit_to_broker()` 方法。

## 调度配置

在 `config.yaml` 中配置定时任务：

```yaml
scheduler:
  enabled: true
  jobs:
    - name: "market_scan"
      cron: "*/5 9-15 * * 1-5"  # 交易日9-15点，每5分钟
      function: "scan_market_opportunities"

    - name: "daily_report"
      cron: "0 16 * * 1-5"  # 每个交易日16点
      function: "generate_daily_report"
```

Cron表达式格式：`分 时 日 月 周`

## 日志和监控

### 查看日志

```bash
# 实时查看日志
tail -f logs/trading_system.log

# 查看调度器日志
tail -f logs/scheduler.log
```

### 日志级别

在 `config.yaml` 中调整：

```yaml
logging:
  level: "INFO"  # DEBUG, INFO, WARNING, ERROR
```

## 常见问题

### Q: 如何调整Agent数量？

A: 修改 `config.yaml` 中的配置：

```yaml
agents:
  retail:
    types:
      momentum_chaser: 100  # 调整数量
      panic_seller: 80
      ...
```

### Q: 如何修改止损止盈比例？

A: 修改 `config.yaml`:

```yaml
risk_management:
  stop_loss:
    percentage: 0.03  # 改为3%
  stop_profit:
    percentage: 0.20  # 改为20%
```

### Q: 系统每次会分析多少只股票？

A: 由 `config.yaml` 控制：

```yaml
market_data:
  stock_universe:
    max_stocks: 500  # 最多监控500只
```

主程序中会进一步筛选前20只进行详细分析。

### Q: 如何禁用某类Agent？

A: 将数量设为0：

```yaml
agents:
  retail:
    types:
      momentum_chaser: 0  # 禁用追涨杀跌型
```

### Q: 模拟交易的初始资金是多少？

A: 默认100万，可以在代码中修改：

编辑 `main.py`，找到 `_init_trading_modules()` 方法：

```python
self.risk_manager = RiskManager(
    initial_capital=1000000,  # 修改这里
    ...
)
```

## 扩展开发

### 添加新的Agent类型

1. 继承 `BaseAgent` 类
2. 实现 `analyze()` 和 `get_personality_prompt()` 方法
3. 在工厂类中注册新类型

### 添加新的数据源

1. 编辑 `data/market_data.py`
2. 在 `MarketDataProvider` 中添加新数据源的实现

### 自定义策略

1. 编辑 `strategy/strategy_generator.py`
2. 修改 `_should_execute()` 方法中的逻辑

## 性能优化建议

1. **减少Agent数量**：开发测试时可以减少到10个散户 + 5个机构
2. **限制分析股票数量**：设置较小的 `max_stocks`
3. **启用缓存**：确保 `cache.enabled: true`
4. **使用本地数据**：避免频繁请求API

## 安全建议

1. ⚠️ 先在模拟模式充分测试
2. 🔐 妥善保管API密钥和券商账号
3. 💰 实盘时从小资金开始
4. 📊 定期检查交易记录
5. 🚨 设置合理的止损止盈
6. 📱 配置异常通知

## 技术支持

如遇问题，请检查：
1. 日志文件 `logs/trading_system.log`
2. 配置文件是否正确
3. API密钥是否有效
4. 网络连接是否正常

## 免责声明

本系统仅供学习研究使用，不构成任何投资建议。股市有风险，投资需谨慎。使用本系统进行实盘交易的一切后果由使用者自行承担。

"""
任务调度器
定时运行交易系统
"""
import sys
from pathlib import Path
from datetime import datetime
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from loguru import logger

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils import setup_logger, load_config
from main import TradingSystem


class TradingScheduler:
    """交易调度器"""

    def __init__(self, config_path: str = "config.yaml"):
        """
        初始化调度器

        Args:
            config_path: 配置文件路径
        """
        self.config = load_config(config_path)

        # 初始化日志
        log_config = self.config.get('logging', {})
        setup_logger(
            log_level=log_config.get('level', 'INFO'),
            log_file="logs/scheduler.log",
            rotation=log_config.get('rotation', '1 day'),
            retention=log_config.get('retention', '30 days')
        )

        # 创建调度器
        self.scheduler = BlockingScheduler()

        # 添加任务
        self._add_jobs()

        logger.info("TradingScheduler initialized")

    def _add_jobs(self):
        """添加调度任务"""
        scheduler_config = self.config.get('scheduler', {})

        if not scheduler_config.get('enabled', False):
            logger.warning("Scheduler is disabled in config")
            return

        jobs = scheduler_config.get('jobs', [])

        for job in jobs:
            job_name = job.get('name')
            cron_expr = job.get('cron')
            function_name = job.get('function')

            if not all([job_name, cron_expr, function_name]):
                logger.warning(f"Invalid job config: {job}")
                continue

            # 解析cron表达式
            # 格式: "分 时 日 月 周"
            # 例: "*/5 9-15 * * 1-5" = 周一到周五，9点到15点，每5分钟
            cron_parts = cron_expr.split()
            if len(cron_parts) != 5:
                logger.warning(f"Invalid cron expression: {cron_expr}")
                continue

            minute, hour, day, month, day_of_week = cron_parts

            # 添加任务
            if function_name == "scan_market_opportunities":
                self.scheduler.add_job(
                    self.scan_market_opportunities,
                    CronTrigger(
                        minute=minute,
                        hour=hour,
                        day=day,
                        month=month,
                        day_of_week=day_of_week
                    ),
                    id=job_name,
                    name=job_name
                )
                logger.info(f"Added job: {job_name} ({cron_expr})")

            elif function_name == "daily_market_analysis":
                self.scheduler.add_job(
                    self.daily_market_analysis,
                    CronTrigger(
                        minute=minute,
                        hour=hour,
                        day=day,
                        month=month,
                        day_of_week=day_of_week
                    ),
                    id=job_name,
                    name=job_name
                )
                logger.info(f"Added job: {job_name} ({cron_expr})")

            elif function_name == "check_positions":
                self.scheduler.add_job(
                    self.check_positions,
                    CronTrigger(
                        minute=minute,
                        hour=hour,
                        day=day,
                        month=month,
                        day_of_week=day_of_week
                    ),
                    id=job_name,
                    name=job_name
                )
                logger.info(f"Added job: {job_name} ({cron_expr})")

            elif function_name == "generate_daily_report":
                self.scheduler.add_job(
                    self.generate_daily_report,
                    CronTrigger(
                        minute=minute,
                        hour=hour,
                        day=day,
                        month=month,
                        day_of_week=day_of_week
                    ),
                    id=job_name,
                    name=job_name
                )
                logger.info(f"Added job: {job_name} ({cron_expr})")

    def scan_market_opportunities(self):
        """扫描市场机会并执行交易"""
        logger.info("\n" + "=" * 80)
        logger.info(f"[{datetime.now()}] 开始扫描市场机会...")
        logger.info("=" * 80)

        try:
            system = TradingSystem()
            system.run()
        except Exception as e:
            logger.error(f"扫描市场机会时出错: {e}", exc_info=True)

        logger.info("市场扫描完成")
        logger.info("=" * 80 + "\n")

    def daily_market_analysis(self):
        """每日市场分析"""
        logger.info("\n" + "=" * 80)
        logger.info(f"[{datetime.now()}] 开始每日市场分析...")
        logger.info("=" * 80)

        try:
            from data import MarketDataProvider, StockAnalyzer

            data_provider = MarketDataProvider()
            analyzer = StockAnalyzer(data_provider)

            # 获取股票列表
            stocks = data_provider.get_stock_list(max_stocks=100)

            logger.info(f"今日分析 {len(stocks)} 只股票")

            # 分析市场整体情况
            total_rise = sum(1 for s in stocks if s['change_pct'] > 0)
            total_fall = sum(1 for s in stocks if s['change_pct'] < 0)
            avg_change = sum(s['change_pct'] for s in stocks) / len(stocks)

            logger.info(f"\n市场概况:")
            logger.info(f"  上涨: {total_rise} ({total_rise/len(stocks):.1%})")
            logger.info(f"  下跌: {total_fall} ({total_fall/len(stocks):.1%})")
            logger.info(f"  平均涨跌幅: {avg_change:.2f}%")

        except Exception as e:
            logger.error(f"每日市场分析时出错: {e}", exc_info=True)

        logger.info("每日市场分析完成")
        logger.info("=" * 80 + "\n")

    def check_positions(self):
        """检查持仓"""
        logger.info("\n" + "=" * 80)
        logger.info(f"[{datetime.now()}] 开始检查持仓...")
        logger.info("=" * 80)

        try:
            # 这里需要持久化持仓数据
            # 简化实现
            logger.info("持仓检查功能待完善")

        except Exception as e:
            logger.error(f"检查持仓时出错: {e}", exc_info=True)

        logger.info("持仓检查完成")
        logger.info("=" * 80 + "\n")

    def generate_daily_report(self):
        """生成每日报告"""
        logger.info("\n" + "=" * 80)
        logger.info(f"[{datetime.now()}] 开始生成每日报告...")
        logger.info("=" * 80)

        try:
            # 生成报告
            report_date = datetime.now().strftime("%Y-%m-%d")
            logger.info(f"\n【{report_date} 交易日报】")
            logger.info("报告生成功能待完善")

        except Exception as e:
            logger.error(f"生成每日报告时出错: {e}", exc_info=True)

        logger.info("每日报告生成完成")
        logger.info("=" * 80 + "\n")

    def start(self):
        """启动调度器"""
        logger.info("\n" + "=" * 80)
        logger.info("交易调度器启动")
        logger.info("=" * 80)

        # 显示已添加的任务
        jobs = self.scheduler.get_jobs()
        logger.info(f"\n已添加 {len(jobs)} 个定时任务:")
        for job in jobs:
            logger.info(f"  - {job.name}: {job.trigger}")

        logger.info("\n调度器运行中... (按 Ctrl+C 停止)")
        logger.info("=" * 80 + "\n")

        try:
            self.scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            logger.info("\n调度器已停止")


def main():
    """主函数"""
    try:
        scheduler = TradingScheduler()
        scheduler.start()
    except Exception as e:
        logger.error(f"调度器异常退出: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

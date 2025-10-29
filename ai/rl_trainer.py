"""
强化学习训练器
使用Stable-Baselines3训练交易策略
"""
import os
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
from loguru import logger

try:
    from stable_baselines3 import PPO, SAC, TD3
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
    from stable_baselines3.common.monitor import Monitor
    SB3_AVAILABLE = True
except ImportError:
    logger.warning("stable-baselines3 not installed. RL training unavailable.")
    SB3_AVAILABLE = False

from .rl_trading_env import StockTradingEnv


class RLTradingTrainer:
    """强化学习交易策略训练器"""

    def __init__(
        self,
        algorithm: str = "PPO",
        env_kwargs: Optional[Dict[str, Any]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        save_dir: str = "models/rl"
    ):
        """
        初始化训练器

        Args:
            algorithm: RL算法 (PPO, SAC, TD3)
            env_kwargs: 环境参数
            model_kwargs: 模型参数
            save_dir: 模型保存目录
        """
        if not SB3_AVAILABLE:
            raise ImportError("Please install stable-baselines3: pip install stable-baselines3")

        self.algorithm = algorithm
        self.env_kwargs = env_kwargs or {}
        self.model_kwargs = model_kwargs or {}
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.model = None
        self.env = None

        logger.info(f"RLTradingTrainer initialized with {algorithm}")

    def prepare_data(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.8
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        准备训练和测试数据

        Args:
            df: 完整数据
            train_ratio: 训练集比例

        Returns:
            (train_df, test_df)
        """
        split_idx = int(len(df) * train_ratio)
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()

        logger.info(f"Data split: train={len(train_df)}, test={len(test_df)}")

        return train_df, test_df

    def create_env(
        self,
        df: pd.DataFrame,
        is_eval: bool = False
    ) -> StockTradingEnv:
        """
        创建交易环境

        Args:
            df: 数据
            is_eval: 是否为评估环境

        Returns:
            StockTradingEnv
        """
        env = StockTradingEnv(df=df, **self.env_kwargs)

        # 使用Monitor包装以记录统计信息
        log_dir = self.save_dir / ("eval_logs" if is_eval else "train_logs")
        log_dir.mkdir(exist_ok=True)
        env = Monitor(env, str(log_dir))

        return env

    def train(
        self,
        train_df: pd.DataFrame,
        eval_df: Optional[pd.DataFrame] = None,
        total_timesteps: int = 100000,
        eval_freq: int = 10000,
        save_freq: int = 10000
    ):
        """
        训练模型

        Args:
            train_df: 训练数据
            eval_df: 评估数据
            total_timesteps: 总训练步数
            eval_freq: 评估频率
            save_freq: 保存频率
        """
        logger.info(f"Starting training for {total_timesteps} timesteps...")

        # 创建环境
        train_env = self.create_env(train_df, is_eval=False)
        train_env = DummyVecEnv([lambda: train_env])

        # 创建评估环境
        if eval_df is not None:
            eval_env = self.create_env(eval_df, is_eval=True)
            eval_env = DummyVecEnv([lambda: eval_env])
        else:
            eval_env = None

        # 创建模型
        model_class = self._get_model_class()

        # 默认模型参数
        default_kwargs = {
            'policy': 'MlpPolicy',
            'env': train_env,
            'verbose': 1,
            'tensorboard_log': str(self.save_dir / 'tensorboard'),
        }

        # 算法特定参数
        if self.algorithm == "PPO":
            default_kwargs.update({
                'learning_rate': 3e-4,
                'n_steps': 2048,
                'batch_size': 64,
                'n_epochs': 10,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_range': 0.2,
                'ent_coef': 0.01,
            })
        elif self.algorithm in ["SAC", "TD3"]:
            default_kwargs.update({
                'learning_rate': 3e-4,
                'buffer_size': 100000,
                'batch_size': 256,
                'tau': 0.005,
                'gamma': 0.99,
            })

        # 合并用户参数
        default_kwargs.update(self.model_kwargs)

        self.model = model_class(**default_kwargs)

        # 创建回调
        callbacks = []

        # 评估回调
        if eval_env is not None:
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=str(self.save_dir / 'best_model'),
                log_path=str(self.save_dir / 'eval_logs'),
                eval_freq=eval_freq,
                deterministic=True,
                render=False
            )
            callbacks.append(eval_callback)

        # 保存检查点
        checkpoint_callback = CheckpointCallback(
            save_freq=save_freq,
            save_path=str(self.save_dir / 'checkpoints'),
            name_prefix=f'{self.algorithm}_trading'
        )
        callbacks.append(checkpoint_callback)

        # 开始训练
        try:
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=callbacks,
                progress_bar=True
            )

            # 保存最终模型
            final_model_path = self.save_dir / f'{self.algorithm}_final.zip'
            self.model.save(str(final_model_path))
            logger.info(f"Training complete! Model saved to {final_model_path}")

        except KeyboardInterrupt:
            logger.warning("Training interrupted by user")
            # 保存中断时的模型
            interrupt_path = self.save_dir / f'{self.algorithm}_interrupted.zip'
            self.model.save(str(interrupt_path))
            logger.info(f"Interrupted model saved to {interrupt_path}")

    def evaluate(
        self,
        test_df: pd.DataFrame,
        model_path: Optional[str] = None,
        n_eval_episodes: int = 10
    ) -> Dict[str, float]:
        """
        评估模型

        Args:
            test_df: 测试数据
            model_path: 模型路径（如果不提供，使用当前模型）
            n_eval_episodes: 评估轮次

        Returns:
            评估指标
        """
        logger.info(f"Evaluating model on {len(test_df)} timesteps...")

        # 加载模型
        if model_path:
            model_class = self._get_model_class()
            model = model_class.load(model_path)
        else:
            model = self.model

        if model is None:
            raise ValueError("No model available. Train or load a model first.")

        # 创建测试环境
        test_env = self.create_env(test_df, is_eval=True)

        # 运行评估
        episode_rewards = []
        episode_returns = []

        for episode in range(n_eval_episodes):
            obs, info = test_env.reset()
            done = False
            episode_reward = 0

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = test_env.step(action)
                done = terminated or truncated
                episode_reward += reward

            episode_rewards.append(episode_reward)
            episode_returns.append(info['total_return'])

            logger.info(
                f"Episode {episode + 1}/{n_eval_episodes}: "
                f"Reward={episode_reward:.2f}, "
                f"Return={info['total_return']:.2%}"
            )

        # 计算统计指标
        metrics = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_return': np.mean(episode_returns),
            'std_return': np.std(episode_returns),
            'max_return': np.max(episode_returns),
            'min_return': np.min(episode_returns),
        }

        logger.info("\n" + "=" * 60)
        logger.info("Evaluation Results:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value:.4f}")
        logger.info("=" * 60)

        return metrics

    def load_model(self, model_path: str):
        """加载模型"""
        model_class = self._get_model_class()
        self.model = model_class.load(model_path)
        logger.info(f"Model loaded from {model_path}")

    def _get_model_class(self):
        """获取模型类"""
        algorithms = {
            'PPO': PPO,
            'SAC': SAC,
            'TD3': TD3
        }

        if self.algorithm not in algorithms:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        return algorithms[self.algorithm]


# 便捷函数
def train_rl_strategy(
    df: pd.DataFrame,
    algorithm: str = "PPO",
    total_timesteps: int = 100000,
    save_dir: str = "models/rl"
) -> RLTradingTrainer:
    """
    快速训练RL策略

    Args:
        df: 历史数据
        algorithm: 算法
        total_timesteps: 训练步数
        save_dir: 保存目录

    Returns:
        训练器实例
    """
    trainer = RLTradingTrainer(
        algorithm=algorithm,
        save_dir=save_dir
    )

    # 准备数据
    train_df, test_df = trainer.prepare_data(df)

    # 训练
    trainer.train(
        train_df=train_df,
        eval_df=test_df,
        total_timesteps=total_timesteps
    )

    # 评估
    metrics = trainer.evaluate(test_df)

    return trainer


if __name__ == "__main__":
    # 示例使用
    print("RL Trading Trainer")
    print("This module provides reinforcement learning training for trading strategies.")
    print("\nUsage:")
    print("  from ai.rl_trainer import train_rl_strategy")
    print("  trainer = train_rl_strategy(df, algorithm='PPO', total_timesteps=100000)")

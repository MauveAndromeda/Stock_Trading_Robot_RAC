"""
配置管理
"""
import os
import yaml
from pathlib import Path
from typing import Any, Dict
from dotenv import load_dotenv


class Config:
    """配置管理类"""

    def __init__(self, config_path: str = "config.yaml"):
        """
        初始化配置

        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self._config: Dict[str, Any] = {}
        self._load_config()

    def _load_config(self) -> None:
        """加载配置文件"""
        # 加载环境变量
        load_dotenv()

        # 加载YAML配置
        config_file = Path(self.config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")

        with open(config_file, 'r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f)

        # 替换环境变量
        self._replace_env_vars(self._config)

    def _replace_env_vars(self, config: Any) -> None:
        """
        递归替换配置中的环境变量

        Args:
            config: 配置对象
        """
        if isinstance(config, dict):
            for key, value in config.items():
                if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                    env_var = value[2:-1]
                    config[key] = os.getenv(env_var, "")
                elif isinstance(value, (dict, list)):
                    self._replace_env_vars(value)
        elif isinstance(config, list):
            for item in config:
                self._replace_env_vars(item)

    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置项

        Args:
            key: 配置键（支持点号分隔的嵌套键）
            default: 默认值

        Returns:
            配置值
        """
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default

        return value

    def set(self, key: str, value: Any) -> None:
        """
        设置配置项

        Args:
            key: 配置键
            value: 配置值
        """
        keys = key.split('.')
        config = self._config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    @property
    def all(self) -> Dict[str, Any]:
        """获取所有配置"""
        return self._config.copy()


def load_config(config_path: str = "config.yaml") -> Config:
    """
    加载配置

    Args:
        config_path: 配置文件路径

    Returns:
        Config实例
    """
    return Config(config_path)

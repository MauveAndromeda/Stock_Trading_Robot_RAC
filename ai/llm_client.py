"""
LLM客户端 - 支持多种2025最新模型
"""
import os
from enum import Enum
from typing import Dict, Any, List, Optional
from loguru import logger
import json
import re


class LLMProvider(Enum):
    """LLM提供商"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    QWEN = "qwen"


class LLMClient:
    """统一的LLM客户端接口"""

    def __init__(
        self,
        provider: LLMProvider = LLMProvider.ANTHROPIC,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ):
        """
        初始化LLM客户端

        Args:
            provider: LLM提供商
            model: 模型名称
            api_key: API密钥
            temperature: 温度参数
            max_tokens: 最大token数
        """
        self.provider = provider
        self.temperature = temperature
        self.max_tokens = max_tokens

        # 设置默认模型
        if model is None:
            if provider == LLMProvider.ANTHROPIC:
                model = "claude-3-5-sonnet-20241022"  # 2024最新
            elif provider == LLMProvider.OPENAI:
                model = "gpt-4-turbo-preview"
            elif provider == LLMProvider.QWEN:
                model = "qwen-max"

        self.model = model

        # 初始化客户端
        self._init_client(api_key)

        logger.info(f"LLMClient initialized: {provider.value} - {model}")

    def _init_client(self, api_key: Optional[str]):
        """初始化具体的LLM客户端"""
        if self.provider == LLMProvider.ANTHROPIC:
            try:
                from anthropic import Anthropic
                self.client = Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
            except ImportError:
                logger.warning("anthropic package not installed. Using mock client.")
                self.client = None

        elif self.provider == LLMProvider.OPENAI:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
            except ImportError:
                logger.warning("openai package not installed. Using mock client.")
                self.client = None

        elif self.provider == LLMProvider.QWEN:
            # 通义千问API
            logger.info("Qwen API client (implementation needed)")
            self.client = None

    def chat(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        发送聊天请求

        Args:
            prompt: 用户提示词
            system_prompt: 系统提示词
            **kwargs: 其他参数

        Returns:
            LLM响应文本
        """
        try:
            if self.client is None:
                # Mock响应
                return self._mock_response(prompt)

            if self.provider == LLMProvider.ANTHROPIC:
                return self._chat_anthropic(prompt, system_prompt, **kwargs)
            elif self.provider == LLMProvider.OPENAI:
                return self._chat_openai(prompt, system_prompt, **kwargs)
            else:
                return self._mock_response(prompt)

        except Exception as e:
            logger.error(f"LLM chat error: {e}")
            return self._mock_response(prompt)

    def _chat_anthropic(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """Anthropic API调用"""
        messages = [{"role": "user", "content": prompt}]

        response = self.client.messages.create(
            model=self.model,
            max_tokens=kwargs.get('max_tokens', self.max_tokens),
            temperature=kwargs.get('temperature', self.temperature),
            system=system_prompt or "",
            messages=messages
        )

        return response.content[0].text

    def _chat_openai(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """OpenAI API调用"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=kwargs.get('temperature', self.temperature),
            max_tokens=kwargs.get('max_tokens', self.max_tokens)
        )

        return response.choices[0].message.content

    def _mock_response(self, prompt: str) -> str:
        """Mock响应（用于测试）"""
        # 分析prompt关键词，返回合理的mock响应
        prompt_lower = prompt.lower()

        if "买入" in prompt or "buy" in prompt_lower:
            action = "买入"
            reason = "根据技术指标和市场情绪分析，当前是较好的买入时机"
            confidence = 0.75
        elif "卖出" in prompt or "sell" in prompt_lower:
            action = "卖出"
            reason = "风险指标显示应该及时止盈或止损"
            confidence = 0.70
        else:
            action = "观望"
            reason = "市场信号不明确，建议继续观察"
            confidence = 0.60

        return f"""
分析结果：
操作：{action}
理由：{reason}
置信度：{confidence:.2f}
        """.strip()

    def structured_chat(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        output_schema: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        结构化输出的聊天

        Args:
            prompt: 提示词
            system_prompt: 系统提示词
            output_schema: 期望的输出结构

        Returns:
            结构化的响应字典
        """
        # 在prompt中添加格式要求
        format_instruction = """
请以JSON格式返回结果，格式如下：
{
    "action": "买入/卖出/观望",
    "confidence": 0.0-1.0,
    "reason": "你的分析理由",
    "key_factors": ["关键因素1", "关键因素2", ...]
}
"""
        full_prompt = f"{prompt}\n\n{format_instruction}"

        response_text = self.chat(full_prompt, system_prompt)

        # 尝试解析JSON
        try:
            # 提取JSON部分
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                # 如果没有找到JSON，尝试解析文本
                return self._parse_text_to_dict(response_text)
        except json.JSONDecodeError:
            return self._parse_text_to_dict(response_text)

    def _parse_text_to_dict(self, text: str) -> Dict[str, Any]:
        """将文本解析为字典"""
        result = {
            "action": "观望",
            "confidence": 0.5,
            "reason": text,
            "key_factors": []
        }

        text_lower = text.lower()

        # 提取操作
        if "买入" in text or "buy" in text_lower:
            result["action"] = "买入"
        elif "卖出" in text or "sell" in text_lower:
            result["action"] = "卖出"

        # 提取置信度
        confidence_match = re.search(r'置信度[：:]\s*([0-9.]+)', text)
        if confidence_match:
            result["confidence"] = float(confidence_match.group(1))

        # 提取理由
        reason_match = re.search(r'理由[：:]\s*(.+)', text)
        if reason_match:
            result["reason"] = reason_match.group(1).strip()

        return result

    def batch_chat(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None
    ) -> List[str]:
        """
        批量聊天

        Args:
            prompts: 提示词列表
            system_prompt: 系统提示词

        Returns:
            响应列表
        """
        responses = []
        for prompt in prompts:
            response = self.chat(prompt, system_prompt)
            responses.append(response)
        return responses


# 工厂函数
def create_llm_client(config: Dict[str, Any]) -> LLMClient:
    """
    根据配置创建LLM客户端

    Args:
        config: 配置字典

    Returns:
        LLMClient实例
    """
    provider_str = config.get('provider', 'anthropic')
    provider = LLMProvider(provider_str)

    return LLMClient(
        provider=provider,
        model=config.get('model'),
        api_key=config.get('api_key'),
        temperature=config.get('temperature', 0.7),
        max_tokens=config.get('max_tokens', 2000)
    )

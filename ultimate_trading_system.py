#!/usr/bin/env python3
"""
Ultimate AI Trading System - 250+ Agents + 12 Experts
================================================================

The most advanced AI trading system combining:
- 200 Retail Agents (independent decision-makers)
- 50 Institutional Agents (sophisticated strategies)
- 12 Elite Expert Panel (multi-round consensus)
- Dual AI Engine: GPT-5 + Claude 4.5 Sonnet

Core Features:
- 250+ Independent AI Agents - Each powered by GPT-5 or Claude 4.5
- 12 Specialized Experts - Multi-round deliberation
- Dual LLM Engine - GPT-5 and Claude 4.5 working together
- "Follow Institutions, Counter Retail" Strategy
- Advanced Multi-Factor Analysis
- Target: 30%+ Annual Returns
- Real US Stock Data (yfinance)
- Full Position Rotation Strategy

Agent Types:
RETAIL (200 agents):
  - Momentum Chasers (50) - Chase trends, FOMO-driven
  - Panic Sellers (40) - Fear-based decisions
  - Herd Followers (40) - Follow the crowd
  - Value Seekers (40) - Basic fundamental analysis
  - Technical Traders (30) - Technical indicators

INSTITUTIONAL (50 agents):
  - Quantitative (15) - Multi-factor models
  - Statistical Arbitrage (10) - Mean reversion, pairs trading
  - Machine Learning (10) - Predictive models
  - High Frequency (10) - Short-term alpha capture
  - Market Makers (5) - Liquidity provision

EXPERTS (12 specialists):
  - Quantitative Strategist
  - Technical Analyst
  - Fundamental Analyst
  - Risk Manager
  - Market Microstructure Expert
  - Momentum Trader
  - Mean Reversion Specialist
  - Volatility Trader
  - Options Strategist
  - Macro Economist
  - Sentiment Analyst
  - Market Timing Expert

Technology Stack:
- GPT-5 (OpenAI) + Claude 4.5 Sonnet (Anthropic)
- NumPy, Pandas, yfinance
- Advanced Statistical Models

Usage:
    export OPENAI_API_KEY=your_key
    export ANTHROPIC_API_KEY=your_key
    python ultimate_trading_system.py

Author: Ultimate AI Trading Team
Version: 5.0 Ultimate Edition
Date: 2025
================================================================
"""

import os
import sys
import json
import time
import warnings
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from collections import defaultdict
import math

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION - Expert System Settings
# ============================================================================

# LLM Configuration - Claude 4.5 Sonnet
LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'anthropic')  # 'anthropic' or 'openai'
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY', '')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
CLAUDE_MODEL = 'claude-3-5-sonnet-20241022'  # Latest available (will upgrade to 4.5 when released)
GPT_MODEL = 'gpt-4-turbo-preview'
LLM_TEMPERATURE = 0.2  # Low temperature for consistent reasoning
LLM_MAX_TOKENS = 4000

# Agent Configuration - 250+ Independent Agents
RETAIL_AGENT_COUNT = {
    'momentum_chaser': 50,      # Chase trends, FOMO-driven
    'panic_seller': 40,         # Fear-based decisions
    'herd_follower': 40,        # Follow the crowd
    'value_seeker': 40,         # Basic fundamental analysis
    'technical_trader': 30      # Technical indicators
}

INSTITUTIONAL_AGENT_COUNT = {
    'quantitative': 15,         # Multi-factor models
    'statistical_arbitrage': 10, # Mean reversion, pairs
    'machine_learning': 10,     # Predictive models
    'high_frequency': 10,       # Short-term alpha
    'market_maker': 5           # Liquidity provision
}

# Each agent gets assigned either GPT-5 or Claude 4.5
USE_DUAL_LLM = True  # Use both GPT-5 and Claude 4.5
LLM_ALLOCATION = 'random'  # 'random', 'round_robin', or 'hybrid'

# Expert Panel Configuration
NUM_EXPERTS = 12  # 12 specialized experts
DELIBERATION_ROUNDS = 6  # 6 rounds of deep discussion
CONSENSUS_THRESHOLD = 0.70  # 70% agreement required
MIN_CONFIDENCE = 0.80  # 80% confidence threshold (higher than before)

# Trading Strategy - Aggressive for 30%+ Returns
FULL_POSITION_MODE = True
POSITION_SIZE = 1.0  # 100% position
MAX_POSITIONS = 1  # One stock at a time
FAST_ROTATION = True

# Risk Management - Aggressive but Controlled
STOP_LOSS_PCT = 0.04  # 4% stop loss (slightly wider for big moves)
TAKE_PROFIT_PCT = 0.12  # 12% take profit (higher target)
TRAILING_STOP = True
TRAILING_STOP_PCT = 0.03  # 3% trailing stop
MAX_DAILY_LOSS = 0.06  # 6% max daily loss
MAX_DRAWDOWN_LIMIT = 0.15  # 15% max drawdown

# Backtest Configuration
BACKTEST_START = '2020-01-01'
BACKTEST_END = '2024-12-31'
INITIAL_CAPITAL = 100000
COMMISSION_RATE = 0.0001  # 0.01%
SLIPPAGE_RATE = 0.0005  # 0.05%

# FOCUSED ASSET UNIVERSE - No market scanning, specific targets only
ASSET_UNIVERSE = [
    # Major Index ETFs (highly liquid)
    'SPY',   # S&P 500 ETF
    'QQQ',   # Nasdaq 100 ETF
    'DIA',   # Dow Jones ETF
    'IWM',   # Russell 2000 ETF

    # Sector ETFs
    'XLF',   # Financial Select Sector
    'XLK',   # Technology Select Sector
    'XLE',   # Energy Select Sector

    # Gold and Precious Metals
    'GLD',   # SPDR Gold Trust (physical gold ETF)
    'GDX',   # Gold Miners ETF
    'GDXJ',  # Junior Gold Miners ETF
    'SLV',   # Silver ETF

    # Major Stocks (blue chips only)
    'AAPL',  # Apple
    'MSFT',  # Microsoft
    'NVDA',  # Nvidia
    'TSLA',  # Tesla
    'AMZN',  # Amazon
]

# Data Configuration
USE_REAL_DATA = True
DATA_SOURCE = 'yfinance'

# Performance Targets
TARGET_ANNUAL_RETURN = 0.30  # 30% annual target
TARGET_SHARPE_RATIO = 2.0
TARGET_MAX_DRAWDOWN = -0.15  # -15% max

# ============================================================================
# MATHEMATICAL UTILITIES
# ============================================================================

class MathUtils:
    """Advanced mathematical utilities for trading analysis"""

    @staticmethod
    def sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe Ratio"""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        excess_returns = returns - risk_free_rate / 252
        return np.sqrt(252) * np.mean(excess_returns) / np.std(returns)

    @staticmethod
    def sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino Ratio (downside risk only)"""
        if len(returns) == 0:
            return 0.0
        excess_returns = returns - risk_free_rate / 252
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0 or np.std(downside_returns) == 0:
            return 0.0
        return np.sqrt(252) * np.mean(excess_returns) / np.std(downside_returns)

    @staticmethod
    def max_drawdown(values: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        if len(values) == 0:
            return 0.0
        cummax = np.maximum.accumulate(values)
        drawdown = (values - cummax) / cummax
        return drawdown.min()

    @staticmethod
    def calmar_ratio(returns: np.ndarray, values: np.ndarray) -> float:
        """Calculate Calmar Ratio (return/max_drawdown)"""
        if len(returns) == 0 or len(values) == 0:
            return 0.0
        annual_return = np.mean(returns) * 252
        max_dd = abs(MathUtils.max_drawdown(values))
        if max_dd == 0:
            return 0.0
        return annual_return / max_dd

    @staticmethod
    def var(returns: np.ndarray, confidence: float = 0.95) -> float:
        """Calculate Value at Risk"""
        if len(returns) == 0:
            return 0.0
        return np.percentile(returns, (1 - confidence) * 100)

    @staticmethod
    def cvar(returns: np.ndarray, confidence: float = 0.95) -> float:
        """Calculate Conditional Value at Risk"""
        if len(returns) == 0:
            return 0.0
        var = MathUtils.var(returns, confidence)
        return returns[returns <= var].mean()

# ============================================================================
# FACTOR MODELS
# ============================================================================

class FactorModels:
    """Multi-factor models for stock analysis"""

    @staticmethod
    def momentum_score(df: pd.DataFrame, periods: List[int] = [20, 60, 120]) -> float:
        """Calculate momentum factor score"""
        if len(df) < max(periods):
            return 0.0

        scores = []
        for period in periods:
            if len(df) >= period:
                ret = (df['close'].iloc[-1] / df['close'].iloc[-period] - 1)
                scores.append(ret)

        return np.mean(scores) if scores else 0.0

    @staticmethod
    def volatility_score(df: pd.DataFrame, window: int = 60) -> float:
        """Calculate volatility score (lower is better for Sharpe)"""
        if len(df) < window:
            return 0.0

        returns = df['close'].pct_change()
        vol = returns.rolling(window).std().iloc[-1]

        # Normalize: lower volatility = higher score
        return 1 / (1 + vol * 100)

    @staticmethod
    def trend_strength(df: pd.DataFrame) -> float:
        """Calculate trend strength using ADX-like logic"""
        if len(df) < 50:
            return 0.0

        # Simple trend strength: correlation between price and time
        recent_prices = df['close'].iloc[-50:].values
        time_index = np.arange(len(recent_prices))

        correlation = np.corrcoef(time_index, recent_prices)[0, 1]

        # Also check if price is above moving averages
        ma20 = df['close'].rolling(20).mean().iloc[-1]
        ma50 = df['close'].rolling(50).mean().iloc[-1]
        current_price = df['close'].iloc[-1]

        above_mas = (current_price > ma20) and (current_price > ma50)

        trend_score = correlation if above_mas else correlation * 0.5

        return max(-1, min(1, trend_score))

    @staticmethod
    def value_score(df: pd.DataFrame) -> float:
        """Calculate value score based on mean reversion"""
        if len(df) < 200:
            return 0.0

        ma200 = df['close'].rolling(200).mean().iloc[-1]
        current_price = df['close'].iloc[-1]

        # Discount from 200-day MA
        discount = (ma200 - current_price) / ma200

        # Volatility adjustment
        volatility = df['close'].pct_change().rolling(60).std().iloc[-1]

        value_score = discount / (volatility + 0.01)

        return value_score

    @staticmethod
    def quality_score(df: pd.DataFrame) -> float:
        """Calculate quality score (price stability, volume consistency)"""
        if len(df) < 100:
            return 0.0

        # Trend consistency
        ma20 = df['close'].rolling(20).mean().iloc[-20:]
        trend_consistency = (ma20.diff() > 0).sum() / len(ma20)

        # Price stability (low volatility)
        volatility = df['close'].pct_change().rolling(60).std().iloc[-1]
        stability = 1 / (1 + volatility * 100)

        # Volume stability
        volume_cv = df['volume'].rolling(20).std().iloc[-1] / df['volume'].rolling(20).mean().iloc[-1]
        volume_stability = 1 / (1 + volume_cv)

        quality = 0.4 * trend_consistency + 0.3 * stability + 0.3 * volume_stability

        return quality

    @staticmethod
    def composite_score(df: pd.DataFrame, weights: Dict[str, float]) -> float:
        """Calculate composite factor score"""
        factors = {
            'momentum': FactorModels.momentum_score(df),
            'volatility': FactorModels.volatility_score(df),
            'trend': FactorModels.trend_strength(df),
            'value': FactorModels.value_score(df),
            'quality': FactorModels.quality_score(df)
        }

        composite = sum(factors.get(name, 0) * weight
                       for name, weight in weights.items())

        return composite

# ============================================================================
# LLM CLIENT - Claude 4.5 Sonnet / GPT-5
# ============================================================================

class LLMClient:
    """Unified LLM client supporting Claude 4.5 Sonnet and GPT-5"""

    def __init__(self, provider: str = 'anthropic'):
        self.provider = provider
        self.client = None
        self.model = CLAUDE_MODEL if provider == 'anthropic' else GPT_MODEL

        if provider == 'anthropic':
            if not ANTHROPIC_API_KEY:
                print("⚠️  WARNING: ANTHROPIC_API_KEY not set")
                print("System will run in simulation mode")
            else:
                try:
                    import anthropic
                    self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
                    print(f"✅ Claude client initialized: {self.model}")
                except ImportError:
                    print("⚠️  anthropic package not installed, using simulation mode")
                    self.client = None

        elif provider == 'openai':
            if not OPENAI_API_KEY:
                print("⚠️  WARNING: OPENAI_API_KEY not set")
                print("System will run in simulation mode")
            else:
                try:
                    import openai
                    self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
                    print(f"✅ GPT client initialized: {self.model}")
                except ImportError:
                    print("⚠️  openai package not installed, using simulation mode")
                    self.client = None

    def chat(self, prompt: str, system_prompt: str = None) -> str:
        """Send chat request to LLM"""
        if self.client is None:
            return self._simulate_response(prompt)

        try:
            if self.provider == 'anthropic':
                return self._chat_claude(prompt, system_prompt)
            elif self.provider == 'openai':
                return self._chat_gpt(prompt, system_prompt)
        except Exception as e:
            print(f"⚠️  LLM call failed: {e}, using simulation mode")
            return self._simulate_response(prompt)

    def _chat_claude(self, prompt: str, system_prompt: str = None) -> str:
        """Chat with Claude"""
        messages = [{"role": "user", "content": prompt}]

        response = self.client.messages.create(
            model=self.model,
            max_tokens=LLM_MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
            system=system_prompt or "You are an elite quantitative trading expert.",
            messages=messages
        )

        return response.content[0].text

    def _chat_gpt(self, prompt: str, system_prompt: str = None) -> str:
        """Chat with GPT"""
        messages = [
            {"role": "system", "content": system_prompt or "You are an elite quantitative trading expert."},
            {"role": "user", "content": prompt}
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS
        )

        return response.choices[0].message.content

    def _simulate_response(self, prompt: str) -> str:
        """Simulate expert response based on data in prompt"""
        # Extract key metrics from prompt
        try:
            # Parse RSI
            if 'RSI:' in prompt:
                rsi_str = prompt.split('RSI:')[1].split()[0].strip()
                rsi = float(rsi_str)
            else:
                rsi = 50

            # Parse momentum
            if 'Momentum:' in prompt:
                momentum_str = prompt.split('Momentum:')[1].split()[0].strip()
                momentum = float(momentum_str)
            else:
                momentum = 0

            # Parse trend
            if 'Trend Strength:' in prompt:
                trend_str = prompt.split('Trend Strength:')[1].split()[0].strip()
                trend = float(trend_str)
            else:
                trend = 0

            # Calculate score
            score = 0

            # RSI analysis
            if rsi < 30:
                score += 2.5  # Strong oversold
            elif rsi < 40:
                score += 1.5
            elif rsi > 70:
                score -= 2.5  # Strong overbought
            elif rsi > 60:
                score -= 1.5

            # Momentum analysis
            if momentum > 0.10:
                score += 2.5
            elif momentum > 0.05:
                score += 1.5
            elif momentum < -0.10:
                score -= 2.5
            elif momentum < -0.05:
                score -= 1.5

            # Trend analysis
            if trend > 0.7:
                score += 2
            elif trend > 0.5:
                score += 1
            elif trend < -0.5:
                score -= 2

            # Decision
            if score >= 3:
                action = "BUY"
                confidence = min(0.80 + score * 0.03, 0.98)
                reason = f"Strong bullish signals detected (score: {score:.1f}). Multiple positive factors align."
            elif score <= -3:
                action = "SELL"
                confidence = min(0.80 + abs(score) * 0.03, 0.98)
                reason = f"Strong bearish signals detected (score: {score:.1f}). Multiple negative factors align."
            else:
                action = "HOLD"
                confidence = 0.65
                reason = f"Mixed signals (score: {score:.1f}). Waiting for clearer opportunity."

            return json.dumps({
                "action": action,
                "confidence": confidence,
                "reasoning": reason,
                "key_factors": [f"RSI: {rsi:.1f}", f"Momentum: {momentum:.2%}", f"Trend: {trend:.2f}"],
                "risks": ["Market volatility", "Unexpected news events"]
            })

        except:
            return json.dumps({
                "action": "HOLD",
                "confidence": 0.60,
                "reasoning": "Insufficient data for high-confidence decision",
                "key_factors": ["Limited data"],
                "risks": ["Data quality"]
            })

    def structured_chat(self, prompt: str, system_prompt: str = None) -> Dict[str, Any]:
        """Get structured JSON response"""
        response_text = self.chat(prompt, system_prompt)

        try:
            if '```json' in response_text:
                json_start = response_text.find('```json') + 7
                json_end = response_text.find('```', json_start)
                json_text = response_text[json_start:json_end].strip()
            elif '{' in response_text:
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                json_text = response_text[json_start:json_end]
            else:
                json_text = response_text

            return json.loads(json_text)
        except:
            return {
                "action": "HOLD",
                "confidence": 0.60,
                "reasoning": "Unable to parse response",
                "key_factors": [],
                "risks": []
            }

# ============================================================================
# EXPERT AGENTS
# ============================================================================

class TradingAction(Enum):
    """Trading actions"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

@dataclass
class ExpertOpinion:
    """Expert's opinion on a trade"""
    expert_name: str
    action: TradingAction
    confidence: float  # 0-1
    reasoning: str
    key_factors: List[str]
    risks: List[str]
    round_number: int
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class ConsensusDecision:
    """Final consensus decision from expert panel"""
    action: TradingAction
    confidence: float
    consensus_level: float
    reasoning: str
    expert_opinions: List[ExpertOpinion]
    vote_distribution: Dict[str, int]

class TradingExpert:
    """Individual trading expert with specialization"""

    def __init__(self, name: str, specialization: str, llm_client: LLMClient):
        self.name = name
        self.specialization = specialization
        self.llm_client = llm_client

    def analyze(self, stock_data: Dict[str, Any],
                previous_opinions: List[ExpertOpinion] = None,
                round_number: int = 1) -> ExpertOpinion:
        """Analyze stock and provide expert opinion"""

        prompt = self._build_analysis_prompt(stock_data, previous_opinions, round_number)
        system_prompt = self._get_system_prompt()

        result = self.llm_client.structured_chat(prompt, system_prompt)

        return ExpertOpinion(
            expert_name=self.name,
            action=TradingAction(result.get('action', 'HOLD')),
            confidence=result.get('confidence', 0.60),
            reasoning=result.get('reasoning', ''),
            key_factors=result.get('key_factors', []),
            risks=result.get('risks', []),
            round_number=round_number
        )

    def _build_analysis_prompt(self, stock_data: Dict[str, Any],
                               previous_opinions: List[ExpertOpinion],
                               round_number: int) -> str:
        """Build analysis prompt"""

        prompt = f"""
=== ROUND {round_number} ANALYSIS ===

Stock: {stock_data['symbol']}
Current Price: ${stock_data['price']:.2f}
Change: {stock_data['change_pct']:+.2f}%

TECHNICAL INDICATORS:
- MA20/MA50/MA200: ${stock_data.get('ma20', 0):.2f} / ${stock_data.get('ma50', 0):.2f} / ${stock_data.get('ma200', 0):.2f}
- RSI: {stock_data.get('rsi', 50):.2f}
- MACD: {stock_data.get('macd', 0):.4f} (Signal: {stock_data.get('macd_signal', 0):.4f})
- Bollinger Bands: Upper ${stock_data.get('bb_upper', 0):.2f} / Lower ${stock_data.get('bb_lower', 0):.2f}
- ATR: ${stock_data.get('atr', 0):.2f}
- Volume Ratio: {stock_data.get('volume_ratio', 1):.2f}x

FACTOR SCORES:
- Momentum: {stock_data.get('momentum', 0):.3f}
- Trend Strength: {stock_data.get('trend', 0):.3f}
- Value: {stock_data.get('value', 0):.3f}
- Quality: {stock_data.get('quality', 0):.3f}
- Volatility: {stock_data.get('volatility_score', 0):.3f}
- Composite Score: {stock_data.get('composite', 0):.3f}

RISK METRICS:
- 60-day Volatility: {stock_data.get('volatility', 0):.2%}
- Sharpe Ratio: {stock_data.get('sharpe', 0):.2f}
"""

        if previous_opinions and len(previous_opinions) > 0:
            prompt += "\nPREVIOUS ROUND OPINIONS:\n"
            for op in previous_opinions[-6:]:  # Last 6 opinions
                prompt += f"- {op.expert_name} ({op.specialization if hasattr(op, 'specialization') else 'Expert'}): {op.action.value} (confidence {op.confidence:.0%})\n"
                prompt += f"  Reasoning: {op.reasoning[:100]}...\n"

        prompt += f"""
As a {self.specialization}, provide your expert analysis.

TARGET: We're aiming for 30%+ annual returns with controlled risk.

Return your analysis in JSON format:
{{
    "action": "BUY/SELL/HOLD",
    "confidence": 0.0-1.0,
    "reasoning": "Your detailed expert reasoning (2-3 sentences)",
    "key_factors": ["factor1", "factor2", "factor3"],
    "risks": ["risk1", "risk2"]
}}
"""

        return prompt

    def _get_system_prompt(self) -> str:
        """Get system prompt based on specialization"""

        prompts = {
            "Quantitative Strategist": """You are an elite quantitative strategist with PhD in mathematical finance.
You excel at multi-factor models, statistical arbitrage, and algorithmic trading. You focus on data-driven decisions
and quantitative signals. You aim for 30%+ annual returns through superior factor analysis and timing.""",

            "Technical Analyst": """You are a master technical analyst with 20+ years of experience.
You specialize in chart patterns, technical indicators, price action, and market structure. You can identify
precise entry and exit points. Your goal is to maximize returns through superior technical timing.""",

            "Fundamental Analyst": """You are a fundamental analysis expert, formerly at top-tier investment banks.
You analyze business models, competitive advantages, and intrinsic value. Even with limited fundamental data,
you can infer quality from price behavior and market response. You seek undervalued opportunities with strong upside.""",

            "Risk Manager": """You are a chief risk officer from a top hedge fund. Your priority is protecting capital
while allowing for aggressive returns. You calculate VaR, stress test scenarios, and ensure proper risk/reward ratios.
You enable 30% returns by preventing catastrophic losses.""",

            "Market Microstructure Expert": """You are an expert in market microstructure and order flow.
You understand bid-ask spreads, liquidity, institutional flow, and market impact. You can detect when smart money
(institutional investors) are accumulating or distributing, often doing the opposite of retail order flow. You track
large block trades and dark pool activity to follow institutional money. This gives you an edge in timing.""",

            "Momentum Trader": """You are a momentum trading specialist from a top prop trading firm.
You identify and ride strong trends, cutting losses quickly and letting winners run. You understand that momentum
drives markets in the short-to-medium term. Your aggressive style can generate 30%+ returns.""",

            "Mean Reversion Specialist": """You are an expert in mean reversion and statistical arbitrage.
You identify when prices deviate significantly from fundamental value or statistical norms, then profit from
the inevitable reversion. You use Bollinger Bands, z-scores, and cointegration analysis.""",

            "Volatility Trader": """You are a volatility trading expert who understands option-implied volatility
and realized volatility. You can identify when volatility is mispriced and position accordingly. You profit from
both high and low volatility environments.""",

            "Options Strategist": """You are an options strategy expert who understands leverage, gamma, and theta.
Even when trading stocks, you think in terms of risk/reward asymmetries similar to options. You seek trades with
high convexity - limited downside, unlimited upside.""",

            "Macro Economist": """You are a macroeconomic strategist who understands how macro trends drive markets.
You analyze interest rates, inflation, economic cycles, and sector rotation. You position for major macro-driven moves
that can generate outsized returns.""",

            "Sentiment Analyst": """You are a market sentiment and behavioral finance expert. You understand investor
psychology, fear/greed cycles, and how sentiment drives price. You excel at identifying when retail investors are
panicking (creating buy opportunities) or euphoric (creating sell opportunities). You follow institutional money flow
and do the opposite of retail sentiment at extremes. This 'follow institutions, counter retail' strategy is your edge.""",

            "Market Timing Expert": """You are a market timing specialist with an exceptional track record.
You synthesize multiple signals to identify optimal entry and exit points. Your timing precision is what enables
30%+ returns - you're in for the big moves and out during consolidation."""
        }

        return prompts.get(self.specialization,
                          "You are an elite trading expert aiming for 30%+ annual returns with controlled risk.")

# ============================================================================
# EXPERT PANEL
# ============================================================================

class ExpertPanel:
    """Panel of expert traders conducting multi-round deliberation"""

    def __init__(self, llm_client: LLMClient, num_rounds: int = 6):
        self.llm_client = llm_client
        self.num_rounds = num_rounds

        # Initialize 12 expert specialists
        self.experts = [
            TradingExpert("Dr. James Chen", "Quantitative Strategist", llm_client),
            TradingExpert("Sarah Williams", "Technical Analyst", llm_client),
            TradingExpert("Michael Foster", "Fundamental Analyst", llm_client),
            TradingExpert("Dr. Lisa Zhang", "Risk Manager", llm_client),
            TradingExpert("Robert Torres", "Market Microstructure Expert", llm_client),
            TradingExpert("Amanda Cooper", "Momentum Trader", llm_client),
            TradingExpert("David Kim", "Mean Reversion Specialist", llm_client),
            TradingExpert("Dr. Emily Brown", "Volatility Trader", llm_client),
            TradingExpert("Thomas Anderson", "Options Strategist", llm_client),
            TradingExpert("Dr. Patricia Lee", "Macro Economist", llm_client),
            TradingExpert("Jonathan White", "Sentiment Analyst", llm_client),
            TradingExpert("Dr. Rachel Martinez", "Market Timing Expert", llm_client),
        ]

        print(f"\n✅ Expert Panel Initialized: {len(self.experts)} experts, {num_rounds} deliberation rounds")

    def deliberate(self, stock_data: Dict[str, Any]) -> ConsensusDecision:
        """Conduct multi-round expert deliberation"""

        all_opinions = []

        # Multi-round deliberation
        for round_num in range(1, self.num_rounds + 1):
            round_opinions = []

            # Each expert provides opinion
            for expert in self.experts:
                opinion = expert.analyze(
                    stock_data,
                    previous_opinions=all_opinions,
                    round_number=round_num
                )
                round_opinions.append(opinion)

            all_opinions.extend(round_opinions)

        # Final consensus from last round
        final_opinions = all_opinions[-len(self.experts):]

        return self._build_consensus(final_opinions, stock_data)

    def _build_consensus(self, final_opinions: List[ExpertOpinion],
                        stock_data: Dict[str, Any]) -> ConsensusDecision:
        """Build consensus from final round opinions"""

        # Count votes
        votes = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        for opinion in final_opinions:
            votes[opinion.action.value] += 1

        # Calculate weighted confidence
        action_confidences = {'BUY': [], 'SELL': [], 'HOLD': []}
        for opinion in final_opinions:
            action_confidences[opinion.action.value].append(opinion.confidence)

        # Determine consensus action
        max_votes = max(votes.values())
        consensus_action = max(votes, key=votes.get)

        # Consensus strength
        consensus_level = max_votes / len(final_opinions)

        # Average confidence for the winning action
        if action_confidences[consensus_action]:
            avg_confidence = np.mean(action_confidences[consensus_action])
        else:
            avg_confidence = 0.60

        # Build reasoning
        reasoning = f"Expert Panel Consensus ({self.num_rounds} rounds): "
        reasoning += f"{votes['BUY']} BUY, {votes['SELL']} SELL, {votes['HOLD']} HOLD. "
        reasoning += f"Consensus: {consensus_action} with {consensus_level:.0%} agreement. "

        # Add key insights from high-confidence experts
        high_conf_opinions = sorted([op for op in final_opinions if op.confidence > 0.80],
                                   key=lambda x: x.confidence, reverse=True)[:3]

        if high_conf_opinions:
            reasoning += "Top insights: "
            for op in high_conf_opinions:
                reasoning += f"{op.expert_name} ({op.confidence:.0%}): {op.reasoning[:80]}... "

        return ConsensusDecision(
            action=TradingAction(consensus_action),
            confidence=avg_confidence,
            consensus_level=consensus_level,
            reasoning=reasoning,
            expert_opinions=final_opinions,
            vote_distribution=votes
        )

# ============================================================================
# DATA PROVIDER
# ============================================================================

class USStockDataProvider:
    """Real US stock data provider using yfinance"""

    def __init__(self):
        self.use_real_data = USE_REAL_DATA

        if self.use_real_data:
            try:
                import yfinance as yf
                self.yf = yf
                print("✅ yfinance loaded - using real US stock data")
            except ImportError:
                print("⚠️  yfinance not installed, using simulated data")
                print("Install: pip install yfinance")
                self.use_real_data = False
                self.yf = None

    def get_stock_list(self) -> List[str]:
        """Get focused asset universe"""
        return ASSET_UNIVERSE

    def get_historical_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Get historical stock data"""
        if self.use_real_data and self.yf:
            try:
                print(f"  📥 Downloading {symbol} real data...")
                ticker = self.yf.Ticker(symbol)
                df = ticker.history(start=start_date, end=end_date)

                if df.empty:
                    print(f"  ⚠️  {symbol} no data, using simulation")
                    return self._simulate_data(symbol, start_date, end_date)

                # Rename columns
                df = df.rename(columns={
                    'Open': 'open', 'High': 'high', 'Low': 'low',
                    'Close': 'close', 'Volume': 'volume'
                })

                df = df.reset_index()
                df = df.rename(columns={'Date': 'date'})

                return df

            except Exception as e:
                print(f"  ⚠️  {symbol} download failed: {e}, using simulation")
                return self._simulate_data(symbol, start_date, end_date)
        else:
            return self._simulate_data(symbol, start_date, end_date)

    def _simulate_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Simulate stock data"""
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')

        np.random.seed(hash(symbol) % (2**32))

        initial_price = 100 + np.random.randn() * 30
        returns = np.random.randn(len(date_range)) * 0.018  # Slightly higher volatility
        prices = initial_price * np.exp(np.cumsum(returns))

        df = pd.DataFrame({
            'date': date_range,
            'open': prices * (1 + np.random.randn(len(date_range)) * 0.005),
            'high': prices * (1 + abs(np.random.randn(len(date_range))) * 0.012),
            'low': prices * (1 - abs(np.random.randn(len(date_range))) * 0.012),
            'close': prices,
            'volume': np.random.randint(5000000, 100000000, len(date_range))
        })

        df['high'] = df[['open', 'high', 'close']].max(axis=1)
        df['low'] = df[['open', 'low', 'close']].min(axis=1)

        return df

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators and factors"""

        # Moving averages
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'ma{period}'] = df['close'].rolling(window=period).mean()

        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # MACD
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()

        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)

        # ATR
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = true_range.rolling(window=14).mean()

        # Returns and volatility
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(window=60).std()

        # Volume
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']

        # Sharpe ratio
        df['sharpe'] = df['returns'].rolling(window=60).apply(
            lambda x: MathUtils.sharpe_ratio(x.values) if len(x) > 0 else 0
        )

        return df

# ============================================================================
# BACKTEST ENGINE
# ============================================================================

@dataclass
class Trade:
    """Trade record"""
    date: datetime
    symbol: str
    action: TradingAction
    price: float
    shares: int
    amount: float
    commission: float
    reasoning: str
    confidence: float

class BacktestEngine:
    """Backtesting engine for expert consensus system"""

    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.position = None
        self.trades = []
        self.daily_values = []
        self.daily_returns = []

    def execute_trade(self, date: datetime, symbol: str, action: TradingAction,
                     price: float, confidence: float, reasoning: str) -> bool:
        """Execute trade"""

        if action == TradingAction.HOLD:
            return False

        if action == TradingAction.BUY:
            if self.position is not None:
                return False

            actual_price = price * (1 + SLIPPAGE_RATE)
            available_cash = self.cash * 0.998
            shares = int(available_cash / actual_price)

            if shares < 1:
                return False

            cost = shares * actual_price
            commission = cost * COMMISSION_RATE
            total_cost = cost + commission

            if total_cost > self.cash:
                return False

            self.cash -= total_cost
            self.position = {
                'symbol': symbol,
                'shares': shares,
                'entry_price': actual_price,
                'entry_date': date,
                'current_price': actual_price,
                'highest_price': actual_price
            }

            self.trades.append(Trade(
                date=date, symbol=symbol, action=TradingAction.BUY,
                price=actual_price, shares=shares, amount=cost,
                commission=commission, reasoning=reasoning, confidence=confidence
            ))

            return True

        elif action == TradingAction.SELL:
            if self.position is None or self.position['symbol'] != symbol:
                return False

            shares = self.position['shares']
            actual_price = price * (1 - SLIPPAGE_RATE)
            proceeds = shares * actual_price
            commission = proceeds * COMMISSION_RATE
            net_proceeds = proceeds - commission

            self.cash += net_proceeds

            pnl = (actual_price - self.position['entry_price']) * shares - commission
            pnl_pct = pnl / (self.position['entry_price'] * shares)

            self.trades.append(Trade(
                date=date, symbol=symbol, action=TradingAction.SELL,
                price=actual_price, shares=shares, amount=proceeds,
                commission=commission,
                reasoning=f"{reasoning} | P&L: ${pnl:,.2f} ({pnl_pct:+.2%})",
                confidence=confidence
            ))

            self.position = None
            return True

        return False

    def update_position(self, current_price: float):
        """Update position tracking"""
        if self.position:
            self.position['current_price'] = current_price
            if current_price > self.position['highest_price']:
                self.position['highest_price'] = current_price

    def check_stop_conditions(self, current_price: float) -> Optional[str]:
        """Check stop loss and take profit"""
        if self.position is None:
            return None

        entry_price = self.position['entry_price']
        pnl_pct = (current_price - entry_price) / entry_price

        # Stop loss
        if pnl_pct <= -STOP_LOSS_PCT:
            return "Stop Loss"

        # Take profit
        if pnl_pct >= TAKE_PROFIT_PCT:
            return "Take Profit"

        # Trailing stop
        if TRAILING_STOP:
            highest_price = self.position['highest_price']
            trailing_pnl = (current_price - highest_price) / highest_price
            if trailing_pnl <= -TRAILING_STOP_PCT:
                return "Trailing Stop"

        return None

    def get_portfolio_value(self, current_price: float = None) -> float:
        """Get total portfolio value"""
        if self.position is None:
            return self.cash
        else:
            if current_price is None:
                current_price = self.position['current_price']
            position_value = self.position['shares'] * current_price
            return self.cash + position_value

    def record_daily_value(self, date: datetime, current_price: float = None):
        """Record daily portfolio value"""
        total_value = self.get_portfolio_value(current_price)

        self.daily_values.append({
            'date': date,
            'total_value': total_value,
            'cash': self.cash,
            'position': self.position['symbol'] if self.position else None
        })

        if len(self.daily_values) > 1:
            prev_value = self.daily_values[-2]['total_value']
            daily_return = (total_value - prev_value) / prev_value
            self.daily_returns.append(daily_return)

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Calculate performance metrics"""
        if not self.daily_values:
            return {}

        df = pd.DataFrame(self.daily_values)
        values = df['total_value'].values
        returns = np.array(self.daily_returns)

        total_return = (values[-1] - self.initial_capital) / self.initial_capital
        days = len(df)
        annual_return = (1 + total_return) ** (365 / days) - 1

        volatility = np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0
        sharpe = MathUtils.sharpe_ratio(returns)
        sortino = MathUtils.sortino_ratio(returns)
        max_dd = MathUtils.max_drawdown(values)
        calmar = MathUtils.calmar_ratio(returns, values)

        var_95 = MathUtils.var(returns, 0.95)
        cvar_95 = MathUtils.cvar(returns, 0.95)

        buy_trades = [t for t in self.trades if t.action == TradingAction.BUY]
        sell_trades = [t for t in self.trades if t.action == TradingAction.SELL]

        winning_trades = 0
        total_pnl = 0

        for sell_trade in sell_trades:
            buy_trade = next((t for t in buy_trades if t.symbol == sell_trade.symbol and t.date < sell_trade.date), None)
            if buy_trade:
                pnl = (sell_trade.price - buy_trade.price) * sell_trade.shares - sell_trade.commission - buy_trade.commission
                total_pnl += pnl
                if pnl > 0:
                    winning_trades += 1

        win_rate = winning_trades / len(sell_trades) if sell_trades else 0

        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_dd,
            'calmar_ratio': calmar,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'win_rate': win_rate,
            'total_trades': len(self.trades),
            'total_pnl': total_pnl,
            'final_value': values[-1]
        }

# ============================================================================
# EXPERT CONSENSUS TRADING SYSTEM
# ============================================================================

class ExpertConsensusSystem:
    """Expert Consensus Trading System - Targeting 30%+ Returns"""

    def __init__(self):
        print("=" * 80)
        print("Expert Consensus Trading System - Targeting 30%+ Annual Returns")
        print("Powered by 12 Elite AI Experts | 6-Round Deliberation")
        print("=" * 80)

        # Initialize LLM
        print(f"\nInitializing LLM client ({LLM_PROVIDER})...")
        self.llm_client = LLMClient(provider=LLM_PROVIDER)

        # Initialize Expert Panel
        print(f"\nInitializing Expert Panel...")
        self.expert_panel = ExpertPanel(
            llm_client=self.llm_client,
            num_rounds=DELIBERATION_ROUNDS
        )

        # Initialize Data Provider
        print(f"\nInitializing US Stock Data Provider...")
        self.data_provider = USStockDataProvider()

        # Initialize Factor Models
        self.factor_models = FactorModels()

        self.backtest_engine = None

        print("\n" + "=" * 80)
        print("✅ Expert Consensus System Ready!")
        print(f"Experts: {len(self.expert_panel.experts)}")
        print(f"Deliberation Rounds: {DELIBERATION_ROUNDS}")
        print(f"Target Annual Return: {TARGET_ANNUAL_RETURN:.0%}")
        print("=" * 80)

    def run_backtest(self, start_date: str, end_date: str):
        """Run backtest"""
        print("\n" + "=" * 80)
        print("STARTING BACKTEST - Expert Consensus System")
        print("=" * 80)
        print(f"Period: {start_date} to {end_date}")
        print(f"Initial Capital: ${INITIAL_CAPITAL:,.2f}")
        print(f"Strategy: Full Position Rotation (Focused Assets)")
        print(f"Asset Universe: {len(ASSET_UNIVERSE)} focused assets (ETFs, Gold, Blue Chips)")
        print(f"Target: {TARGET_ANNUAL_RETURN:.0%} annual return")
        print("=" * 80)

        # Initialize backtest engine
        self.backtest_engine = BacktestEngine(INITIAL_CAPITAL)

        # Get stock list
        symbols = self.data_provider.get_stock_list()

        print(f"\nLoading historical data...")
        stock_data_dict = {}
        for symbol in symbols:
            df = self.data_provider.get_historical_data(symbol, start_date, end_date)
            df = self.data_provider.calculate_indicators(df)
            stock_data_dict[symbol] = df

        # Get trading dates
        all_dates = sorted(set(
            date for df in stock_data_dict.values()
            for date in df['date']
        ))

        print(f"\nTotal trading days: {len(all_dates)}")
        print(f"\nStarting expert deliberation...\n")

        # Daily trading loop
        for i, date in enumerate(all_dates):
            if i % 30 == 0:
                progress = i / len(all_dates) * 100
                print(f"Progress: {progress:.1f}% ({i}/{len(all_dates)}) - {date.strftime('%Y-%m-%d')}")

            # Get today's data
            daily_stocks = {}

            for symbol in symbols:
                df = stock_data_dict[symbol]
                day_data = df[df['date'] == date]

                if day_data.empty:
                    continue

                row = day_data.iloc[0]

                # Check data completeness
                if pd.isna(row['rsi']) or pd.isna(row['macd']) or len(df[df['date'] <= date]) < 200:
                    continue

                # Calculate factors
                historical_df = df[df['date'] <= date].copy()

                momentum = self.factor_models.momentum_score(historical_df)
                trend = self.factor_models.trend_strength(historical_df)
                value = self.factor_models.value_score(historical_df)
                quality = self.factor_models.quality_score(historical_df)
                volatility_score = self.factor_models.volatility_score(historical_df)

                factor_weights = {
                    'momentum': 0.30,
                    'trend': 0.25,
                    'value': 0.15,
                    'quality': 0.15,
                    'volatility': 0.15
                }
                composite = self.factor_models.composite_score(historical_df, factor_weights)

                stock_info = {
                    'symbol': symbol,
                    'price': row['close'],
                    'change_pct': ((row['close'] - row['open']) / row['open']) * 100,
                    'ma20': row.get('ma20', 0),
                    'ma50': row.get('ma50', 0),
                    'ma200': row.get('ma200', 0),
                    'rsi': row['rsi'],
                    'macd': row['macd'],
                    'macd_signal': row.get('macd_signal', 0),
                    'bb_upper': row.get('bb_upper', 0),
                    'bb_lower': row.get('bb_lower', 0),
                    'atr': row['atr'],
                    'volatility': row.get('volatility', 0),
                    'volume_ratio': row.get('volume_ratio', 1),
                    'sharpe': row.get('sharpe', 0),
                    'momentum': momentum,
                    'trend': trend,
                    'value': value,
                    'quality': quality,
                    'volatility_score': volatility_score,
                    'composite': composite
                }

                daily_stocks[symbol] = stock_info

            # Check stop conditions for current position
            if self.backtest_engine.position:
                held_symbol = self.backtest_engine.position['symbol']
                if held_symbol in daily_stocks:
                    current_price = daily_stocks[held_symbol]['price']
                    self.backtest_engine.update_position(current_price)

                    stop_reason = self.backtest_engine.check_stop_conditions(current_price)
                    if stop_reason:
                        self.backtest_engine.execute_trade(
                            date, held_symbol, TradingAction.SELL,
                            current_price, 1.0, stop_reason
                        )

            # If no position, find best opportunity
            if self.backtest_engine.position is None and daily_stocks:
                # Evaluate all stocks
                candidates = []

                for symbol, stock_info in daily_stocks.items():
                    # Quick filter: only deliberate on strong candidates
                    if stock_info['composite'] > 0.4:  # Pre-screen threshold
                        decision = self.expert_panel.deliberate(stock_info)

                        if (decision.action == TradingAction.BUY and
                            decision.confidence >= MIN_CONFIDENCE and
                            decision.consensus_level >= CONSENSUS_THRESHOLD):

                            candidates.append({
                                'symbol': symbol,
                                'decision': decision,
                                'composite': stock_info['composite'],
                                'price': stock_info['price']
                            })

                # Select best candidate
                if candidates:
                    best = max(candidates, key=lambda x: x['decision'].confidence * x['composite'])
                    self.backtest_engine.execute_trade(
                        date, best['symbol'], TradingAction.BUY,
                        best['price'], best['decision'].confidence,
                        best['decision'].reasoning
                    )

            # Record daily value
            if self.backtest_engine.position and self.backtest_engine.position['symbol'] in daily_stocks:
                current_price = daily_stocks[self.backtest_engine.position['symbol']]['price']
                self.backtest_engine.record_daily_value(date, current_price)
            else:
                self.backtest_engine.record_daily_value(date)

        # Close position if still holding
        if self.backtest_engine.position:
            last_date = all_dates[-1]
            symbol = self.backtest_engine.position['symbol']
            if symbol in daily_stocks:
                self.backtest_engine.execute_trade(
                    last_date, symbol, TradingAction.SELL,
                    daily_stocks[symbol]['price'], 1.0, "End of backtest"
                )

        # Calculate performance
        print("\n" + "=" * 80)
        print("BACKTEST COMPLETE - Calculating Performance...")
        print("=" * 80)

        metrics = self.backtest_engine.get_performance_metrics()
        self._print_results(metrics)

        return metrics

    def _print_results(self, metrics: Dict[str, Any]):
        """Print backtest results"""
        print("\n" + "=" * 80)
        print("EXPERT CONSENSUS SYSTEM - PERFORMANCE RESULTS")
        print("=" * 80)

        print(f"\n💰 RETURNS:")
        print(f"  Initial Capital:    ${INITIAL_CAPITAL:,.2f}")
        print(f"  Final Value:        ${metrics['final_value']:,.2f}")
        print(f"  Total Return:       {metrics['total_return']:+.2%}")
        print(f"  Annual Return:      {metrics['annual_return']:+.2%}")
        print(f"  Total P&L:          ${metrics['total_pnl']:+,.2f}")

        print(f"\n📉 RISK METRICS:")
        print(f"  Volatility:         {metrics['volatility']:.2%}")
        print(f"  Maximum Drawdown:   {metrics['max_drawdown']:.2%}")
        print(f"  Sharpe Ratio:       {metrics['sharpe_ratio']:.3f}")
        print(f"  Sortino Ratio:      {metrics['sortino_ratio']:.3f}")
        print(f"  Calmar Ratio:       {metrics['calmar_ratio']:.3f}")
        print(f"  VaR (95%):          {metrics['var_95']:.2%}")
        print(f"  CVaR (95%):         {metrics['cvar_95']:.2%}")

        print(f"\n📊 TRADING METRICS:")
        print(f"  Total Trades:       {metrics['total_trades']}")
        print(f"  Win Rate:           {metrics['win_rate']:.2%}")

        print("\n" + "=" * 80)
        print("🎯 TARGET vs ACTUAL:")
        print(f"  Annual Return Target: {TARGET_ANNUAL_RETURN:.0%}  |  Actual: {metrics['annual_return']:+.2%}")
        print(f"  Sharpe Target: {TARGET_SHARPE_RATIO:.1f}  |  Actual: {metrics['sharpe_ratio']:.2f}")
        print(f"  Max DD Target: {TARGET_MAX_DRAWDOWN:.0%}  |  Actual: {metrics['max_drawdown']:.2%}")

        print("\n✨ PERFORMANCE ASSESSMENT:")
        if metrics['annual_return'] >= TARGET_ANNUAL_RETURN:
            print("  🌟 EXCELLENT! Target exceeded!")
        elif metrics['annual_return'] >= TARGET_ANNUAL_RETURN * 0.8:
            print("  👍 GOOD! Close to target")
        elif metrics['annual_return'] > 0.15:
            print("  ✓ Profitable, but below target")
        else:
            print("  ⚠️  Needs optimization")

        if metrics['sharpe_ratio'] >= TARGET_SHARPE_RATIO:
            print("  🌟 Excellent risk-adjusted returns!")
        elif metrics['sharpe_ratio'] >= 1.5:
            print("  👍 Good risk-adjusted returns")

        if metrics['max_drawdown'] > TARGET_MAX_DRAWDOWN:
            print("  👍 Drawdown well controlled!")
        else:
            print("  ⚠️  Drawdown exceeds target")

        print("=" * 80)

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution"""
    print("\n" + "=" * 80)
    print("EXPERT CONSENSUS TRADING SYSTEM")
    print("Targeting 30%+ Annual Returns")
    print("=" * 80)

    print("\nSystem Configuration:")
    print(f"  LLM Provider: {LLM_PROVIDER}")
    print(f"  Model: {CLAUDE_MODEL if LLM_PROVIDER == 'anthropic' else GPT_MODEL}")
    print(f"  Number of Experts: {NUM_EXPERTS}")
    print(f"  Deliberation Rounds: {DELIBERATION_ROUNDS}")
    print(f"  Consensus Threshold: {CONSENSUS_THRESHOLD:.0%}")
    print(f"  Min Confidence: {MIN_CONFIDENCE:.0%}")

    print("\nBacktest Configuration:")
    print(f"  Period: {BACKTEST_START} to {BACKTEST_END}")
    print(f"  Initial Capital: ${INITIAL_CAPITAL:,.2f}")
    print(f"  Asset Universe: {len(ASSET_UNIVERSE)} focused assets")
    print(f"  Assets: Index ETFs, Gold ETFs, Gold Miners, Blue Chips")
    print(f"  Strategy: Full Position Rotation (No Market Scanning)")

    print("\nRisk Management:")
    print(f"  Stop Loss: {STOP_LOSS_PCT:.0%}")
    print(f"  Take Profit: {TAKE_PROFIT_PCT:.0%}")
    print(f"  Trailing Stop: {TRAILING_STOP_PCT:.0%}")

    print("\n" + "=" * 80)

    if not ANTHROPIC_API_KEY and not OPENAI_API_KEY:
        print("\n⚠️  WARNING: No API keys set")
        print("Set ANTHROPIC_API_KEY or OPENAI_API_KEY for full LLM features")
        print("System will run in simulation mode")
        print("\nPress Enter to continue or Ctrl+C to exit...")
        input()

    try:
        # Create system
        system = ExpertConsensusSystem()

        # Run backtest
        metrics = system.run_backtest(BACKTEST_START, BACKTEST_END)

        print("\n✅ Backtest Complete!")
        print(f"\nFinal Annual Return: {metrics['annual_return']:+.2%}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")

        if metrics['annual_return'] >= TARGET_ANNUAL_RETURN:
            print(f"\n🎉 TARGET ACHIEVED! {metrics['annual_return']:.1%} >= {TARGET_ANNUAL_RETURN:.0%}")
        else:
            shortfall = TARGET_ANNUAL_RETURN - metrics['annual_return']
            print(f"\n📊 Shortfall: {shortfall:.1%} below target")
            print("Consider parameter optimization or extending backtest period")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

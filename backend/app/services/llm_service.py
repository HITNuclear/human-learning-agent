from __future__ import annotations
import logging
import re
from typing import AsyncIterator
from openai import AsyncOpenAI
import httpx
from app.config import settings

logger = logging.getLogger(__name__)

_client: AsyncOpenAI | None = None


def get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            timeout=httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=5.0),
        )
    return _client


async def stream_completion(messages: list[dict]) -> AsyncIterator[str]:
    client = get_client()
    logger.info("Calling DeepSeek API, model=%s, messages=%d", settings.deepseek_model, len(messages))
    stream = await client.chat.completions.create(
        model=settings.deepseek_model,
        messages=messages,
        stream=True,
        max_tokens=4096,
    )
    logger.info("DeepSeek stream opened, reading tokens...")
    async for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


async def complete_text(messages: list[dict], max_tokens: int = 2048) -> str:
    client = get_client()
    logger.info("Calling DeepSeek API for completion, model=%s, messages=%d", settings.deepseek_model, len(messages))
    response = await client.chat.completions.create(
        model=settings.deepseek_model,
        messages=messages,
        stream=False,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    cjk_chars = len(re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]", text))
    other_chars = max(0, len(text) - cjk_chars)
    # Heuristic approximation without tokenizer dependency.
    estimated = int(cjk_chars / 1.6 + other_chars / 4.0)
    return max(1, estimated)


def estimate_message_tokens(messages: list[dict]) -> int:
    # Rough chat framing overhead.
    total = 2
    for msg in messages:
        total += 4
        total += estimate_tokens(str(msg.get("content", "")))
    return total


def estimate_cost_usd(input_tokens: int, output_tokens: int) -> float:
    input_cost = (input_tokens / 1_000_000) * settings.deepseek_input_price_per_1m
    output_cost = (output_tokens / 1_000_000) * settings.deepseek_output_price_per_1m
    return round(input_cost + output_cost, 8)

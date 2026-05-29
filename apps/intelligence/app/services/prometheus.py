"""Core Prometheus AI logic — system prompt, streaming chat, and structured analysis."""

import json
import logging
import time
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import Any, Literal, cast

import anthropic
import aiohttp

from app.config import settings
from app.services.conversation_store import store as conversation_store
from app.services.vault_context import VaultContextFetcher

logger = logging.getLogger(__name__)

CHAT_MAX_TOKENS = 1024
ANALYZE_MAX_TOKENS = 800

_CONTEXT_CACHE_TTL = 60  # seconds
_CONTEXT_KEY_PREFIX = "prometheus:ctx:"

_client: anthropic.AsyncAnthropic | None = None
_vault_context_fetcher: VaultContextFetcher | None = None
_redis_client: Any = None
_redis_available: bool = False

# In-process fallback cache: {user_id: (context_dict, expires_at)}
_mem_context_cache: dict[str, tuple[dict[str, Any], float]] = {}


def get_client() -> anthropic.AsyncAnthropic:
    global _client
    if _client is None:
        _client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
    return _client


def get_vault_context_fetcher() -> VaultContextFetcher:
    global _vault_context_fetcher
    if _vault_context_fetcher is None:
        _vault_context_fetcher = VaultContextFetcher(
            api_base_url=settings.nester_api_base_url,
            service_api_key=settings.nester_service_api_key
        )
    return _vault_context_fetcher


def _get_redis() -> Any:
    """Return a shared Redis client, or None if unavailable."""
    global _redis_client, _redis_available
    if _redis_client is not None:
        return _redis_client if _redis_available else None
    try:
        import redis as _redis
        _redis_client = _redis.from_url(settings.redis_url, decode_responses=True)
        _redis_client.ping()
        _redis_available = True
        logger.info("prometheus context cache: redis connected")
    except Exception as exc:
        logger.warning("prometheus context cache: redis unavailable (%s), using in-memory", exc)
        _redis_available = False
    return _redis_client if _redis_available else None


def _cache_get(user_id: str) -> dict[str, Any] | None:
    key = _CONTEXT_KEY_PREFIX + user_id
    r = _get_redis()
    if r is not None:
        try:
            raw = r.get(key)
            if raw:
                return dict(json.loads(raw))
        except Exception as exc:
            logger.warning("context cache redis get failed: %s", exc)
    # In-process fallback
    entry = _mem_context_cache.get(user_id)
    if entry and time.monotonic() < entry[1]:
        return entry[0]
    return None


def _cache_set(user_id: str, context: dict[str, Any]) -> None:
    key = _CONTEXT_KEY_PREFIX + user_id
    r = _get_redis()
    if r is not None:
        try:
            r.setex(key, _CONTEXT_CACHE_TTL, json.dumps(context))
            return
        except Exception as exc:
            logger.warning("context cache redis set failed: %s", exc)
    # In-process fallback
    _mem_context_cache[user_id] = (context, time.monotonic() + _CONTEXT_CACHE_TTL)


async def fetch_user_context(user_id: str, api_base_url: str, service_api_key: str) -> dict[str, Any]:
    """Fetch user vaults, balances, allocations, and recent APY snapshots.

    Returns a dict with 'vaults', 'performance', and 'fetched_at'.
    Raises on network or auth errors so the caller can fall back gracefully.
    """
    headers = {
        "Authorization": f"Bearer {service_api_key}",
        "Content-Type": "application/json",
    }
    base = api_base_url.rstrip("/")

    async with aiohttp.ClientSession() as session:
        # Fetch vaults scoped to this user
        async with session.get(
            f"{base}/api/v1/users/{user_id}/vaults",
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=5),
        ) as resp:
            vaults_data = await resp.json() if resp.status == 200 else {}

        # Fetch recent performance snapshots (best-effort)
        async with session.get(
            f"{base}/api/v1/performance/snapshots",
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=5),
        ) as resp:
            performance_data = await resp.json() if resp.status == 200 else {}

    return {
        "vaults": vaults_data,
        "performance": performance_data,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


async def _get_cached_user_context(user_id: str) -> dict[str, Any] | None:
    """Return cached user context, or fetch and cache it. Returns None on failure."""
    cached = _cache_get(user_id)
    if cached is not None:
        return cached
    try:
        ctx = await fetch_user_context(
            user_id,
            settings.nester_api_base_url,
            settings.nester_service_api_key,
        )
        _cache_set(user_id, ctx)
        return ctx
    except Exception as exc:
        logger.warning("fetch_user_context failed for user %s: %s", user_id, exc)
        return None


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are Prometheus, the financial intelligence layer of Nester.

Nester is a yield-bearing savings platform built on the Stellar blockchain. It lets users in
Africa (primarily Nigeria, Ghana, and Kenya) deposit USDC or XLM into yield-generating vaults,
earn APY, and withdraw earnings directly to their local bank accounts (NGN, GHS, KES).

## Your role
Help users make smart decisions about their Nester vaults, understand their portfolio, and
optimise their yield strategy. You are knowledgeable about:
- DeFi yield strategies on Stellar
- Nester's vault risk tiers: Conservative, Balanced, Growth, DeFi500
- Stellar-native assets: USDC, XLM
- Offramp and settlement flow (crypto to NGN/GHS/KES)
- Savings goals and compounding yield

## Strict scope
You ONLY answer questions about:
- The user's Nester vaults, deposits, and portfolio
- Yield strategies, APYs, and vault risk tiers on Nester
- Savings goals and reaching them with Nester
- How offramp/settlement to local fiat works
- How Nester's contracts, fees, and mechanics work

If asked something outside this scope, respond warmly and briefly. Acknowledge the question,
explain you are focused on Nester savings and yield topics, and invite them to ask anything
about their portfolio, vaults, or yield strategy. Keep it friendly, not robotic.

## Vault tiers (reference)
- Conservative: Stablecoin-only, lowest risk, ~4-6% APY. Good for emergency funds.
- Balanced: Mix of stablecoin and blue-chip DeFi, ~8-12% APY. Good for medium-term goals.
- Growth: Higher-yield DeFi strategies, ~15-25% APY. Suited for long-term horizon with
  risk tolerance.
- DeFi500: Curated top-500 DeFi index exposure, ~20-30% APY. Highest risk, highest reward.

## Formatting rules
- Never use em dashes (the -- or long dash character). Use commas, colons, or plain periods instead.
- Separate distinct thoughts or points into their own paragraphs with a blank line between them.
- Keep responses concise. The user is reading a sidebar panel, not an article.
- Do not use bullet points for simple single-topic answers.

## Tone
Be direct, warm, and specific. Use plain language. When you recommend something, say why in
one sentence. Never start a sentence with an em dash."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_anthropic_messages(history: list[dict[str, str]]) -> list[anthropic.types.MessageParam]:
    """Convert conversation store format to Anthropic message params.

    Conversation store uses {"role": "user"|"assistant", "content": str}.
    Anthropic uses the same role names.
    """
    return [
        {"role": cast(Literal["user", "assistant"], msg["role"]), "content": msg["content"]}
        for msg in history
    ]


# ---------------------------------------------------------------------------
# Streaming chat
# ---------------------------------------------------------------------------

async def stream_chat(user_id: str, message: str) -> AsyncIterator[str]:
    """Yield SSE-formatted data strings for a streaming Claude response.

    Each yielded string is formatted as `data: <text>\\n\\n`.
    A final `data: [DONE]\\n\\n` is yielded when the stream ends.
    """
    # Get conversation history
    history = conversation_store.get(user_id)
    conversation_store.append(user_id, "user", message)

    # Fetch vault context, market rates, and risk data
    vault_context_fetcher = get_vault_context_fetcher()
    vaults = await vault_context_fetcher.fetch_user_vaults(user_id)
    market_rates = await vault_context_fetcher.fetch_market_rates()

    # Fetch risk data for each vault
    risk_data = {}
    for vault in vaults:
        vault_id = vault.get("id")
        if vault_id:
            risk_info = await vault_context_fetcher.fetch_vault_risk(vault_id)
            if risk_info:
                risk_data[vault_id] = risk_info

    context_block = vault_context_fetcher.build_context_block(vaults, market_rates)
    risk_profile_block = vault_context_fetcher.build_risk_profile_block(vaults, risk_data)

    dynamic_system_prompt = f"""You are Prometheus, an AI financial advisor for the
Nester DeFi platform.

{context_block}

{risk_profile_block}

## Nester Platform Context
- Vault types: Flexible (no lock), Fixed-30d (30-day lock, higher APY),
  Fixed-90d (90-day lock, highest APY)
- Rebalancing threshold: triggered when allocation drift exceeds 10%
- Fee structure: 0.5% management fee on yield
- Protocols supported: Aave, Blend, Compound

Provide personalized, data-driven advice based on user positions
and current market conditions. Always cite specific numbers from their
portfolio."""

    # Fetch live portfolio context (60s Redis-backed cache).
    # Injected as a prepended user message so Claude can personalise responses
    # without the instruction appearing in the visible conversation history.
    # If the fetch fails, continue with static knowledge — no error surfaced.
    user_context = await _get_cached_user_context(user_id)
    context_injection: list[dict[str, str]] = []
    if user_context:
        context_injection = [
            {
                "role": "user",
                "content": (
                    "[PORTFOLIO CONTEXT — do not quote this back, use it to personalise your response]\n"
                    + json.dumps(user_context, indent=2)
                ),
            }
        ]

    messages = context_injection + _to_anthropic_messages(history) + [
        {"role": "user", "content": message}
    ]

    client = get_client()
    full_response = ""

    try:
        async with client.messages.stream(
            model=settings.anthropic_model,
            max_tokens=CHAT_MAX_TOKENS,
            system=dynamic_system_prompt,
            messages=messages,
        ) as stream:
            async for text in stream.text_stream:
                full_response += text
                safe = text.replace("\n", "\\n")
                yield f"data: {safe}\n\n"

        conversation_store.append(user_id, "assistant", full_response)
        yield "data: [DONE]\n\n"

    except Exception:
        logger.exception("Anthropic streaming error for user %s", user_id)
        yield "data: Sorry, I had trouble connecting. Please try again.\n\n"
        yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# Structured analysis (non-streaming)
# ---------------------------------------------------------------------------

def _json_strip(raw: str) -> str:
    return raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()


async def get_portfolio_insights(user_id: str) -> list[dict[str, Any]]:
    """Return 2 insight cards for the user's portfolio."""
    schema = (
        '[{"title": str, "body": str, "confidence": float,'
        ' "action": {"label": str, "href": str} | null}]'
    )
    prompt = (
        f"Generate 2 concise portfolio insight cards for a Nester user (id: {user_id}). "
        "Each card should have a short title, a one-sentence body, a confidence score "
        "(0.0–1.0), and optionally an action with a label and href. "
        "Focus on practical savings advice relevant to Nester vaults on Stellar. "
        f"Respond with a JSON array only, no markdown, matching this schema: {schema}"
    )

    client = get_client()
    try:
        response = await client.messages.create(
            model=settings.anthropic_model,
            max_tokens=ANALYZE_MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        text = next(
            (b.text for b in response.content if isinstance(b, anthropic.types.TextBlock)), ""
        )
        return list(json.loads(_json_strip(text)))
    except Exception:
        logger.exception("Failed to get portfolio insights for user %s", user_id)
        return []


async def get_market_sentiment() -> dict[str, Any]:
    """Return a market sentiment summary for the Stellar DeFi / stablecoin space."""
    schema = (
        '{"signal": "bull"|"bear"|"neutral", "summary": str (1 sentence),'
        ' "confidence": float (0.0–1.0), "updatedAt": str (ISO timestamp now)}'
    )
    prompt = (
        "Give a brief market sentiment assessment for the Stellar DeFi and stablecoin "
        "yield space as it relates to Nester users in Africa. "
        f"Respond with JSON only, no markdown, matching this schema: {schema}"
    )

    client = get_client()
    try:
        response = await client.messages.create(
            model=settings.anthropic_model,
            max_tokens=200,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        text = next(
            (b.text for b in response.content if isinstance(b, anthropic.types.TextBlock)), ""
        )
        return dict(json.loads(_json_strip(text)))
    except Exception:
        logger.exception("Failed to get market sentiment")
        return {
            "signal": "neutral",
            "summary": "Sentiment data temporarily unavailable.",
            "confidence": 0.0,
            "updatedAt": "",
        }


async def get_vault_recommendations(vault_id: str) -> dict[str, Any]:
    """Return AI commentary and recommendations for a specific vault."""
    schema = (
        '{"vaultId": str, "commentary": str, "percentileRank": int (0-100),'
        ' "recommendations": [str], "confidence": float}'
    )
    prompt = (
        f"Give an AI commentary and recommendations for Nester vault id '{vault_id}'. "
        "Assume it is a yield-bearing Stellar vault. "
        "Be specific about what type of user this vault suits. "
        f"Respond with JSON only, no markdown, matching this schema: {schema}"
    )

    client = get_client()
    try:
        response = await client.messages.create(
            model=settings.anthropic_model,
            max_tokens=ANALYZE_MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        text = next(
            (b.text for b in response.content if isinstance(b, anthropic.types.TextBlock)), ""
        )
        return dict(json.loads(_json_strip(text)))
    except Exception:
        logger.exception(
            "Failed to get vault recommendations for vault %s", vault_id
        )
        return {
            "vaultId": vault_id,
            "commentary": "Recommendations temporarily unavailable.",
            "percentileRank": 0,
            "recommendations": [],
            "confidence": 0.0,
        }

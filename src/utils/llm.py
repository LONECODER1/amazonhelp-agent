"""
LLM and Embeddings factory for AmazonHelp RAG Agent.

Primary: Groq API for chat models (llama-3.3-70b-versatile, qwen-2.5-32b) - 100% Free & Fast.
Fallback: HuggingFace Inference API.

Embeddings: BAAI/bge-small-en-v1.5 via local sentence-transformers (or HF Inference API).
"""

import os
import logging
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ── API Keys ──────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()

HF_TOKENS_RAW = os.getenv("HF_TOKENS") or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN", "")
HF_TOKENS = [t.strip() for t in HF_TOKENS_RAW.split(",") if t.strip()]
HF_TOKEN = HF_TOKENS[0] if HF_TOKENS else ""

# ── Model names ───────────────────────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

DEFAULT_GROQ_CHAT = "qwen/qwen3.8-27b"
DEFAULT_GROQ_JUDGE = "qwen/qwen3.8-27b"

CHAT_MODEL = os.getenv("CHAT_MODEL", DEFAULT_GROQ_CHAT)
JUDGE_MODEL = os.getenv("JUDGE_MODEL", DEFAULT_GROQ_JUDGE)


def _groq_llm(model: str, temperature: float, max_tokens: int):
    """Create a Groq Chat LLM using langchain_groq.ChatGroq."""
    from langchain_groq import ChatGroq

    # Cap max_tokens safely below Groq's 1000 OTPM limit on free tier
    safe_max_tokens = min(max_tokens, 800) if max_tokens else 800

    # Map model names to available Groq models
    if "qwen" in model.lower():
        groq_model = "qwen/qwen3.8-27b"
    elif "gpt" in model.lower():
        groq_model = "openai/gpt-oss-120b"
    elif model in ("qwen/qwen3.8-27b", "openai/gpt-oss-120b", "openai/gpt-oss-20b"):
        groq_model = model
    else:
        groq_model = DEFAULT_GROQ_CHAT

    logger.info("Using Groq API for model: %s (input: %s, max_tokens: %d)", groq_model, model, safe_max_tokens)
    return ChatGroq(
        model=groq_model,
        temperature=temperature,
        max_tokens=safe_max_tokens,
        groq_api_key=GROQ_API_KEY,
    )


def _hf_llm(model: str, temperature: float, max_tokens: int, token_index: int = 0):
    """Create a HuggingFace Inference API Chat LLM as fallback."""
    from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

    token = HF_TOKENS[token_index % len(HF_TOKENS)] if HF_TOKENS else HF_TOKEN
    logger.info("Using HuggingFace Inference API for model: %s", model)

    endpoint = HuggingFaceEndpoint(
        repo_id=model,
        task="text-generation",
        max_new_tokens=max_tokens,
        temperature=temperature,
        huggingfacehub_api_token=token,
        do_sample=temperature > 0,
    )
    return ChatHuggingFace(llm=endpoint)


def get_chat_llm(model: str = None, temperature: float = 0.3, max_tokens: int = 1024):
    """
    Return chat LLM:
    Uses Groq (free, fast) if GROQ_API_KEY is configured, else falls back to HuggingFace.
    """
    target_model = model or CHAT_MODEL
    try:
        if GROQ_API_KEY:
            return _groq_llm(target_model, temperature, max_tokens)
        return _hf_llm(target_model, temperature, max_tokens)
    except Exception as e:
        logger.error("get_chat_llm failed for model %s: %s", target_model, e)
        raise


def get_judge_llm(model: str = None, temperature: float = 0.1, max_tokens: int = 1024):
    """
    Return judge LLM:
    Uses Groq (free, fast) if GROQ_API_KEY is configured, else falls back to HuggingFace.
    """
    target_model = model or JUDGE_MODEL
    try:
        if GROQ_API_KEY:
            return _groq_llm(target_model, temperature, max_tokens)
        return _hf_llm(target_model, temperature, max_tokens)
    except Exception as e:
        logger.error("get_judge_llm failed for model %s: %s", target_model, e)
        raise


@lru_cache(maxsize=1)
def get_embeddings():
    """Return embeddings for BAAI/bge-small-en-v1.5 (local sentence-transformers or HF API fallback)."""
    logger.info("Loading embeddings: %s", EMBEDDING_MODEL)
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    except Exception as e:
        logger.info("Local HuggingFaceEmbeddings unavailable (%s), trying Inference API", e)
        from langchain_huggingface import HuggingFaceEndpointEmbeddings

        if not HF_TOKEN:
            raise ValueError(
                "HF_TOKEN is required for embeddings fallback. Set it in your .env file."
            )

        return HuggingFaceEndpointEmbeddings(
            huggingfacehub_api_token=HF_TOKEN,
            model=EMBEDDING_MODEL,
        )

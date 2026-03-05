"""LLM adapter supporting multiple backends with task-aware routing, caching, and rate limiting."""
import hashlib
import logging
import time
from typing import List, Optional

from config import get_llm_config, get_task_profile

logger = logging.getLogger(__name__)

# ─── Global rate-limit state (shared across ALL SimpleLLM instances) ─────────
# Prevents overwhelming API rate limits when many instances are created in bulk.
_global_last_request_time: float = 0.0

# ─── Simple response cache ──────────────────────────────────────────────────
# Keyed by (model, temperature, prompt_hash).  Avoids redundant LLM calls on
# retrain or repeated entity lookups.  Lives in-process only (no disk).
_response_cache: dict[str, str] = {}
_CACHE_MAX = 256


def _cache_key(model: str, temperature: float, prompt: str) -> str:
    h = hashlib.sha256(prompt.encode("utf-8", errors="replace")).hexdigest()[:16]
    return f"{model}|{temperature}|{h}"


def _cache_get(key: str) -> Optional[str]:
    return _response_cache.get(key)


def _cache_put(key: str, value: str) -> None:
    if len(_response_cache) >= _CACHE_MAX:
        # Evict oldest half (dict is insertion-ordered in Python 3.7+)
        keys = list(_response_cache.keys())
        for k in keys[: _CACHE_MAX // 2]:
            _response_cache.pop(k, None)
    _response_cache[key] = value


def _join_messages(messages: List[object]) -> str:
    """Convert message objects to a single prompt string."""
    parts = []
    for m in messages:
        content = getattr(m, "content", str(m))
        parts.append(content)
    return "\n\n".join(parts)


def _build_openai_messages(messages: List[object]) -> list[dict]:
    """Convert LangChain-style message objects to OpenAI chat format."""
    openai_messages = []
    for m in messages:
        msg_type = getattr(m, "type", "human")
        role = {"system": "system", "human": "user", "ai": "assistant"}.get(msg_type, "user")
        content = getattr(m, "content", str(m))
        openai_messages.append({"role": role, "content": content})
    return openai_messages or [{"role": "user", "content": _join_messages(messages)}]


class SimpleLLM:
    """Unified LLM interface for different providers with task-aware settings."""

    def __init__(self, backend: str, key_or_path: str, model: str, task: str = "default"):
        self.backend = backend
        self.key = key_or_path
        self.model = model
        self.task = task
        self._client = None
        self._last_request_time = 0
        self._min_request_interval = 0.5

        # Task-aware defaults from config
        profile = get_task_profile(task)
        self._default_temperature = profile["temperature"]
        self._default_max_tokens = profile["max_tokens"]

        # Initialize specific backend clients
        if backend == "local_transformers":
            try:
                from transformers import pipeline
                self._client = pipeline("text2text-generation", model=model)
            except Exception:
                self._client = None
        elif backend == "local":
            try:
                from llama_cpp import Llama
                self._client = Llama(
                    model_path=key_or_path,
                    chat_format="chatml",
                    verbose=False,
                    n_ctx=8192,  # Increased from 4096 — enriched prompts are larger now
                    n_gpu_layers=-1,
                )
            except Exception:
                self._client = None
        elif backend in ["anthropic", "fireworks", "groq", "qwen_local"]:
            self._client = None
        elif backend == "huggingface":
            try:
                from huggingface_hub import InferenceClient
                self._client = InferenceClient(token=key_or_path)
            except Exception:
                self._client = None

    def invoke(self, messages: List[object], max_tokens: int = 0, temperature: float = -1.0) -> str:
        """Generate text from messages using the configured backend.

        Parameters:
            max_tokens:  0 → use task default from config.
            temperature: <0 → use task default from config.
        """
        effective_max_tokens = max_tokens if max_tokens > 0 else self._default_max_tokens
        effective_temperature = temperature if temperature >= 0 else self._default_temperature

        prompt = _join_messages(messages)

        # ── Check cache (skip for creative tasks) ───────────────────────
        use_cache = effective_temperature <= 0.2
        ck = ""
        if use_cache:
            ck = _cache_key(self.model, effective_temperature, prompt)
            cached = _cache_get(ck)
            if cached is not None:
                logger.debug("Cache hit for task=%s model=%s", self.task, self.model)
                return cached

        result = self._invoke_backend(messages, effective_max_tokens, effective_temperature)

        if use_cache and result:
            _cache_put(ck, result)

        return result

    def _invoke_backend(self, messages: List[object], max_tokens: int, temperature: float) -> str:
        """Dispatch to the actual backend."""

        # Local Transformers
        if self.backend == "local_transformers" and self._client is not None:
            try:
                prompt = _join_messages(messages)
                out = self._client(prompt, max_new_tokens=max_tokens)
                if isinstance(out, list) and len(out) > 0 and "generated_text" in out[0]:
                    return out[0]["generated_text"].strip()
                if isinstance(out, dict) and "generated_text" in out:
                    return out["generated_text"].strip()
                return str(out)
            except Exception:
                return ""

        # Local llama-cpp
        if self.backend == "local" and self._client is not None:
            try:
                chat_messages = _build_openai_messages(messages)
                resp = self._client.create_chat_completion(
                    messages=chat_messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                if isinstance(resp, dict) and "choices" in resp and len(resp["choices"]) > 0:
                    choice = resp["choices"][0]
                    text = choice.get("message", {}).get("content") or choice.get("text", "")
                    return (text or "").strip()
                return ""
            except Exception as e:
                logger.warning("Local LLM exception: %s", e)
                raise

        # HuggingFace Inference
        if self.backend == "huggingface" and self._client is not None:
            try:
                prompt = _join_messages(messages)
                out = self._client.text_generation(model=self.model, inputs=prompt, max_new_tokens=max_tokens)
                if isinstance(out, list) and len(out) > 0 and "generated_text" in out[0]:
                    return out[0]["generated_text"].strip()
                if isinstance(out, dict) and "generated_text" in out:
                    return out["generated_text"].strip()
                return str(out)
            except Exception:
                return ""

        # Fireworks.ai
        if self.backend == "fireworks":
            try:
                import requests
                headers = {
                    "Authorization": f"Bearer {self.key}",
                    "Content-Type": "application/json",
                }
                data = {
                    "model": self.model,
                    "messages": _build_openai_messages(messages),
                    "temperature": temperature,
                    "max_tokens": min(1024, max_tokens),
                }
                response = requests.post(
                    "https://api.fireworks.ai/inference/v1/chat/completions",
                    json=data, headers=headers, timeout=60,
                )
                if response.status_code == 200:
                    result = response.json()
                    if "choices" in result and len(result["choices"]) > 0:
                        return (result["choices"][0].get("message", {}).get("content", "") or "").strip()
                else:
                    logger.warning("Fireworks error %s: %s", response.status_code, response.text)
                return ""
            except Exception as e:
                logger.warning("Fireworks exception: %s", e)
                return ""

        # Local Qwen via OpenAI-compatible HTTP API
        if self.backend == "qwen_local":
            try:
                import requests
                self._rate_limit_wait()
                data = {
                    "model": self.model,
                    "messages": _build_openai_messages(messages),
                    "temperature": temperature,
                    "max_tokens": min(1024, max_tokens),
                }
                url = f"{self.key}/v1/chat/completions"
                response = requests.post(url, json=data, headers={"Content-Type": "application/json"}, timeout=120)
                self._last_request_time = time.time()
                if response.status_code == 200:
                    result = response.json()
                    if "choices" in result and len(result["choices"]) > 0:
                        return (result["choices"][0].get("message", {}).get("content", "") or "").strip()
                else:
                    logger.warning("Local LLM error %s: %s", response.status_code, response.text)
                return ""
            except Exception as e:
                logger.warning("Local LLM exception: %s", e)
                return ""

        # Groq
        if self.backend == "groq":
            return self._invoke_groq(messages, max_tokens, temperature)

        return ""

    def _rate_limit_wait(self) -> None:
        global _global_last_request_time
        elapsed = time.time() - _global_last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        _global_last_request_time = time.time()

    def _invoke_groq(self, messages: List[object], max_tokens: int, temperature: float) -> str:
        """Groq-specific invocation with retry / rate-limit handling."""
        import re
        import requests

        headers = {
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
        }
        data = {
            "model": self.model,
            "messages": _build_openai_messages(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        max_retries = 5
        backoff_seconds = 1.0

        for attempt in range(max_retries):
            self._rate_limit_wait()

            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                json=data, headers=headers, timeout=60,
            )
            global _global_last_request_time
            _global_last_request_time = time.time()

            if response.status_code == 200:
                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    return (result["choices"][0].get("message", {}).get("content", "") or "").strip()
                logger.warning("Unexpected Groq response: %s", result)
                return ""

            if response.status_code == 429:
                try:
                    err = response.json().get("error", {})
                    msg = err.get("message", "")
                except Exception:
                    msg = response.text or ""

                logger.warning("Groq 429 (attempt %d): %s", attempt + 1, msg)

                wait = backoff_seconds
                m_ms = re.search(r"try again in ([0-9.]+)ms", msg)
                m_s = re.search(r"try again in ([0-9.]+)s", msg)
                if m_ms:
                    wait = max(wait, float(m_ms.group(1)) / 1000.0)
                elif m_s:
                    wait = max(wait, float(m_s.group(1)))
                time.sleep(wait)
                backoff_seconds *= 2
                continue

            logger.warning("Groq error %s: %s", response.status_code, response.text)
            return ""

        return ""

    def __or__(self, parser):
        """Support composition with parsers via | operator."""
        return CombinedChain(self, parser)


class CombinedChain:
    """Chain combining SimpleLLM with a parser."""

    def __init__(self, llm: SimpleLLM, parser):
        self.llm = llm
        self.parser = parser

    def invoke(self, messages: List[object]):
        """Generate text and parse result."""
        text = self.llm.invoke(messages)
        try:
            parse_fn = getattr(self.parser, "parse", None)
            if callable(parse_fn):
                return parse_fn(text)
        except Exception:
            pass
        return text


def get_llm(task: str = "default"):
    """Get the configured LLM adapter, optionally routed by task.

    Task names: categorize, entity_extract, summarize, flag_urgent, meeting_extract,
                draft, plan_reply, decide, style_learn, default.
    """
    provider, key_or_path, model = get_llm_config(task=task)
    return SimpleLLM(provider, key_or_path, model, task=task)

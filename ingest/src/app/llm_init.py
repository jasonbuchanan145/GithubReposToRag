from __future__ import annotations
import json
import logging
import os
import re
from typing import Any

# Fix the import - use the correct module path
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.core.llms import CustomLLM, CompletionResponse, LLMMetadata
import requests
from llama_index.core import Settings


from app import config

# -----------------------
# Config (via env)
# -----------------------
QWEN_BASE_URL = os.getenv("QWEN_BASE_URL", "").rstrip("/")
USE_CHAT_ENDPOINT = os.getenv("QWEN_USE_CHAT", "true").lower() == "true"
ALLOW_THINKING = os.getenv("INGEST_ALLOW_THINKING", "true").lower() == "true"
FINAL_ONLY_INSTRUCTION = os.getenv("INGEST_FINAL_ONLY", "true").lower() == "true"
LOG_RAW = os.getenv("QWEN_LOG_RAW", "false").lower() == "true"


def _final_only_system_msg() -> str:
    if not FINAL_ONLY_INSTRUCTION:
        return ""
    return (
        "You are a metadata writer for an indexing pipeline. "
        "Return ONLY the final answer requested by the prompt. "
        "Do not include internal reasoning, prefaces, apologies, or meta-commentary. "
        "No headings, no role tags. Output just the final text."
    )


_COT_PATTERNS = [
    r"(?is)<think>.*?</think>",                       # Qwen/DeepSeek-style tags
    r"(?im)^(assistant|system|user)\s*:\s*\d*\s*",    # stray role markers
    r"(?im)^(okay|alright|let me|i need to|thinking|hmm)[^\n]*\n",  # chatty lines
]

def _sanitize(text: str) -> str:
    s = text or ""
    for pat in _COT_PATTERNS:
        s = re.sub(pat, "", s)
    # Strip common prefixes
    s = re.sub(r"(?i)^\s*(final answer|summary)\s*:\s*", "", s.strip())
    return s.strip()


class QwenLLM(CustomLLM):
    """Custom LLM implementation for Qwen model in ingest service (soft suppression)."""

    model_name: str = os.getenv("QWEN_MODEL", "Qwen/Qwen3-4B-FP8")
    context_window: int = 11712
    num_output: int = 1024

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name,
            is_chat_model=True,
            is_function_calling_model=False,
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """Complete the prompt with Qwen model."""
        logging.debug(f"QwenLLM.complete called with prompt length: {len(prompt)}")
        try:
            # Prefer chat endpoint to access Qwen chat template kwargs.
            if USE_CHAT_ENDPOINT:
                response_text = self._call_qwen_chat_api(prompt, **kwargs)
            else:
                response_text = self._call_qwen_completion_api(prompt, **kwargs)
            return CompletionResponse(text=response_text)
        except Exception as e:
            logging.error(f"QwenLLM.complete failed: {str(e)}")
            return CompletionResponse(text=f"Error: {str(e)}")

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any):
        """Simple streaming facade (calls non-stream complete)."""
        resp = self.complete(prompt, **kwargs)
        yield resp

    def _base_url(self) -> str:
        try:
            if getattr(config.SETTINGS, "qwen_endpoint", None):
                return config.SETTINGS.qwen_endpoint.rstrip("/")
        except Exception:
            raise RuntimeError("SETTINGS.qwen_endpoint not set.")
        if not QWEN_BASE_URL:
            raise RuntimeError("QWEN_BASE_URL or SETTINGS.qwen_endpoint must be set.")
        return QWEN_BASE_URL

    def _call_qwen_chat_api(self, prompt: str, **kwargs) -> str:
        url = f"{self._base_url()}/v1/chat/completions"

        messages = []
        sys = _final_only_system_msg()
        if sys:
            messages.append({"role": "system", "content": sys})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self.num_output),
            "temperature": kwargs.get("temperature", 0.2 if FINAL_ONLY_INSTRUCTION else 0.7),
            "top_p": kwargs.get("top_p", 0.9),
        }

        # Qwen-specific soft control: allow thinking (default), but don't force-disable it.
        # If you *do* want to hard-disable (efficiency over accuracy), set INGEST_ALLOW_THINKING=false.
        if not ALLOW_THINKING:
            payload["chat_template_kwargs"] = {"enable_thinking": False}

        headers = {"Content-Type": "application/json"}
        logging.debug(f"POST {url}")
        if LOG_RAW:
            logging.debug("➡️ Payload: %s", json.dumps(payload)[:2000])

        r = requests.post(url, json=payload, headers=headers, timeout=60)
        if r.status_code != 200:
            msg = f"API Error {r.status_code}: {r.text}"
            logging.warning(msg)
            return msg

        data = r.json()
        if LOG_RAW:
            logging.debug("Raw: %s", json.dumps(data)[:2000])

        try:
            choice = (data.get("choices") or [])[0]
            msg = choice.get("message") or {}
            # If the server exposes reasoning separately, ignore it (soft suppression).
            content = msg.get("content") or choice.get("text") or ""
        except Exception:
            raise RuntimeError("Error parsing Qwen chat response.")

        return _sanitize(content) or "No response generated"

    def _call_qwen_completion_api(self, prompt: str, **kwargs) -> str:
        """Fallback: classic /v1/completions (no chat_template control)."""
        url = f"{self._base_url()}/v1/completions"
        payload = {
            "model": self.model_name,
            "prompt": self._maybe_inject_instruction(prompt),
            "max_tokens": kwargs.get("max_tokens", self.num_output),
            "temperature": kwargs.get("temperature", 0.2 if FINAL_ONLY_INSTRUCTION else 0.7),
            "top_p": kwargs.get("top_p", 0.9),
        }

        headers = {"Content-Type": "application/json"}
        logging.debug(f"🔗 POST {url}")
        if LOG_RAW:
            logging.debug("➡️ Payload: %s", json.dumps(payload)[:2000])

        r = requests.post(url, json=payload, headers=headers, timeout=60)
        if r.status_code != 200:
            msg = f"API Error {r.status_code}: {r.text}"
            logging.warning(msg)
            return msg

        data = r.json()
        if LOG_RAW:
            logging.debug("⬅️ Raw: %s", json.dumps(data)[:2000])

        try:
            text = (data.get("choices") or [])[0].get("text", "")
        except Exception:
            text = ""

        return _sanitize(text) or "No response generated"

    # When using /v1/completions, we can still bias the model away from prefaces:
    def _maybe_inject_instruction(self, prompt: str) -> str:
        if not FINAL_ONLY_INSTRUCTION:
            return prompt
        return (
                "Return ONLY the final answer requested. "
                "No internal reasoning, no preface, no meta commentary.\n\n"
                + prompt
        )


def initialize_llm_settings():
    """Initialize LlamaIndex with custom LLM and embedding model."""
    logging.info(f"🔧 Initializing LLM settings for ingest service...")
    try:
        logging.info(f"🔗 Qwen endpoint: {getattr(config.SETTINGS.qwen_endpoint, 'qwen_endpoint', QWEN_BASE_URL)}")
        logging.info(f"📝 Embedding model: {config.SETTINGS.embed_model}")
        embed_model = HuggingFaceEmbedding(model_name=config.SETTINGS.embed_model)
    except Exception:
        embed_model = None
        logging.warning("Embedding model not initialized here (using existing global config).")

    Settings.llm = QwenLLM()
    if embed_model is not None:
        Settings.embed_model = embed_model
    logging.info("✅ LLM settings initialized successfully")

import logging
import requests
from typing import Any
from llama_index.core.llms import CustomLLM, CompletionResponse, LLMMetadata
from llama_index.core.llms.callbacks import llm_completion_callback
from rag_shared.config import QWEN_ENDPOINT, QWEN_MODEL, QWEN_MAX_OUTPUT

class QwenLLM(CustomLLM):
    model_name: str = QWEN_MODEL
    context_window: int = 11712
    num_output: int = QWEN_MAX_OUTPUT

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
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "max_tokens": kwargs.get("max_tokens", self.num_output),
                "temperature": kwargs.get("temperature"),
                "top_p": kwargs.get("top_p"),
            }
            resp = requests.post(f"{QWEN_ENDPOINT}/v1/completions", json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            text = (data.get("choices") or [{}])[0].get("text", "")
            return CompletionResponse(text=text or "")
        except Exception as e:
            logging.exception("QwenLLM.complete failed")
            return CompletionResponse(text=f"Error: {e}")

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any):
        yield self.complete(prompt, **kwargs)
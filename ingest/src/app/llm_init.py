"""Initialize LlamaIndex LLM settings for ingest service."""
import logging
from typing import Any

import requests
from llama_index.core import Settings
from llama_index.core.llms import CustomLLM, CompletionResponse, LLMMetadata
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from app.config import SETTINGS


class QwenLLM(CustomLLM):
    """Custom LLM implementation for Qwen model in ingest service."""

    model_name: str = "Qwen/Qwen3-4B-FP8"
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
        logging.debug(f"🤖 QwenLLM.complete called with prompt length: {len(prompt)}")
        try:
            response_text = self._call_qwen_api(prompt, **kwargs)
            return CompletionResponse(text=response_text)
        except Exception as e:
            logging.error(f"🤖 QwenLLM.complete failed: {str(e)}")
            return CompletionResponse(text=f"Error: {str(e)}")

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any):
        """Stream completion (optional but recommended)."""
        response = self.complete(prompt, **kwargs)
        yield response

    def _call_qwen_api(self, prompt: str, **kwargs) -> str:
        """Make API call to Qwen service using vLLM OpenAI API."""
        logging.debug(f"🔗 Making API call to Qwen at {SETTINGS.qwen_endpoint}")
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "max_tokens": kwargs.get("max_tokens", self.num_output),
                "temperature": kwargs.get("temperature", 0.7),
                "top_p": kwargs.get("top_p", 0.9),
            }

            response = requests.post(
                f"{SETTINGS.qwen_endpoint}/v1/completions",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=60
            )

            if response.status_code == 200:
                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    return result["choices"][0]["text"]
                else:
                    return "No response generated"
            else:
                error_msg = f"API Error {response.status_code}: {response.text}"
                logging.warning(error_msg)
                return error_msg

        except requests.RequestException as e:
            error_msg = f"Connection Error: {str(e)}"
            logging.error(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"Unexpected Error: {str(e)}"
            logging.error(error_msg)
            return error_msg


def initialize_llm_settings():
    """Initialize LlamaIndex with custom LLM and embedding model."""
    logging.info(f"🔧 Initializing LLM settings for ingest service...")
    logging.info(f"🔗 Qwen endpoint: {SETTINGS.qwen_endpoint}")
    logging.info(f"📝 Embedding model: {SETTINGS.embed_model}")

    # Load the embedding model
    embed_model = HuggingFaceEmbedding(model_name=SETTINGS.embed_model)

    # Configure global settings
    Settings.llm = QwenLLM()
    Settings.embed_model = embed_model

    logging.info("✅ LLM settings initialized successfully")

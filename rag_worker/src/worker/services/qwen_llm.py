import logging
import requests
import re
import json
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


    def _clean_response(self, text: str) -> str:
        """Remove markdown code block formatting from LLM responses."""
        if not text:
            return text
        # Remove markdown code blocks (```json, ```, etc.)
        cleaned = re.sub(r'```(?:json|yaml)?\s*\n', '', text)
        cleaned = re.sub(r'\n```\s*$', '', cleaned)
        cleaned = re.sub(r'```\s*$', '', cleaned)

        # Also handle cases where there are multiple code blocks
        cleaned = re.sub(r'```(?:json|yaml)?\s*', '', cleaned)
        cleaned = re.sub(r'```', '', cleaned)

        return cleaned.strip()

    def _is_selector_prompt(self, prompt: str) -> bool:
        """Check if this is a router selector prompt that needs JSON response."""
        selector_indicators = [
            "choice 1:",
            "choice 2:",
            "choice 3:",
            "choice 4:",
            "select one of the following",
            "choose from the following options",
            "pick the best option"
        ]
        return any(indicator in prompt.lower() for indicator in selector_indicators)

    def _clean_selector_response(self, text: str) -> str:
        """Clean malformed JSON responses for router selection."""
        if not text:
            return "1"

        # Remove repetitive patterns first
        lines = text.split('\n')
        unique_lines = []
        seen = set()
        for line in lines:
            if line.strip() and line.strip() not in seen:
                unique_lines.append(line.strip())
                seen.add(line.strip())
                if len(unique_lines) >= 3:  # Limit to prevent excessive content
                    break

        cleaned_text = '\n'.join(unique_lines)

        # Try to extract valid JSON from the response
        import json

        # Look for JSON objects with "choice" field
        json_pattern = r'\{"choice":\s*(\d+)(?:,\s*"reason":[^}]*)?\}'
        matches = re.findall(json_pattern, cleaned_text)
        if matches:
            # Return the first valid choice as a simple number
            return matches[0]

        # Look for simple number patterns
        number_pattern = r'\b([1-4])\b'
        numbers = re.findall(number_pattern, cleaned_text)
        if numbers:
            return numbers[0]

        # Fallback: try to parse as JSON and extract choice
        try:
            # Clean up malformed JSON brackets
            json_cleaned = re.sub(r']}]+$', ']', cleaned_text)
            json_cleaned = re.sub(r'^[{\[]+', '[', json_cleaned)

            parsed = json.loads(json_cleaned)
            if isinstance(parsed, list) and len(parsed) > 0:
                if isinstance(parsed[0], dict) and "choice" in parsed[0]:
                    return str(parsed[0]["choice"])
        except:
            pass

        # Final fallback
        return "1"

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        try:
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_completion_tokens": kwargs.get("max_tokens", self.num_output),
                "temperature": kwargs.get("temperature", 0.4),  # Lower temperature for more focused responses
                "top_p": kwargs.get("top_p", 0.8),  # More focused token selection
                "repetition_penalty": kwargs.get("repetition_penalty", 1.2),  # Penalize repetition
            }

            # Log the request payload
            logging.info(f"Qwen LLM Request Payload: {json.dumps(payload, indent=2)}")

            resp = requests.post(f"{QWEN_ENDPOINT}/v1/chat/completions", json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()

            # Log the full response
            logging.info(f"Qwen LLM Full Response: {json.dumps(data, indent=2)}")

            text = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")

            # Check for tool calls
            choices = data.get("choices", [])
            if choices:
                choice = choices[0]
                message = choice.get("message", {})
                tool_calls = message.get("tool_calls", [])
                if tool_calls:
                    logging.info(f"Qwen LLM Tool Calls Found: {json.dumps(tool_calls, indent=2)}")

            # Check if this is a selector prompt and handle accordingly
            if self._is_selector_prompt(prompt):
                cleaned_text = self._clean_selector_response(text)
                logging.info(f"Qwen LLM selector response: {cleaned_text}")
            else:
                cleaned_text = text
                logging.info(f"Qwen LLM response: {cleaned_text}")

            return CompletionResponse(text=cleaned_text or "")
        except Exception as e:
                logging.exception("QwenLLM.complete failed")
                return CompletionResponse(text=f"Error: {e}")
    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any):
        yield self.complete(prompt, **kwargs)
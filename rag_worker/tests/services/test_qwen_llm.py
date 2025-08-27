"""
Unit tests for QwenLLM.
"""
import pytest
import json
from unittest.mock import MagicMock, patch
from llama_index.core.llms import CompletionResponse
from worker.services.qwen_llm import QwenLLM


class TestQwenLLM:
    """Tests for QwenLLM."""

    def test_metadata(self):
        """Test LLM metadata properties."""
        llm = QwenLLM()
        metadata = llm.metadata

        assert metadata.context_window == 11712
        assert metadata.num_output > 0
        assert metadata.is_chat_model is True
        assert metadata.is_function_calling_model is False

    def test_is_selector_prompt(self):
        """Test selector prompt detection."""
        llm = QwenLLM()

        # Test positive cases
        assert llm._is_selector_prompt("Please select one of the following options: choice 1: code")
        assert llm._is_selector_prompt("Choose from the following options: choice 2: package")
        assert llm._is_selector_prompt("Pick the best option choice 3: project")

        # Test negative cases
        assert not llm._is_selector_prompt("What is the weather today?")
        assert not llm._is_selector_prompt("Explain this code snippet")

    def test_clean_selector_response_json(self):
        """Test cleaning selector response with valid JSON."""
        llm = QwenLLM()

        # Test valid JSON with choice
        text = '{"choice": 2, "reason": "Best match"}'
        cleaned = llm._clean_selector_response(text)
        assert cleaned == "2"

        # Test JSON without reason
        text = '{"choice": 1}'
        cleaned = llm._clean_selector_response(text)
        assert cleaned == "1"

    def test_clean_selector_response_number_pattern(self):
        """Test cleaning selector response with number patterns."""
        llm = QwenLLM()

        # Test simple number
        text = "The answer is 3"
        cleaned = llm._clean_selector_response(text)
        assert cleaned == "3"

        # Test multiple numbers (should return first valid)
        text = "Options 2 and 4 are good, but 2 is better"
        cleaned = llm._clean_selector_response(text)
        assert cleaned == "2"

    def test_clean_selector_response_empty(self):
        """Test cleaning empty selector response."""
        llm = QwenLLM()

        # Test empty string
        cleaned = llm._clean_selector_response("")
        assert cleaned == "1"

        # Test None
        cleaned = llm._clean_selector_response(None)
        assert cleaned == "1"

    def test_clean_selector_response_malformed_json(self):
        """Test cleaning malformed JSON selector response."""
        llm = QwenLLM()

        # Test malformed JSON that can be fixed
        text = '[{"choice": 2}]'
        cleaned = llm._clean_selector_response(text)
        assert cleaned == "2"

        # Test completely invalid JSON
        text = "This is not JSON at all"
        cleaned = llm._clean_selector_response(text)
        assert cleaned == "1"

    @patch('worker.services.qwen_llm.requests')
    def test_complete_success(self, mock_requests):
        """Test successful completion request."""
        llm = QwenLLM()

        # Mock successful response
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": "Test response from LLM"
                }
            }]
        }
        mock_requests.post.return_value = mock_response

        result = llm.complete("Test prompt")

        assert isinstance(result, CompletionResponse)
        assert result.text == "Test response from LLM"

        # Verify request was made correctly
        mock_requests.post.assert_called_once()
        call_args = mock_requests.post.call_args
        assert "chat/completions" in call_args[0][0]
        assert call_args[1]["timeout"] == 60

    @patch('worker.services.qwen_llm.requests')
    def test_complete_selector_prompt(self, mock_requests):
        """Test completion with selector prompt."""
        llm = QwenLLM()

        # Mock response with JSON choice
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": '{"choice": 3, "reason": "best match"}'
                }
            }]
        }
        mock_requests.post.return_value = mock_response

        result = llm.complete("Select one: choice 1: code, choice 2: package")

        assert isinstance(result, CompletionResponse)
        assert result.text == "3"

    @patch('worker.services.qwen_llm.requests')
    def test_complete_with_tool_calls(self, mock_requests):
        """Test completion response with tool calls."""
        llm = QwenLLM()

        # Mock response with tool calls
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": "Response with tools",
                    "tool_calls": [{"function": {"name": "test_tool"}}]
                }
            }]
        }
        mock_requests.post.return_value = mock_response

        with patch('worker.services.qwen_llm.logging') as mock_logging:
            result = llm.complete("Test prompt")

            assert isinstance(result, CompletionResponse)
            assert result.text == "Response with tools"
            # Verify tool calls were logged
            mock_logging.info.assert_called()

    @patch('worker.services.qwen_llm.requests')
    def test_complete_request_failure(self, mock_requests):
        """Test completion with request failure."""
        llm = QwenLLM()

        # Mock request exception
        mock_requests.post.side_effect = Exception("Network error")

        with patch('worker.services.qwen_llm.logging') as mock_logging:
            result = llm.complete("Test prompt")

            assert isinstance(result, CompletionResponse)
            assert "Error: Network error" in result.text
            mock_logging.exception.assert_called_once()

    @patch('worker.services.qwen_llm.requests')
    def test_complete_empty_response(self, mock_requests):
        """Test completion with empty response."""
        llm = QwenLLM()

        # Mock response with empty content
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "choices": [{"message": {"content": ""}}]
        }
        mock_requests.post.return_value = mock_response

        result = llm.complete("Test prompt")

        assert isinstance(result, CompletionResponse)
        assert result.text == ""

    @patch('worker.services.qwen_llm.requests')
    def test_complete_malformed_response(self, mock_requests):
        """Test completion with malformed response structure."""
        llm = QwenLLM()

        # Mock response with missing structure
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"choices": [{}]}
        mock_requests.post.return_value = mock_response

        result = llm.complete("Test prompt")

        assert isinstance(result, CompletionResponse)
        assert result.text == ""

    @patch('worker.services.qwen_llm.requests')
    def test_complete_custom_params(self, mock_requests):
        """Test completion with custom parameters."""
        llm = QwenLLM()

        # Mock successful response
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Custom response"}}]
        }
        mock_requests.post.return_value = mock_response

        result = llm.complete(
            "Test prompt",
            max_tokens=512,
            temperature=0.8,
            top_p=0.9,
            repetition_penalty=1.1
        )

        assert isinstance(result, CompletionResponse)
        assert result.text == "Custom response"

        # Verify custom parameters were passed
        call_args = mock_requests.post.call_args
        payload = call_args[1]["json"]
        assert payload["max_completion_tokens"] == 512
        assert payload["temperature"] == 0.8
        assert payload["top_p"] == 0.9
        assert payload["repetition_penalty"] == 1.1

def test_stream_complete():
    """Test stream_complete delegates to complete."""
    llm = QwenLLM()

    # Patch on the CLASS, not the instance; keep the bound-method signature.
    with patch.object(QwenLLM, "complete", autospec=True) as mock_complete:
        mock_complete.return_value = CompletionResponse(text="Stream response")

        result = list(llm.stream_complete("Test prompt"))

        assert len(result) == 1
        assert result[0].text == "Stream response"

        # Because we patched the class with autospec=True, the first arg is `self`.
        mock_complete.assert_called_once()
        # Check the prompt argument (index 1; index 0 is `self`)
        assert mock_complete.call_args.args[1] == "Test prompt"
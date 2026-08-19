import os
import asyncio
import httpx
from openai import AsyncOpenAI, APIConnectionError, APITimeoutError, InternalServerError, RateLimitError
import structlog
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

from .schemas import ChatRequest, ChatResponse
from .providers import ProviderConfig

logger = structlog.get_logger(__name__)

class CompletionTruncatedError(Exception):
    pass

class LLMClient:
    """Unified LLM client that works with OpenAI-compatible APIs."""

    def __init__(self, provider: ProviderConfig, timeout_seconds: float = 30.0):
        self.provider = provider
        self.timeout_seconds = timeout_seconds
        
        api_key = "dummy" if not provider.api_key_env_var else os.environ.get(provider.api_key_env_var, "")
        if provider.api_key_env_var and not api_key:
            logger.warning(f"API key environment variable {provider.api_key_env_var} not set for provider {provider.name}")
            
        self.client = AsyncOpenAI(
            base_url=provider.base_url,
            api_key=api_key,
            timeout=httpx.Timeout(timeout_seconds),
            max_retries=0,
        )

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type((APIConnectionError, APITimeoutError, InternalServerError, RateLimitError, CompletionTruncatedError)),
        reraise=True
    )
    async def complete(self, request: ChatRequest) -> ChatResponse:
        """Execute a chat completion request with retry logic."""
        logger.info("llm_completion_started", model=request.model, provider=self.provider.name)
        
        kwargs = {
            "model": request.model,
            "messages": [msg.model_dump() for msg in request.messages],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        if request.response_format:
            kwargs["response_format"] = request.response_format

        try:
            response = await self.client.chat.completions.create(**kwargs)
            
            content = response.choices[0].message.content or ""
            usage = response.usage.model_dump() if response.usage else None
            
            finish_reason = response.choices[0].finish_reason if hasattr(response.choices[0], 'finish_reason') else None
            if finish_reason == "length":
                logger.error("llm_completion_truncated", model=request.model, provider=self.provider.name)
                raise CompletionTruncatedError("Completion truncated due to max_tokens (finish_reason='length')")

            logger.info("llm_completion_success", model=request.model, provider=self.provider.name)
            return ChatResponse(
                content=content,
                model=response.model,
                usage=usage
            )
        except Exception as e:
            logger.error("llm_completion_failed", error=str(e), provider=self.provider.name)
            raise

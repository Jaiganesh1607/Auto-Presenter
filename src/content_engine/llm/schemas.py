from typing import Optional, Literal
from pydantic import BaseModel, Field

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float = 0.7
    max_tokens: int = 1024
    response_format: Optional[dict] = None

class ChatResponse(BaseModel):
    content: str
    model: str
    usage: Optional[dict] = None

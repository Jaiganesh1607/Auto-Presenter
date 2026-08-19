import structlog
from pydantic import BaseModel
from src.content_engine.llm.client import LLMClient
from src.content_engine.llm.schemas import ChatRequest, ChatMessage
from src.content_engine.outline.templates import GENRE_TEMPLATES
from src.content_engine.llm.providers import get_provider

logger = structlog.get_logger(__name__)

class GenreResponse(BaseModel):
    genre: str

async def classify_genre(topic: str, llm_client: LLMClient) -> str:
    """Classify the topic into one of the supported outline genres."""
    allowed_genres = list(GENRE_TEMPLATES.keys())
    
    prompt = f"""You are a content planner. Categorize the topic "{topic}" into exactly ONE of these genres: {', '.join(allowed_genres)}.
Return a JSON object with a single key "genre". Example: {{"genre": "explainer"}}."""
    
    fast_model = llm_client.provider.model_name
        
    request = ChatRequest(
        model=fast_model,
        messages=[ChatMessage(role="user", content=prompt)],
        temperature=0.1,
        response_format={"type": "json_object"}
    )
    
    try:
        response = await llm_client.complete(request)
        import json_repair
        data = json_repair.repair_json(response.content, return_objects=True)
        if not isinstance(data, dict):
            data = {}
        genre = data.get("genre", "").strip()
        
        if genre in allowed_genres:
            return genre
        else:
            logger.warning("genre_classifier_invalid_output", output=genre, fallback="explainer")
            return "explainer"
    except Exception as e:
        logger.error("genre_classifier_failed", error=str(e), fallback="explainer")
        return "explainer"

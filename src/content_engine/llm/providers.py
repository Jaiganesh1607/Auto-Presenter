import os
import yaml
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field

class ProviderConfig(BaseModel):
    name: str
    base_url: str
    api_key_env_var: Optional[str] = None
    model_name: str

def load_providers() -> dict[str, ProviderConfig]:
    config_path = Path(__file__).parent.parent.parent.parent / "configs" / "models.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)
        
    providers = {}
    for name, config in data.get("providers", {}).items():
        providers[name] = ProviderConfig(
            name=name,
            base_url=config["base_url"],
            api_key_env_var=config.get("api_key_env_var"),
            model_name=config["model_name"],
        )
    return providers

def get_provider(name: str) -> ProviderConfig:
    providers = load_providers()
    if name not in providers:
        raise ValueError(f"Provider '{name}' not found in configs/models.yaml")
    return providers[name]

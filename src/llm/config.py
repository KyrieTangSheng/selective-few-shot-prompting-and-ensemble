from dataclasses import dataclass

@dataclass
class LLMConfig:
    model_name: str = "gpt-3.5-turbo"
    temperature: float = 0.0
    max_tokens: int = 50
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    retry_attempts: int = 3
    retry_delay: float = 1.0
    timeout: float = 30.0
    cache_dir: str = "./cache/llm"
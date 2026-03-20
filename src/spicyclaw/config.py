"""Application configuration via pydantic-settings."""

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_prefix": "SPICYCLAW_", "env_file": ".env"}

    # LLM
    api_base_url: str = "http://localhost:11434/v1"
    api_key: str = "sk-placeholder"
    model: str = "qwen2.5:14b"
    max_tokens: int = 32768
    request_timeout: float = 1800.0  # 30 minutes
    idle_timeout: float = 30.0
    health_check_interval: float = 10.0

    # Server
    host: str = "127.0.0.1"
    port: int = 8000

    # Agent
    max_steps: int = 1000
    yolo: bool = True
    max_repeat_errors: int = 5
    max_repeat_outputs: int = 10
    full_compact_ratio: float = 0.8
    compact_keep_rounds: int = 4

    # Shell
    shell_timeout: float = 120.0
    shell_max_output: int = 10000

    # Language
    lang: str = "en"  # "en" or "zh"

    # Paths
    data_dir: Path = Path("data")
    roles_dir: Path = Path("roles")
    skills_dir: Path = Path("skills")
    logs_dir: Path = Path("logs")

    @property
    def sessions_dir(self) -> Path:
        return self.data_dir / "sessions"

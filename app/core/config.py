from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM
    llm_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    # Statewave (self-hosted — no API key required for local dev)
    statewave_api_key: str = ""
    statewave_base_url: str = "http://localhost:8100"
    statewave_max_tokens: int = 800

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"

    # Comma-separated allowed CORS origins.
    # Defaults to localhost only. Override in .env for production.
    cors_origins: str = "http://localhost:8000,http://127.0.0.1:8000"

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid = {"debug", "info", "warning", "error", "critical"}
        if v.lower() not in valid:
            raise ValueError(f"log_level must be one of {valid}, got {v!r}")
        return v.lower()

    @field_validator("statewave_max_tokens")
    @classmethod
    def validate_max_tokens(cls, v: int) -> int:
        if v < 1 or v > 128_000:
            raise ValueError(f"statewave_max_tokens must be 1..128000, got {v}")
        return v

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def statewave_configured(self) -> bool:
        return bool(self.statewave_base_url)

    @property
    def llm_configured(self) -> bool:
        return bool(self.llm_api_key)


settings = Settings()

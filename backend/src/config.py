from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class ConfigError(RuntimeError):
    pass


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    azure_openai_endpoint: str = ""
    azure_openai_api_key: str = ""
    azure_openai_api_version: str = "2024-10-21"
    azure_openai_chat_deployment: str = ""
    azure_openai_embedding_deployment: str = "text-embedding-3-small"

    azure_search_endpoint: str = ""
    azure_search_api_key: str = ""
    azure_search_index_name: str = "brand-guidelines"

    azure_vi_account_id: str = ""
    azure_vi_location: str = ""
    azure_vi_name: str = ""
    azure_subscription_id: str = ""
    azure_resource_group: str = ""

    applicationinsights_connection_string: str = ""

    retrieval_k: int = 3

    def require(self, *fields: str) -> None:
        missing = [f.upper() for f in fields if not getattr(self, f)]
        if missing:
            raise ConfigError(f"missing environment variables: {', '.join(missing)}")


@lru_cache
def get_settings() -> Settings:
    return Settings()

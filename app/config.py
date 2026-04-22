"""应用配置：基于 pydantic-settings，支持环境变量与 .env。"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = Field(default="enterprise-ai-agent", description="服务名称")
    app_env: str = Field(default="development", description="运行环境")
    debug: bool = Field(default=False, description="调试模式")
    api_prefix: str = Field(default="/api/v1", description="API 前缀")
    host: str = Field(default="0.0.0.0", description="监听地址")
    port: int = Field(default=8000, description="监听端口")

    openai_api_key: str = Field(default="", description="OpenAI API Key")
    openai_api_base: str = Field(default="https://api.openai.com/v1", description="OpenAI 兼容 API Base")
    openai_model: str = Field(default="gpt-4o-mini", description="默认对话模型")
    openai_embedding_model: str = Field(default="text-embedding-3-small", description="默认 embedding 模型")

    database_url: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/agent_db",
        description="SQLAlchemy 异步数据库 URL",
    )
    redis_url: str = Field(default="redis://localhost:6379/0", description="Redis URL")

    milvus_host: str = Field(default="localhost", description="Milvus 主机")
    milvus_port: int = Field(default=19530, description="Milvus 端口")
    milvus_user: str = Field(default="", description="Milvus 用户名")
    milvus_password: str = Field(default="", description="Milvus 密码")
    milvus_collection_name: str = Field(default="agent_knowledge", description="默认向量集合")
    enable_embedding_upload: bool = Field(default=True, description="上传时是否写入 embedding")

    triage_retention_days: int = Field(default=7, description="分诊会话留存天数")
    triage_top_k: int = Field(default=6, description="分诊检索 TopK")
    triage_max_chunks: int = Field(default=1200, description="分诊候选文档上限")

    log_level: str = Field(default="INFO", description="日志级别")


@lru_cache
def get_settings() -> Settings:
    return Settings()

# -*- coding: utf-8 -*-
"""FastAPI 应用入口：生命周期内初始化异步数据库引擎。"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from loguru import logger

from app.api.routes import chat, document, health, triage
from app.config import get_settings
from app.infrastructure.database.session import configure_session, init_engine, init_schema


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info("启动 {} ({})", settings.app_name, settings.app_env)
    engine = init_engine(settings.database_url)
    await init_schema(engine)
    configure_session(engine)
    app.state.engine = engine
    yield
    await engine.dispose()
    logger.info("关闭 {}", settings.app_name)


def create_app() -> FastAPI:
    settings = get_settings()
    application = FastAPI(
        title=settings.app_name,
        debug=settings.debug,
        lifespan=lifespan,
    )
    application.include_router(health.router, prefix=settings.api_prefix)
    application.include_router(chat.router, prefix=settings.api_prefix)
    application.include_router(document.router, prefix=settings.api_prefix)
    application.include_router(triage.router, prefix=settings.api_prefix)

    web_dir = Path("web")
    if web_dir.exists():
        application.mount("/web", StaticFiles(directory=str(web_dir), html=True), name="web")

    return application


app = create_app()

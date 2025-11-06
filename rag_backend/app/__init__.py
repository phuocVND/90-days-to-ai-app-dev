from fastapi import FastAPI
from app.core.config import setup_middlewares
from app.api.routes.ask_router import router as ask_router

def create_app():
    app = FastAPI(title="RAG Backend API")
    setup_middlewares(app)
    app.include_router(ask_router)
    return app
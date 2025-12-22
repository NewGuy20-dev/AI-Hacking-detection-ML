"""FastAPI server for AI Hacking Detection."""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import time

# Global state
predictor = None
start_time = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    global predictor, start_time
    
    print("Loading models...")
    from src.batch_predictor import BatchHybridPredictor
    
    predictor = BatchHybridPredictor(models_dir='models', validator=True)
    predictor.load_models()
    start_time = time.time()
    
    print(f"Loaded {len(predictor.pytorch_models)} PyTorch, {len(predictor.sklearn_models)} sklearn models")
    yield
    print("Shutting down...")


app = FastAPI(
    title="AI Hacking Detection API",
    description="Real-time cyber attack detection using ensemble ML models",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://*.vercel.app",
    ],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Import and include routers
from src.api.routes import predict, health
app.include_router(predict.router, prefix="/api/v1")
app.include_router(health.router)


def get_predictor():
    return predictor


def get_uptime() -> float:
    return time.time() - start_time if start_time else 0

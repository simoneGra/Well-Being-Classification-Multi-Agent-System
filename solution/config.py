import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent.parent
load_dotenv(BASE_DIR / ".env")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST")
TEAM_NAME = os.getenv("TEAM_NAME")

MODEL_ID = "stepfun/step-3.5-flash:free"
BASE_URL = "https://openrouter.ai/api/v1"
TEMPERATURE = 0.1  # Low temperature for deterministic classification
MAX_TOKENS = 4000  # Reasoning model needs extra budget for chain-of-thought before output

DATA_DIR = BASE_DIR / "data"
TRAIN_DIR = DATA_DIR / "training" / os.getenv("TRAIN_FOLDER", "public_lev_1")
EVAL_DIR = DATA_DIR / "evaluation" / os.getenv("EVAL_FOLDER", "public_lev_1_ev")
OUTPUT_DIR = BASE_DIR / "output"

# High-risk event types (signals escalating care needs)
ESCALATED_EVENT_TYPES = {
    "specialist consultation",
    "follow-up assessment",
    "emergency visit",
    "urgent care visit",
    "hospitalization",
}

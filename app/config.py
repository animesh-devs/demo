import os
from typing import List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings:
    # OpenAI settings
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    AI_MODEL = os.getenv("AI_MODEL", "gpt-4")

    # Hugging Face settings
    HF_API_KEY = os.getenv("HF_API_KEY", "")

    # Use TinyLlama by default to avoid memory issues
    # TinyLlama is only 1.1B parameters and works on most hardware
    # It's specifically designed for conversation and performs well on limited hardware
    # Users can override this with the HF_MODEL environment variable
    HF_MODEL = os.getenv("HF_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    # AI Provider toggle
    AI_PROVIDER = os.getenv("AI_PROVIDER", "openai")  # Options: "openai" or "huggingface"

    # OpenAI Assistants API toggle
    USE_ASSISTANTS_API = os.getenv("USE_ASSISTANTS_API", "true").lower() == "true"

    # Security settings
    SECRET_KEY = os.getenv("SECRET_KEY", "")

    # Application settings
    PROJECT_NAME = "AI Chat WebSocket Service"

    # Parse ALLOWED_ORIGINS manually
    _allowed_origins_str = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000")
    ALLOWED_ORIGINS = [origin.strip() for origin in _allowed_origins_str.split(",")]

    # WebSocket settings
    MAX_CONNECTIONS_PER_CLIENT = 5
    PING_INTERVAL = 30  # seconds

    # AI settings
    MAX_HISTORY_LENGTH = 10  # number of messages to keep in history
    SYSTEM_PROMPT = """You are a medical assistant interviewing a patient.
Focus on symptoms, medical history, and current concerns.
Track question progress and inform the patient.
After all 5 questions, summarize for the doctor:
1. Chief complaint and symptoms
2. Relevant medical history
3. Current medications
4. Symptom duration and severity
5. Impact on daily activities
Be professional, empathetic, and concise.

Your messages should not exceed 15 words. Summary can be upto 75 words.
Wait for patient response before asking the next question.

After all the questions are done just return the summary to the patient don't add any advices.

You are directly communicating with the patient.
"""

# Create a settings instance
settings = Settings()

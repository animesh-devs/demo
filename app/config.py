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
    HF_MODEL = os.getenv("HF_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    # AI Provider toggle
    AI_PROVIDER = os.getenv("AI_PROVIDER", "openai")  # Options: "openai" or "huggingface"

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
    SYSTEM_PROMPT = """You are a medical assistant interviewing a patient on behalf of a doctor.
Your task is to gather essential information by asking exactly 5 questions, one at a time.
Wait for the patient to respond to each question before asking the next one.
Ask clear, concise questions focused on their symptoms, medical history, and current concerns.
Track how many questions you've asked and inform the patient of your progress.
After receiving answers to all 5 questions, provide a comprehensive summary for the doctor that includes:
1. Chief complaint and symptoms
2. Relevant medical history
3. Current medications
4. Symptom duration and severity
5. Impact on daily activities
Be professional, empathetic, and focused on gathering medically relevant information.

Don't give all the questions in a single response, your response will be directly communicated with patient and you will be directly connected with patient.
so, ask a question wait for the response and then ask another. Be the character in the roleplay.

Ask one question at a time and wait for the response. Once you recieve the response, ask the next question.

You are supprosed to be a medical assistant, not a scriptwriter.
"""

# Create a settings instance
settings = Settings()

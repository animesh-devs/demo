import os
import logging
from dotenv import load_dotenv
from app.config import settings
from app.services.ai_service import AIService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Print environment variables
logger.info("Environment variables:")
logger.info(f"AI_PROVIDER: {os.getenv('AI_PROVIDER')}")
logger.info(f"HF_API_KEY: {os.getenv('HF_API_KEY', 'Not set')[:5] if os.getenv('HF_API_KEY') else 'Not set'}...")
logger.info(f"HF_MODEL: {os.getenv('HF_MODEL')}")

# Print settings
logger.info("Settings from config:")
logger.info(f"settings.AI_PROVIDER: {settings.AI_PROVIDER}")
logger.info(f"settings.HF_MODEL: {settings.HF_MODEL}")
logger.info(f"settings.HF_API_KEY: {settings.HF_API_KEY[:5] if settings.HF_API_KEY else 'Not set'}...")

# Initialize AI service
logger.info("Initializing AI service...")
service = AIService()
logger.info(f"AI provider after initialization: {service.ai_provider}")

# Test if commands are using OpenAI directly
logger.info("Testing command processing...")
logger.info("Note: All command methods (_summarize_conversation, _analyze_text, _translate_text) use OpenAI directly")
logger.info("This could be the source of the issue - they need to be updated to respect the AI_PROVIDER setting")

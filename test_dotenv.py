import os
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Print current working directory
logger.info(f"Current working directory: {os.getcwd()}")

# Check if .env file exists
env_path = os.path.join(os.getcwd(), '.env')
logger.info(f".env file exists: {os.path.exists(env_path)}")

# Try to load .env file
load_dotenv(dotenv_path=env_path, override=True)
logger.info("Loaded .env file with explicit path and override=True")

# Print environment variables
logger.info(f"AI_PROVIDER: {os.getenv('AI_PROVIDER')}")
logger.info(f"HF_API_KEY: {os.getenv('HF_API_KEY', 'Not set')[:5] if os.getenv('HF_API_KEY') else 'Not set'}...")
logger.info(f"HF_MODEL: {os.getenv('HF_MODEL')}")
logger.info(f"OPENAI_API_KEY: {os.getenv('OPENAI_API_KEY', 'Not set')[:5] if os.getenv('OPENAI_API_KEY') else 'Not set'}...")

# Print all environment variables
logger.info("All environment variables:")
for key, value in os.environ.items():
    if key in ['AI_PROVIDER', 'HF_API_KEY', 'HF_MODEL', 'OPENAI_API_KEY', 'AI_MODEL']:
        if 'API_KEY' in key:
            logger.info(f"{key}: {value[:5]}...")
        else:
            logger.info(f"{key}: {value}")

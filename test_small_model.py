import os
import logging
from dotenv import load_dotenv
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Print environment variables
logger.info("Environment variables:")
logger.info(f"AI_PROVIDER: {os.getenv('AI_PROVIDER')}")
logger.info(f"HF_API_KEY: {os.getenv('HF_API_KEY', 'Not set')[:5] if os.getenv('HF_API_KEY') else 'Not set'}...")

# Use a very small model for testing
model_name = "distilbert-base-uncased"
logger.info(f"Using model: {model_name}")

try:
    # Initialize the pipeline
    logger.info("Initializing sentiment analysis pipeline...")
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model=model_name,
        token=os.getenv("HF_API_KEY")
    )
    
    # Test the pipeline
    logger.info("Testing the pipeline...")
    result = sentiment_pipeline("I love this product!")
    logger.info(f"Result: {result}")
    
    logger.info("Hugging Face pipeline is working correctly!")
    logger.info("The issue with your application is likely that it's still using OpenAI for command processing.")
    logger.info("You need to update the command processing methods to use Hugging Face when AI_PROVIDER is set to 'huggingface'.")
    
except Exception as e:
    logger.error(f"Failed to initialize Hugging Face pipeline: {str(e)}")
    logger.info("Falling back to OpenAI is expected behavior when Hugging Face initialization fails.")
    logger.info("To fix this, make sure you have the necessary dependencies installed and the correct API key.")

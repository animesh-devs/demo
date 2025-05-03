import openai
from typing import List, Dict, Any, Optional
import logging
import json
import asyncio
from app.config import settings
from transformers import pipeline
import requests
import os

# Configure logging
logging.basicConfig(level=logging.INFO)

# Debug environment variables
logger = logging.getLogger(__name__)
logger.info(f"Environment variables:")
logger.info(f"AI_PROVIDER: {os.getenv('AI_PROVIDER')}")
logger.info(f"HF_API_KEY: {os.getenv('HF_API_KEY', 'REDACTED')[:5]}...")
logger.info(f"HF_MODEL: {os.getenv('HF_MODEL')}")

# Configure OpenAI API
openai.api_key = settings.OPENAI_API_KEY
logger.info(f"Settings from config:")
logger.info(f"settings.AI_PROVIDER: {settings.AI_PROVIDER}")
logger.info(f"settings.HF_MODEL: {settings.HF_MODEL}")
logger.info(f"settings.HF_API_KEY: {settings.HF_API_KEY[:5] if settings.HF_API_KEY else 'Not set'}...")

class AIService:
    def __init__(self):
        self.model = settings.AI_MODEL
        self.system_prompt = settings.SYSTEM_PROMPT
        self.client_contexts = {}
        self.question_counts = {}

        # Initialize Hugging Face if selected
        self.ai_provider = settings.AI_PROVIDER
        logger.info(f"Initial AI provider from settings: {self.ai_provider}")

        if self.ai_provider == "huggingface":
            logger.info("Attempting to initialize Hugging Face pipeline")
            try:
                if not settings.HF_API_KEY:
                    logger.error("HF_API_KEY is not set or empty")
                    raise ValueError("HF_API_KEY is not set")

                if not settings.HF_MODEL:
                    logger.error("HF_MODEL is not set or empty")
                    raise ValueError("HF_MODEL is not set")

                logger.info(f"Creating pipeline with model {settings.HF_MODEL}")
                # Use TinyLlama-1.1B-Chat-v1.0, which is a small but powerful chat model
                # It's specifically designed for conversation and performs well on limited hardware
                self.hf_pipeline = pipeline(
                    "text-generation",
                    model=settings.HF_MODEL,
                    token=settings.HF_API_KEY,
                    truncation=True,  # Explicitly enable truncation to avoid warnings
                    torch_dtype="auto"  # Automatically use the best precision for the device
                )
                logger.info(f"Successfully initialized Hugging Face with model TinyLlama/TinyLlama-1.1B-Chat-v1.0")

                # Also initialize a sentiment analysis pipeline for command processing
                self.hf_sentiment_pipeline = pipeline(
                    "sentiment-analysis",
                    model="distilbert-base-uncased",
                    token=settings.HF_API_KEY,
                    truncation=True  # Explicitly enable truncation to avoid warnings
                )
                logger.info("Successfully initialized Hugging Face sentiment analysis pipeline")
            except Exception as e:
                logger.error(f"Failed to initialize Hugging Face: {str(e)}")
                # Fallback to OpenAI
                self.ai_provider = "openai"
                logger.info("Falling back to OpenAI due to Hugging Face initialization error")

    def set_context(self, client_id: str, context: str) -> None:
        """Set the context for a specific client"""
        self.client_contexts[client_id] = context
        logger.info(f"Context set for client {client_id}: {context[:50]}...")

    def get_context(self, client_id: str) -> Optional[str]:
        """Get the context for a specific client"""
        return self.client_contexts.get(client_id)

    def clear_context(self, client_id: str) -> None:
        """Clear the context for a specific client"""
        if client_id in self.client_contexts:
            del self.client_contexts[client_id]
            logger.info(f"Context cleared for client {client_id}")

        # Also reset question count when context is cleared
        if client_id in self.question_counts:
            del self.question_counts[client_id]
            logger.info(f"Question count reset for client {client_id}")

    async def process_message(self, message: str, history: List[Dict[str, Any]], client_id: str = None, is_first_message: bool = False) -> str:
        """Process a message using either OpenAI or Hugging Face"""
        try:
            logger.info(f"Processing message with AI provider: {self.ai_provider}")

            if self.ai_provider == "openai":
                logger.info("Using OpenAI for processing")
                return await self._process_with_openai(message, history, client_id, is_first_message)
            elif self.ai_provider == "huggingface":
                logger.info("Using Hugging Face for processing")
                return await self._process_with_huggingface(message, history, client_id, is_first_message)
            else:
                logger.error(f"Unknown AI provider: {self.ai_provider}")
                return "Configuration error: Unknown AI provider specified."
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            return f"I'm sorry, I encountered an error processing your request. Please try again later."

    async def _process_with_openai(self, message: str, history: List[Dict[str, Any]], client_id: str = None, is_first_message: bool = False) -> str:
        """Process message with OpenAI"""
        # Format conversation history for OpenAI
        messages = []
        
        # Only include system prompt for the first message in the conversation
        if is_first_message:
            system_content = self.system_prompt
            logger.info("system_prompt: {system_content}")

            if client_id and client_id in self.client_contexts:
                system_content = f"{system_content}\n\nContext: {self.client_contexts[client_id]}"

            # Initialize question count for this client if not exists
            if client_id and client_id not in self.question_counts:
                self.question_counts[client_id] = 0

            # Add question count to system prompt
            question_count = self.question_counts[client_id]
            if question_count < 5:
                system_content += f"\n\nYou have asked {question_count} questions so far. You need to ask {5 - question_count} more questions before providing a summary."
            else:
                system_content += "\n\nYou have asked all 5 questions. Now provide a summary of the patient's condition based on their answers."
            
            messages.append({"role": "system", "content": system_content})

        # Add conversation history
        for entry in history:
            if entry["role"] in ["user", "assistant"]:
                messages.append({
                    "role": entry["role"],
                    "content": entry["content"]
                })

        # Add the current message
        messages.append({"role": "user", "content": message})

        # Count assistant messages in history to track questions asked
        if client_id:
            assistant_messages = [entry for entry in history if entry["role"] == "assistant"]
            self.question_counts[client_id] = len(assistant_messages)

        # Call OpenAI API with a 5-minute timeout
        response = await asyncio.wait_for(
            openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            ),
            timeout=300  # 5 minutes timeout
        )

        return response.choices[0].message.content

    async def _process_with_huggingface(self, message: str, history: List[Dict[str, Any]], client_id: str = None, is_first_message: bool = False) -> str:
        """Process message with Hugging Face"""
        try:
            # For TinyLlama-1.1B-Chat-v1.0, we'll use its chat template
            # First, prepare the messages in the format expected by the chat template
            messages = []

            # Only include system prompt for the first message in the conversation
            if is_first_message:
                logger.info("system_prompt: {system_content}")
                system_content = self.system_prompt
                if client_id and client_id in self.client_contexts:
                    system_content += f"\n\nContext: {self.client_contexts[client_id]}"

                # Initialize question count for this client if not exists
                if client_id and client_id not in self.question_counts:
                    self.question_counts[client_id] = 0

                # Add question count to system prompt
                question_count = self.question_counts[client_id]
                if question_count < 5:
                    system_content += f"\n\nYou have asked {question_count} questions so far. You need to ask {5 - question_count} more questions before providing a summary."
                else:
                    system_content += "\n\nYou have asked all 5 questions. Now provide a summary of the patient's condition based on their answers."
                
                messages.append({"role": "system", "content": system_content})

            # Add conversation history
            for entry in history:
                if entry["role"] in ["user", "assistant"]:
                    messages.append({
                        "role": entry["role"],
                        "content": entry["content"]
                    })

            # Add the current message
            messages.append({"role": "user", "content": message})

            # Count assistant messages in history to track questions asked
            if client_id:
                assistant_messages = [entry for entry in history if entry["role"] == "assistant"]
                self.question_counts[client_id] = len(assistant_messages)

            # Apply the chat template
            prompt = self.hf_pipeline.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            logger.info(f"Sending prompt to TinyLlama: {prompt[:100]}...")

            # Run in executor to avoid blocking, with a 5-minute timeout
            loop = asyncio.get_event_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: self.hf_pipeline(
                        prompt,
                        max_new_tokens=1024,  # Generate more tokens for a complete response
                        temperature=0.7,
                        top_p=0.95,
                        do_sample=True,
                        num_return_sequences=1,
                        truncation=True  # Explicitly enable truncation to avoid warnings
                    )
                ),
                timeout=300  # 5 minutes timeout
            )

            # Extract generated text
            generated_text = result[0]['generated_text']

            # Extract only the assistant's response
            # The format should be something like: <prompt>...<|assistant|>\nResponse
            if "<|assistant|>" in generated_text:
                response = generated_text.split("<|assistant|>")[-1].strip()
                # Remove any trailing system or user messages if present
                if "<|system|>" in response:
                    response = response.split("<|system|>")[0].strip()
                if "<|user|>" in response:
                    response = response.split("<|user|>")[0].strip()
            else:
                # Fallback to simpler extraction if the template markers aren't found
                response = generated_text[len(prompt):].strip()

            return response

        except asyncio.TimeoutError:
            logger.error("Hugging Face request timed out after 5 minutes")
            return "I'm sorry, but the request timed out. Please try again with a shorter message."
        except Exception as e:
            logger.error(f"Error with Hugging Face: {str(e)}")
            # Fallback to OpenAI if HF fails
            logger.info("Falling back to OpenAI for this request")
            return await self._process_with_openai(message, history, client_id, is_first_message)

    async def process_command(self, command: str, params: Dict[str, Any], history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process a command with specific parameters"""
        try:
            logger.info(f"Processing command: {command} with AI provider: {self.ai_provider}")

            # Now respecting the AI_PROVIDER setting
            logger.info(f"Using AI provider: {self.ai_provider} for command processing")

            if command == "summarize":
                return await self._summarize_conversation(history)
            elif command == "analyze":
                return await self._analyze_text(params.get("text", ""))
            elif command == "translate":
                return await self._translate_text(
                    params.get("text", ""),
                    params.get("target_language", "English")
                )
            else:
                return {
                    "status": "error",
                    "message": f"Unknown command: {command}"
                }

        except Exception as e:
            logger.error(f"Error processing command {command}: {str(e)}")
            return {
                "status": "error",
                "message": f"Error processing command: {str(e)}"
            }

    async def _summarize_conversation(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize the conversation history"""
        if not history:
            return {"status": "error", "message": "No conversation history to summarize"}

        # Format conversation for summarization
        conversation_text = "\n".join([
            f"{entry['role'].capitalize()}: {entry['content']}"
            for entry in history
        ])

        prompt = f"""Please summarize the following conversation concisely:

{conversation_text}

Summary:"""

        # Check AI provider setting
        logger.info(f"Summarizing with AI provider: {self.ai_provider}")

        if self.ai_provider == "huggingface" and hasattr(self, 'hf_pipeline'):
            try:
                # Use the chat template for TinyLlama
                messages = [
                    {"role": "system", "content": "You are a helpful assistant that summarizes conversations."},
                    {"role": "user", "content": prompt}
                ]

                # Apply the chat template
                formatted_prompt = self.hf_pipeline.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )

                # Run in executor to avoid blocking
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda: self.hf_pipeline(
                        formatted_prompt,
                        max_new_tokens=1024,
                        temperature=0.3,
                        top_p=0.95,
                        do_sample=True,
                        num_return_sequences=1,
                        truncation=True  # Explicitly enable truncation to avoid warnings
                    )
                )

                # Extract generated text
                generated_text = result[0]['generated_text']

                # Extract only the assistant's response
                if "<|assistant|>" in generated_text:
                    summary = generated_text.split("<|assistant|>")[-1].strip()
                    # Remove any trailing system or user messages if present
                    if "<|system|>" in summary:
                        summary = summary.split("<|system|>")[0].strip()
                    if "<|user|>" in summary:
                        summary = summary.split("<|user|>")[0].strip()
                else:
                    # Fallback to simpler extraction if the template markers aren't found
                    summary = generated_text[len(formatted_prompt):].strip()

                return {
                    "status": "success",
                    "summary": summary
                }
            except Exception as e:
                logger.error(f"Error with Hugging Face summarization: {str(e)}")
                logger.info("Falling back to OpenAI for summarization")
                # Fall through to OpenAI implementation

        # OpenAI implementation (default or fallback)
        messages = [
            {"role": "system", "content": "You are a helpful assistant that summarizes conversations."},
            {"role": "user", "content": prompt}
        ]

        response = await openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            temperature=0.3,
            max_tokens=250
        )

        return {
            "status": "success",
            "summary": response.choices[0].message.content
        }

    async def _analyze_text(self, text: str) -> Dict[str, Any]:
        """Analyze the sentiment and key points of a text"""
        if not text:
            return {"status": "error", "message": "No text provided for analysis"}

        # Check AI provider setting
        logger.info(f"Analyzing text with AI provider: {self.ai_provider}")

        if self.ai_provider == "huggingface" and hasattr(self, 'hf_pipeline'):
            try:
                # Use Hugging Face for text analysis
                logger.info("Using Hugging Face for text analysis")

                # First try sentiment analysis if available
                if hasattr(self, 'hf_sentiment_pipeline'):
                    # Run sentiment analysis in executor to avoid blocking
                    loop = asyncio.get_event_loop()
                    sentiment_result = await loop.run_in_executor(
                        None,
                        lambda: self.hf_sentiment_pipeline(text)
                    )

                    # Format the sentiment result
                    sentiment = sentiment_result[0]['label']
                    score = sentiment_result[0]['score']

                    # Include sentiment in our prompt to TinyLlama
                    sentiment_info = f"Sentiment: {sentiment} (confidence: {score:.2f})"
                else:
                    sentiment_info = "Sentiment analysis not available"

                # Use the chat template for TinyLlama for more detailed analysis
                messages = [
                    {"role": "system", "content": "You are a helpful assistant that analyzes text."},
                    {"role": "user", "content": f"""Please analyze the following text and provide:
1. Key points or main ideas
2. Any notable entities mentioned
3. Brief summary

Text: {text}

Note: {sentiment_info}"""}
                ]

                # Apply the chat template
                formatted_prompt = self.hf_pipeline.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )

                # Run in executor to avoid blocking
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda: self.hf_pipeline(
                        formatted_prompt,
                        max_new_tokens=350,
                        temperature=0.3,
                        top_p=0.95,
                        do_sample=True,
                        num_return_sequences=1,
                        truncation=True  # Explicitly enable truncation to avoid warnings
                    )
                )

                # Extract generated text
                generated_text = result[0]['generated_text']

                # Extract only the assistant's response
                if "<|assistant|>" in generated_text:
                    analysis = generated_text.split("<|assistant|>")[-1].strip()
                    # Remove any trailing system or user messages if present
                    if "<|system|>" in analysis:
                        analysis = analysis.split("<|system|>")[0].strip()
                    if "<|user|>" in analysis:
                        analysis = analysis.split("<|user|>")[0].strip()
                else:
                    # Fallback to simpler extraction if the template markers aren't found
                    analysis = generated_text[len(formatted_prompt):].strip()

                return {
                    "status": "success",
                    "analysis": analysis
                }
            except Exception as e:
                logger.error(f"Error with Hugging Face analysis: {str(e)}")
                logger.info("Falling back to OpenAI for text analysis")
                # Fall through to OpenAI implementation

        # OpenAI implementation (default or fallback)
        prompt = f"""Please analyze the following text and provide:
1. Overall sentiment (positive, negative, or neutral)
2. Key points or main ideas
3. Any notable entities mentioned

Text: {text}

Analysis:"""

        messages = [
            {"role": "system", "content": "You are a helpful assistant that analyzes text."},
            {"role": "user", "content": prompt}
        ]

        response = await openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            temperature=0.3,
            max_tokens=350
        )

        return {
            "status": "success",
            "analysis": response.choices[0].message.content
        }

    async def _translate_text(self, text: str, target_language: str) -> Dict[str, Any]:
        """Translate text to the target language"""
        if not text:
            return {"status": "error", "message": "No text provided for translation"}

        # Check AI provider setting
        logger.info(f"Translating text with AI provider: {self.ai_provider}")

        if self.ai_provider == "huggingface" and hasattr(self, 'hf_pipeline'):
            try:
                # Use Hugging Face for translation
                logger.info("Using Hugging Face for translation")

                # Use the chat template for TinyLlama
                messages = [
                    {"role": "system", "content": f"You are a helpful assistant that translates text to {target_language}."},
                    {"role": "user", "content": f"Please translate the following text to {target_language}:\n\n{text}"}
                ]

                # Apply the chat template
                formatted_prompt = self.hf_pipeline.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )

                # Run in executor to avoid blocking
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda: self.hf_pipeline(
                        formatted_prompt,
                        max_new_tokens=len(text.split()) * 2,  # Allocate enough tokens for translation
                        temperature=0.3,
                        top_p=0.95,
                        do_sample=True,
                        num_return_sequences=1,
                        truncation=True  # Explicitly enable truncation to avoid warnings
                    )
                )

                # Extract generated text
                generated_text = result[0]['generated_text']

                # Extract only the assistant's response
                if "<|assistant|>" in generated_text:
                    translation = generated_text.split("<|assistant|>")[-1].strip()
                    # Remove any trailing system or user messages if present
                    if "<|system|>" in translation:
                        translation = translation.split("<|system|>")[0].strip()
                    if "<|user|>" in translation:
                        translation = translation.split("<|user|>")[0].strip()
                else:
                    # Fallback to simpler extraction if the template markers aren't found
                    translation = generated_text[len(formatted_prompt):].strip()

                return {
                    "status": "success",
                    "translation": translation,
                    "source_text": text,
                    "target_language": target_language
                }
            except Exception as e:
                logger.error(f"Error with Hugging Face translation: {str(e)}")
                logger.info("Falling back to OpenAI for translation")
                # Fall through to OpenAI implementation

        # OpenAI implementation (default or fallback)
        prompt = f"""Please translate the following text to {target_language}:

{text}

Translation:"""

        messages = [
            {"role": "system", "content": f"You are a helpful assistant that translates text to {target_language}."},
            {"role": "user", "content": prompt}
        ]

        response = await openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            temperature=0.3,
            max_tokens=500
        )

        return {
            "status": "success",
            "translation": response.choices[0].message.content,
            "source_text": text,
            "target_language": target_language
        }

# Create a global AI service instance
ai_service = AIService()

from typing import List, Dict, Any, Optional, Tuple
import logging
import json
import asyncio
from app.config import settings
from transformers import pipeline, BitsAndBytesConfig
import os
import platform
import psutil  # For checking system memory

# Import OpenAI with async support
from openai import AsyncOpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)

# Debug environment variables
logger = logging.getLogger(__name__)
logger.info(f"Environment variables:")
logger.info(f"AI_PROVIDER: {os.getenv('AI_PROVIDER')}")
logger.info(f"HF_API_KEY: {os.getenv('HF_API_KEY', 'REDACTED')[:5]}...")
logger.info(f"HF_MODEL: {os.getenv('HF_MODEL')}")
logger.info(f"Settings from config:")
logger.info(f"settings.AI_PROVIDER: {settings.AI_PROVIDER}")
logger.info(f"settings.HF_MODEL: {settings.HF_MODEL}")
logger.info(f"settings.HF_API_KEY: {settings.HF_API_KEY[:5] if settings.HF_API_KEY else 'Not set'}...")

def get_system_memory() -> Tuple[float, float]:
    """Get available and total system memory in GB"""
    try:
        # Get memory information
        memory = psutil.virtual_memory()
        total_gb = memory.total / (1024 ** 3)  # Convert to GB
        available_gb = memory.available / (1024 ** 3)  # Convert to GB
        return available_gb, total_gb
    except Exception as e:
        logger.error(f"Error getting system memory: {str(e)}")
        # Return conservative defaults
        return 4.0, 8.0

class AIService:
    def __init__(self):
        self.model = settings.AI_MODEL
        self.system_prompt = settings.SYSTEM_PROMPT
        self.client_contexts = {}
        self.question_counts = {}
        self.client_threads = {}  # Store thread IDs for each client
        self.assistant_id = None  # Will store the assistant ID once created
        self.hf_conversation_states = {}  # Store conversation states for Hugging Face

        # Initialize OpenAI client
        self.openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

        # Initialize the assistant asynchronously
        asyncio.create_task(self._initialize_assistant())

        # Check available system memory
        try:
            available_memory, total_memory = get_system_memory()
            logger.info(f"System memory: {available_memory:.2f}GB available out of {total_memory:.2f}GB total")

            # Store memory info for later use
            self.available_memory = available_memory
            self.total_memory = total_memory
        except Exception as e:
            logger.error(f"Failed to check system memory: {str(e)}")
            # Use conservative defaults
            self.available_memory = 4.0
            self.total_memory = 8.0

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

                # Check if we're using a large model that might cause memory issues
                model_name = settings.HF_MODEL.lower()

                # Determine if this is a large model
                is_large_model = any(name in model_name for name in ["llama-2-7b", "llama-3", "mistral", "mixtral", "7b", "13b", "70b"])

                # Estimate model size based on name with more precise detection
                estimated_model_size = 0
                if "tinyllama" in model_name or "1.1b" in model_name or "1b" in model_name:
                    estimated_model_size = 2  # ~2GB for 1B parameter models
                elif "7b" in model_name:
                    estimated_model_size = 14  # ~14GB for 7B parameter models
                elif "13b" in model_name:
                    estimated_model_size = 26  # ~26GB for 13B parameter models
                elif "70b" in model_name:
                    estimated_model_size = 140  # ~140GB for 70B parameter models
                else:
                    # Default conservative estimate based on model name
                    if any(name in model_name for name in ["large", "xl", "xxl"]):
                        estimated_model_size = 8  # Larger models
                    else:
                        estimated_model_size = 4  # Smaller models

                logger.info(f"Estimated model size: {estimated_model_size}GB")

                # Add a safety margin for memory usage (models often need more memory during inference)
                safety_margin = 1.5  # 50% extra memory for safety
                required_memory = estimated_model_size * safety_margin

                # Check if we have enough memory with safety margin
                if self.available_memory < required_memory:
                    logger.warning(f"Available memory ({self.available_memory:.2f}GB) is less than required memory ({required_memory:.2f}GB)")

                    # If we're severely under the memory requirement, switch to TinyLlama immediately
                    if self.available_memory < estimated_model_size:
                        logger.warning("Insufficient memory for the requested model. Switching to TinyLlama.")
                        logger.info("Attempting to initialize with TinyLlama")
                        self._initialize_tiny_llama()

                    # Otherwise try with memory-efficient settings
                    logger.warning("Using memory-efficient settings and offloading to disk")
                    use_memory_efficient = True
                else:
                    logger.info(f"Sufficient memory available for model")
                    use_memory_efficient = is_large_model

                # Configure settings based on memory availability
                if use_memory_efficient:
                    logger.info(f"Using memory-efficient settings for model: {settings.HF_MODEL}")

                    # Create offload folder if it doesn't exist
                    os.makedirs("offload_folder", exist_ok=True)

                    # Create a BitsAndBytesConfig for proper quantization
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_threshold=6.0,
                        llm_int8_has_fp16_weight=False
                    )

                    # Try with memory-efficient settings first
                    try:
                        # First try with 8-bit quantization
                        logger.info("Trying with 8-bit quantization")

                        # Check if bitsandbytes is installed and up to date
                        try:
                            import bitsandbytes
                            logger.info(f"Using bitsandbytes version: {bitsandbytes.__version__}")
                        except ImportError:
                            logger.warning("bitsandbytes not installed. Will try without quantization.")
                            raise ImportError("bitsandbytes not installed")
                        except Exception as bb_error:
                            logger.warning(f"Error checking bitsandbytes: {str(bb_error)}")

                        # The quantization_config should be passed as a model_kwargs parameter
                        self.hf_pipeline = pipeline(
                            "text-generation",
                            model=settings.HF_MODEL,
                            token=settings.HF_API_KEY,
                            torch_dtype="auto",
                            device_map="auto",
                            model_kwargs={"quantization_config": quantization_config}  # Correct way to pass quantization_config
                        )
                    except Exception as e:
                        error_message = str(e)
                        logger.warning(f"Error with 8-bit quantization: {error_message}")

                        # Check for bitsandbytes specific errors
                        if "bitsandbytes" in error_message and "pip install" in error_message:
                            logger.warning("bitsandbytes needs to be updated. Will try to install it automatically.")
                            logger.info("To fix this issue manually, run: pip install -U bitsandbytes")

                            # Try to install bitsandbytes
                            if self._try_install_bitsandbytes():
                                # Try again with 8-bit quantization after installing bitsandbytes
                                try:
                                    logger.info("Retrying with 8-bit quantization after bitsandbytes update")
                                    self.hf_pipeline = pipeline(
                                        "text-generation",
                                        model=settings.HF_MODEL,
                                        token=settings.HF_API_KEY,
                                        torch_dtype="auto",
                                        device_map="auto",
                                        model_kwargs={"quantization_config": quantization_config}
                                    )
                                    logger.info("Successfully initialized with 8-bit quantization after bitsandbytes update")
                                    return
                                except Exception as retry_error:
                                    logger.error(f"Still failed with 8-bit quantization after bitsandbytes update: {str(retry_error)}")
                                    logger.info("Will try without quantization")

                        # Check for specific errors
                        if "Invalid buffer size" in error_message:
                            # Memory issue - switch to TinyLlama immediately
                            logger.warning("Memory issue detected. Switching to TinyLlama.")
                            self._initialize_tiny_llama()
                            return
                        elif "model_kwargs" in error_message or "not used by the model" in error_message:
                            # Issue with model_kwargs - try without quantization
                            logger.warning("Issue with model parameters. Trying without quantization.")

                        # Try with simpler configuration without quantization
                        try:
                            logger.info("Trying with simpler configuration without quantization")
                            self.hf_pipeline = pipeline(
                                "text-generation",
                                model=settings.HF_MODEL,
                                token=settings.HF_API_KEY,
                                torch_dtype="auto",
                                device_map="auto"
                            )
                        except Exception as e2:
                            error_message2 = str(e2)
                            logger.error(f"Error with simpler configuration: {error_message2}")

                            # If we still have memory issues, switch to TinyLlama
                            if "Invalid buffer size" in error_message2:
                                logger.warning("Memory issue persists. Switching to TinyLlama.")
                                self._initialize_tiny_llama()
                                return
                            else:
                                # Re-raise if it's not a memory issue
                                raise
                else:
                    # Standard settings for smaller models or when sufficient memory is available
                    logger.info(f"Using standard settings for model: {settings.HF_MODEL}")
                    self.hf_pipeline = pipeline(
                        "text-generation",
                        model=settings.HF_MODEL,
                        token=settings.HF_API_KEY,
                        torch_dtype="auto",
                        device_map="auto"
                    )
                logger.info(f"Successfully initialized Hugging Face with model TinyLlama/TinyLlama-1.1B-Chat-v1.0")

                # Also initialize a sentiment analysis pipeline for command processing
                # self.hf_sentiment_pipeline = pipeline(
                #     "sentiment-analysis",
                #     model="distilbert-base-uncased",
                #     token=settings.HF_API_KEY,
                #     truncation=True  # Explicitly enable truncation to avoid warnings
                # )
                logger.info("Successfully initialized Hugging Face sentiment analysis pipeline")
            except Exception as e:
                error_message = str(e)
                logger.error(f"Failed to initialize Hugging Face: {error_message}")

                # Check for specific error types and provide more helpful messages
                if "Invalid buffer size" in error_message:
                    logger.error("Memory error detected. The model is too large for the available memory.")
                    logger.info("Try using a smaller model like 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'")
                    use_fallback = True
                elif "bitsandbytes" in error_message or "quantization" in error_message or "model_kwargs" in error_message or "not used by the model" in error_message:
                    # Provide specific guidance for bitsandbytes errors
                    if "bitsandbytes" in error_message and "pip install" in error_message:
                        logger.error("bitsandbytes package needs to be updated.")
                        logger.info("To fix this issue, run: pip install -U bitsandbytes")
                    else:
                        logger.error("Quantization or model parameter error detected. Trying without quantization.")

                    # Try again without quantization
                    try:
                        logger.info("Attempting to initialize without quantization")
                        self.hf_pipeline = pipeline(
                            "text-generation",
                            model=settings.HF_MODEL,
                            token=settings.HF_API_KEY,
                            torch_dtype="auto",
                            device_map="auto"
                        )
                        logger.info("Successfully initialized Hugging Face without quantization")
                        return
                    except Exception as no_quant_error:
                        error_msg = str(no_quant_error)
                        logger.error(f"Failed to initialize without quantization: {error_msg}")

                        # If we still have memory issues, switch to TinyLlama
                        if "Invalid buffer size" in error_msg:
                            logger.warning("Memory issue detected. Switching to TinyLlama.")
                            self._initialize_tiny_llama()
                            return

                        use_fallback = True
                else:
                    use_fallback = True

                # Try to initialize with a smaller model as a fallback
                if use_fallback:
                    self._initialize_tiny_llama()
                    return

                # Fallback to OpenAI
                self.ai_provider = "openai"
                logger.info("Falling back to OpenAI due to Hugging Face initialization error")

    def set_context(self, client_id: str, context: str) -> None:
        """Set the context for a specific client"""
        self.client_contexts[client_id] = context
        logger.info(f"Context set for client {client_id}: {context[:50]}...")

    def _initialize_tiny_llama(self):
        """Initialize the TinyLlama model as a fallback"""
        try:
            logger.info("Attempting to initialize with TinyLlama model")
            self.hf_pipeline = pipeline(
                "text-generation",
                model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                token=settings.HF_API_KEY,
                torch_dtype="auto",
                device_map="auto"
            )
            logger.info("Successfully initialized Hugging Face with TinyLlama model")
        except Exception as fallback_error:
            error_message = str(fallback_error)
            logger.error(f"Failed to initialize TinyLlama model: {error_message}")

            # Check for bitsandbytes errors
            if "bitsandbytes" in error_message and "pip install" in error_message:
                logger.warning("bitsandbytes package needs to be updated.")
                logger.info("To fix this issue, run: pip install -U bitsandbytes")
                self._try_install_bitsandbytes()

                # Try again after installing bitsandbytes
                try:
                    logger.info("Retrying TinyLlama initialization after bitsandbytes update")
                    self.hf_pipeline = pipeline(
                        "text-generation",
                        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                        token=settings.HF_API_KEY,
                        torch_dtype="auto",
                        device_map="auto"
                    )
                    logger.info("Successfully initialized Hugging Face with TinyLlama model after bitsandbytes update")
                    return
                except Exception as retry_error:
                    logger.error(f"Failed to initialize TinyLlama model after bitsandbytes update: {str(retry_error)}")

            # Fallback to OpenAI as a last resort
            self.ai_provider = "openai"
            logger.info("Falling back to OpenAI due to TinyLlama initialization failure")

    def _try_install_bitsandbytes(self):
        """Try to install or update bitsandbytes package"""
        try:
            logger.info("Attempting to install/update bitsandbytes package")
            import subprocess
            import sys

            # Use the current Python executable to ensure we're installing in the right environment
            python_executable = sys.executable

            # Run pip install command
            result = subprocess.run(
                [python_executable, "-m", "pip", "install", "-U", "bitsandbytes"],
                capture_output=True,
                text=True,
                check=False
            )

            if result.returncode == 0:
                logger.info("Successfully installed/updated bitsandbytes")
                logger.info(f"Output: {result.stdout}")
                return True
            else:
                logger.error(f"Failed to install bitsandbytes: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"Error trying to install bitsandbytes: {str(e)}")
            return False

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

        # Create a new thread for this client if using Assistants API
        if client_id in self.client_threads:
            # We'll create a new thread when the client sends their next message
            del self.client_threads[client_id]
            logger.info(f"Thread cleared for client {client_id}")

        # Clear Hugging Face conversation state
        if client_id in self.hf_conversation_states:
            del self.hf_conversation_states[client_id]
            logger.info(f"Hugging Face conversation state cleared for client {client_id}")

    async def _initialize_assistant(self) -> None:
        """Initialize the OpenAI Assistant"""
        try:
            logger.info("Initializing OpenAI Assistant...")

            # Check if we already have an assistant
            if self.assistant_id:
                logger.info(f"Using existing assistant with ID: {self.assistant_id}")
                return

            # Create a new assistant with enhanced instructions
            enhanced_instructions = f"""
{self.system_prompt}

Additional instructions:
1. Ask the patient 5 questions one at a time.
2. Wait for the patient's response after each question.
3. After the 5th question, when instructed, provide a comprehensive summary of the patient's condition.
4. The summary should include:
   - Chief complaint and symptoms
   - Relevant medical history
   - Current medications
   - Symptom duration and severity
   - Impact on daily activities
5. Be professional, empathetic, and concise.
"""

            # Create the assistant using the current API structure
            assistant = await self.openai_client.beta.assistants.create(
                name="Medical Assistant",
                instructions=enhanced_instructions,
                model=self.model,
            )

            self.assistant_id = assistant.id
            logger.info(f"Successfully created assistant with ID: {self.assistant_id}")
        except Exception as e:
            logger.error(f"Error initializing OpenAI Assistant: {str(e)}")
            logger.error(f"Exception details: {str(e)}")
            # We'll fall back to the regular chat completions API if this fails

    async def process_message(self, message: str, history: List[Dict[str, Any]], client_id: str = None, is_first_message: bool = False) -> str:
        """Process a message using either OpenAI or Hugging Face"""
        try:
            logger.info(f"Processing message with AI provider: {self.ai_provider}")

            if self.ai_provider == "openai":
                logger.info("Using OpenAI for processing")

                # Check if we can use the Assistants API
                if self.assistant_id and settings.USE_ASSISTANTS_API:
                    logger.info("Using OpenAI Assistants API")
                    return await self._process_with_openai_assistant(message, history, client_id, is_first_message)
                else:
                    logger.info("Using OpenAI Chat Completions API")
                    return await self._process_with_openai(message, history, client_id, is_first_message)
            elif self.ai_provider == "huggingface":
                logger.info("Using Hugging Face for processing")

                # Check if we should use the assistant-like stateful approach
                if settings.USE_ASSISTANTS_API:
                    logger.info("Using Hugging Face Assistant-like API")
                    return await self._process_with_huggingface_assistant(message, history, client_id, is_first_message)
                else:
                    logger.info("Using standard Hugging Face processing")
                    return await self._process_with_huggingface(message, history, client_id, is_first_message)
            else:
                logger.error(f"Unknown AI provider: {self.ai_provider}")
                return "Configuration error: Unknown AI provider specified."
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            return f"I'm sorry, I encountered an error processing your request. Please try again later."

    async def _process_with_openai_assistant(self, message: str, history: List[Dict[str, Any]], client_id: str = None, is_first_message: bool = False) -> str:
        """Process message with OpenAI Assistants API"""
        try:
            # Make sure we have an assistant
            if not self.assistant_id:
                logger.warning("Assistant not initialized, falling back to Chat Completions API")
                return await self._process_with_openai(message, history, client_id, is_first_message)

            # Create or get thread for this client
            thread_id = None
            if client_id:
                thread_id = self.client_threads.get(client_id)

            # Create a new thread if needed
            if not thread_id:
                logger.info(f"Creating new thread for client {client_id}")
                thread = await self.openai_client.beta.threads.create()
                thread_id = thread.id

                # Store the thread ID for this client
                if client_id:
                    self.client_threads[client_id] = thread_id

                # If this is the first message and we have context, add it to the thread
                if is_first_message and client_id and client_id in self.client_contexts:
                    context = self.client_contexts[client_id]
                    await self.openai_client.beta.threads.messages.create(
                        thread_id=thread_id,
                        role="user",
                        content=f"Context information: {context}"
                    )

            # Check if this is the 5th question and we need to request a summary
            message_content = message
            if client_id and client_id in self.question_counts and self.question_counts[client_id] == 5:
                # Add a special instruction to provide a summary
                message_content = f"{message}\n\nNow please provide a summary of my medical condition based on all my answers."

            # Add the user's message to the thread
            await self.openai_client.beta.threads.messages.create(
                thread_id=thread_id,
                role="user",
                content=message_content
            )

            # Run the assistant on the thread
            run = await self.openai_client.beta.threads.runs.create(
                thread_id=thread_id,
                assistant_id=self.assistant_id
            )

            # Wait for the run to complete
            while True:
                run_status = await self.openai_client.beta.threads.runs.retrieve(
                    thread_id=thread_id,
                    run_id=run.id
                )

                if run_status.status == "completed":
                    break
                elif run_status.status in ["failed", "cancelled", "expired"]:
                    logger.error(f"Assistant run failed with status: {run_status.status}")
                    return f"I'm sorry, I encountered an error processing your request. Status: {run_status.status}"

                # Wait a bit before checking again
                await asyncio.sleep(0.5)

            # Get the assistant's response
            messages_response = await self.openai_client.beta.threads.messages.list(
                thread_id=thread_id
            )

            # The most recent message should be the assistant's response
            assistant_message = None
            for msg in messages_response.data:
                if msg.role == "assistant":
                    assistant_message = msg
                    break

            if assistant_message and assistant_message.content:
                # Extract the content - handle different content types
                content_text = ""
                for content_item in assistant_message.content:
                    if hasattr(content_item, 'text') and content_item.text:
                        content_text += content_item.text.value

                # Count assistant messages for question tracking
                # Only increment if we haven't reached 5 questions yet
                if client_id:
                    # Initialize question count if not exists
                    if client_id not in self.question_counts:
                        self.question_counts[client_id] = 0

                    # Only increment if we haven't reached 5 questions yet
                    if self.question_counts[client_id] < 5:
                        self.question_counts[client_id] += 1
                        logger.info(f"Question count for client {client_id}: {self.question_counts[client_id]}")

                    # Log if we've reached the summary stage
                    if self.question_counts[client_id] == 5:
                        logger.info(f"Reached 5 questions for client {client_id}, providing summary")

                return content_text

            return "I'm sorry, I couldn't generate a response."
        except Exception as e:
            logger.error(f"Error with OpenAI Assistants API: {str(e)}")
            logger.error(f"Exception details: {str(e)}")
            logger.info("Falling back to Chat Completions API")
            return await self._process_with_openai(message, history, client_id, is_first_message)

    async def _process_with_openai(self, message: str, history: List[Dict[str, Any]], client_id: str = None, is_first_message: bool = False) -> str:
        """Process message with OpenAI

        With the first message, the system prompt goes to AI and socket remains open to maintain context.
        For subsequent messages, only the user's message is sent to AI.
        """
        # Initialize client message history if not exists
        if not hasattr(self, 'client_message_history'):
            self.client_message_history = {}

        if client_id and client_id not in self.client_message_history:
            self.client_message_history[client_id] = []

        # For the first message, send the system prompt to establish context with the AI
        if is_first_message:
            # Format conversation for OpenAI
            messages = []

            # Prepare system content
            system_content = self.system_prompt
            logger.info(f"system_prompt: {system_content}")

            if client_id and client_id in self.client_contexts:
                system_content = f"{system_content}\n\nContext: {self.client_contexts[client_id]}"

            # Initialize question count for this client if not exists
            if client_id and client_id not in self.question_counts:
                self.question_counts[client_id] = 0

            # Add question count to system prompt
            question_count = self.question_counts[client_id]
            if question_count == 5:
                system_content += "\nNow provide a summary of the patient's answers."

            # Add system message to establish context
            messages.append({"role": "system", "content": system_content})

            # Add the current message
            messages.append({"role": "user", "content": message})

            # Store this in client message history
            if client_id:
                self.client_message_history[client_id] = messages.copy()

            logger.info("First message to OpenAI with system prompt and user message")
        else:
            # For subsequent messages, we need to maintain context with OpenAI
            # OpenAI doesn't maintain state between requests, so we need to send the full conversation history
            if client_id and client_id in self.client_message_history:
                # Get the stored message history which includes the system prompt
                messages = self.client_message_history[client_id].copy()

                # Add the current message only
                messages.append({"role": "user", "content": message})
            else:
                # Fallback if we don't have stored history (shouldn't happen, but just in case)
                messages = []

                # Add system message
                system_content = self.system_prompt
                if client_id and client_id in self.client_contexts:
                    system_content = f"{system_content}\n\nContext: {self.client_contexts[client_id]}"

                messages.append({"role": "system", "content": system_content})

                # Add the current message
                messages.append({"role": "user", "content": message})

            logger.info("Subsequent message to OpenAI with only the current message added to context")

        # Count assistant messages in history to track questions asked
        if client_id:
            assistant_messages = [entry for entry in history if entry["role"] == "assistant"]
            self.question_counts[client_id] = len(assistant_messages)
            logger.info(f"Question count for client {client_id}: {self.question_counts[client_id]}")

        try:
            # Call OpenAI API with a 5-minute timeout
            logger.info(f"Message to OpenAI: {messages}")

            response = await asyncio.wait_for(
                self.openai_client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=1000
                ),
                timeout=300  # 5 minutes timeout
            )

            # Add the AI's response to our stored history
            if client_id and client_id in self.client_message_history:
                self.client_message_history[client_id].append({
                    "role": "assistant",
                    "content": response.choices[0].message.content
                })

            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error with OpenAI API: {str(e)}")
            return f"I'm sorry, I encountered an error processing your request. Please try again later."

    async def _process_with_huggingface_assistant(self, message: str, history: List[Dict[str, Any]], client_id: str = None, is_first_message: bool = False) -> str:
        """Process message with Hugging Face in an assistant-like stateful manner

        This method maintains conversation state for each client to avoid sending the full history each time.
        It also tracks question count and provides a summary after the 5th question.
        """
        try:
            # Make sure we have a Hugging Face pipeline
            if not hasattr(self, 'hf_pipeline'):
                logger.warning("Hugging Face pipeline not initialized, falling back to standard processing")
                return await self._process_with_huggingface(message, history, client_id, is_first_message)

            # Get or initialize conversation state for this client
            if client_id and client_id in self.hf_conversation_states:
                # Get existing conversation state
                conversation_state = self.hf_conversation_states[client_id]
            else:
                # Initialize new conversation state with system prompt
                conversation_state = []

                # Add system message to establish context
                system_content = self.system_prompt
                if client_id and client_id in self.client_contexts:
                    system_content = f"{system_content}\n\nContext: {self.client_contexts[client_id]}"

                # Add enhanced instructions for the 5-question format
                system_content += """
                    Additional instructions:
                    1. Ask the patient 5 questions one at a time.
                    2. Wait for the patient's response after each question.
                    3. After the 5th question, provide a comprehensive summary of the patient's condition.
                    4. The summary should include:
                    - Chief complaint and symptoms
                    - Relevant medical history
                    - Current medications
                    - Symptom duration and severity
                    - Impact on daily activities
                    5. Be professional, empathetic, and concise.
                    """

                conversation_state.append({"role": "system", "content": system_content})

                # Initialize question count
                if client_id and client_id not in self.question_counts:
                    self.question_counts[client_id] = 0

            # Check if this is the 5th question and we need to request a summary
            message_content = message
            if client_id and client_id in self.question_counts and self.question_counts[client_id] == 5:
                # Add a special instruction to provide a summary
                message_content = f"{message}\n\nNow please provide a summary of my medical condition based on all my answers."

            # Add the current message to the conversation state
            conversation_state.append({"role": "user", "content": message_content})

            # Prepare messages for the Hugging Face pipeline
            messages = conversation_state.copy()

            # Log the message structure for debugging
            roles_sequence = [msg["role"] for msg in messages]
            logger.info(f"Message roles sequence: {roles_sequence}")

            # Apply the chat template
            prompt = self.hf_pipeline.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            logger.info(f"Sending prompt to Hugging Face: {prompt[:100]}...")

            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            try:
                # Generate response with a timeout
                result = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        lambda: self.hf_pipeline(
                            prompt,
                            max_new_tokens=2048,
                            temperature=0.7,
                            top_p=0.95,
                            do_sample=True,
                            num_return_sequences=1
                        )
                    ),
                    timeout=120  # 2 minutes timeout
                )

                # Extract generated text
                generated_text = result[0]['generated_text']

                # Extract only the assistant's response
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

                # Add the assistant's response to the conversation state
                conversation_state.append({"role": "assistant", "content": response})

                # Store the updated conversation state
                if client_id:
                    self.hf_conversation_states[client_id] = conversation_state

                # Count assistant messages for question tracking
                if client_id:
                    # Only increment if we haven't reached 5 questions yet
                    if client_id not in self.question_counts:
                        self.question_counts[client_id] = 0

                    if self.question_counts[client_id] < 5:
                        self.question_counts[client_id] += 1
                        logger.info(f"Question count for client {client_id}: {self.question_counts[client_id]}")

                    # Log if we've reached the summary stage
                    if self.question_counts[client_id] == 5:
                        logger.info(f"Reached 5 questions for client {client_id}, providing summary")

                return response

            except asyncio.TimeoutError:
                logger.error("Hugging Face request timed out after 2 minutes")
                # Fall back to OpenAI for this request
                logger.info("Falling back to OpenAI due to timeout")
                return await self._process_with_openai(message, history, client_id, is_first_message)

        except Exception as e:
            logger.error(f"Error with Hugging Face assistant: {str(e)}")
            logger.error(f"Exception details: {str(e)}")
            # Fallback to standard processing
            logger.info("Falling back to standard Hugging Face processing")
            return await self._process_with_huggingface(message, history, client_id, is_first_message)

    async def _process_with_huggingface(self, message: str, history: List[Dict[str, Any]], client_id: str = None, is_first_message: bool = False) -> str:
        """Process message with Hugging Face"""
        try:
            # For TinyLlama-1.1B-Chat-v1.0, we'll use its chat template
            # First, prepare the messages in the format expected by the chat template
            messages = []

            # Only include system prompt for the first message in the conversation
            if is_first_message:
                logger.info(f"system_prompt: {self.system_prompt}")
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

            # Process conversation history to ensure roles alternate properly
            # Hugging Face models require strict alternation of user/assistant roles
            filtered_history = []
            last_role = None

            # Filter history to ensure proper alternation
            for entry in history:
                if entry["role"] in ["user", "assistant"]:
                    # Skip consecutive entries with the same role
                    if entry["role"] != last_role or last_role is None:
                        filtered_history.append(entry)
                        last_role = entry["role"]

            # Ensure history starts with a user message and ends with an assistant message
            # This is required for proper alternation when adding the current message
            if filtered_history:
                # If history doesn't start with user, remove the first message
                if filtered_history[0]["role"] != "user":
                    filtered_history = filtered_history[1:]

                # If after filtering we still have messages and the last one is a user message,
                # we need to remove it to maintain alternation when adding the current message
                if filtered_history and filtered_history[-1]["role"] == "user":
                    filtered_history = filtered_history[:-1]

            # Add filtered conversation history
            for entry in filtered_history:
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

            # Log the final message structure for debugging
            roles_sequence = [msg["role"] for msg in messages]
            logger.info(f"Message roles sequence: {roles_sequence}")

            # Apply the chat template
            prompt = self.hf_pipeline.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            logger.info(f"Sending prompt to TinyLlama: {prompt[:100]}...")

            # Run in executor to avoid blocking, with a reduced timeout
            loop = asyncio.get_event_loop()
            try:
                # Reduce timeout from 5 minutes to 2 minutes for better user experience
                result = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        lambda: self.hf_pipeline(
                            prompt,
                            max_new_tokens=2048,  # Generate up to 2048 new tokens
                            temperature=0.7,
                            top_p=0.95,
                            do_sample=True,
                            num_return_sequences=1
                        )
                    ),
                    timeout=120  # 2 minutes timeout (reduced from 5 minutes)
                )
            except asyncio.TimeoutError:
                logger.error("Hugging Face request timed out after 2 minutes")
                # Fall back to OpenAI for this request
                logger.info("Falling back to OpenAI due to timeout")
                return await self._process_with_openai(message, history, client_id, is_first_message)

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

        # Timeout is now handled in the try/except block above
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

                # Log the message structure for debugging
                roles_sequence = [msg["role"] for msg in messages]
                logger.info(f"Summarize message roles sequence: {roles_sequence}")

                # Apply the chat template
                formatted_prompt = self.hf_pipeline.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )

                # Run in executor to avoid blocking with timeout
                loop = asyncio.get_event_loop()
                try:
                    result = await asyncio.wait_for(
                        loop.run_in_executor(
                            None,
                            lambda: self.hf_pipeline(
                                formatted_prompt,
                                max_new_tokens=1024,
                                temperature=0.3,
                                top_p=0.95,
                                do_sample=True,
                                num_return_sequences=1
                            )
                        ),
                        timeout=120  # 2 minutes timeout
                    )
                except asyncio.TimeoutError:
                    logger.error("Hugging Face summarization timed out after 2 minutes")
                    # Fall through to OpenAI implementation
                    raise Exception("Timeout occurred during summarization")

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

        try:
            response = await self.openai_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,
                max_tokens=250
            )
        except Exception as e:
            logger.error(f"Error with OpenAI summarization: {str(e)}")
            return {
                "status": "error",
                "message": "Failed to generate summary"
            }

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

                # Log the message structure for debugging
                roles_sequence = [msg["role"] for msg in messages]
                logger.info(f"Analyze message roles sequence: {roles_sequence}")

                # Apply the chat template
                formatted_prompt = self.hf_pipeline.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )

                # Run in executor to avoid blocking with timeout
                loop = asyncio.get_event_loop()
                try:
                    result = await asyncio.wait_for(
                        loop.run_in_executor(
                            None,
                            lambda: self.hf_pipeline(
                                formatted_prompt,
                                max_new_tokens=350,
                                temperature=0.3,
                                top_p=0.95,
                                do_sample=True,
                                num_return_sequences=1
                            )
                        ),
                        timeout=120  # 2 minutes timeout
                    )
                except asyncio.TimeoutError:
                    logger.error("Hugging Face analysis timed out after 2 minutes")
                    # Fall through to OpenAI implementation
                    raise Exception("Timeout occurred during text analysis")

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

        try:
            response = await self.openai_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,
                max_tokens=350
            )
        except Exception as e:
            logger.error(f"Error with OpenAI analysis: {str(e)}")
            return {
                "status": "error",
                "message": "Failed to analyze text"
            }

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

                # Log the message structure for debugging
                roles_sequence = [msg["role"] for msg in messages]
                logger.info(f"Translate message roles sequence: {roles_sequence}")

                # Apply the chat template
                formatted_prompt = self.hf_pipeline.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )

                # Run in executor to avoid blocking with timeout
                loop = asyncio.get_event_loop()
                try:
                    result = await asyncio.wait_for(
                        loop.run_in_executor(
                            None,
                            lambda: self.hf_pipeline(
                                formatted_prompt,
                                max_new_tokens=min(len(text.split()) * 2, 500),  # Limit max tokens for better performance
                                temperature=0.3,
                                top_p=0.95,
                                do_sample=True,
                                num_return_sequences=1
                            )
                        ),
                        timeout=120  # 2 minutes timeout
                    )
                except asyncio.TimeoutError:
                    logger.error("Hugging Face translation timed out after 2 minutes")
                    # Fall through to OpenAI implementation
                    raise Exception("Timeout occurred during translation")

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

        try:
            response = await self.openai_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,
                max_tokens=500
            )
        except Exception as e:
            logger.error(f"Error with OpenAI translation: {str(e)}")
            return {
                "status": "error",
                "message": "Failed to translate text"
            }

        return {
            "status": "success",
            "translation": response.choices[0].message.content,
            "source_text": text,
            "target_language": target_language
        }

# Create a global AI service instance
ai_service = AIService()

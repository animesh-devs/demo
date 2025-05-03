# AI Chat WebSocket Service

A real-time WebSocket API for chatting with an AI engine using pre-configured syntax and commands.

## Features

- 🔄 **Real-time Communication**: WebSocket-based messaging for instant responses
- 🤖 **AI Integration**: Powered by OpenAI's GPT models or Hugging Face models
- 🔍 **Command System**: Special commands for summarization, analysis, and translation
- 📝 **Context Management**: Set and manage persistent context for AI interactions
- 🔐 **Authentication Support**: Optional JWT-based authentication
- 🌐 **Cross-Origin Support**: Configurable CORS for frontend integration
- 📱 **Simple Test Client**: HTML/JS client for testing the WebSocket API

## Command Syntax

The service supports a command syntax for special operations:

- `/help` - Show available commands
- `/clear` - Clear conversation history
- `/summarize` - Summarize the conversation
- `/analyze text="Your text here"` - Analyze text sentiment and key points
- `/translate text="Your text here" target_language="Spanish"` - Translate text
- `/setcontext text="Your context here"` - Set persistent context for all interactions
- `/getcontext` - View the current context
- `/clearcontext` - Clear the current context

## Getting Started

### Prerequisites

- Python 3.9+
- OpenAI API key (if using OpenAI)
- Hugging Face API key (if using Hugging Face)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ai-chat-websocket.git
   cd ai-chat-websocket
   ```

2. Set up a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and other settings
   # Set AI_PROVIDER to either "openai" or "huggingface"
   ```

5. Run the application:
   ```bash
   uvicorn app.main:app --reload
   ```

### Using Docker

For a containerized setup:

1. Build and run with Docker Compose:
   ```bash
   docker-compose up -d
   ```

2. The API will be available at http://localhost:8000

## WebSocket Endpoints

- `/ws/chat` - Anonymous chat (no authentication)
- `/ws/chat/auth?token=YOUR_JWT_TOKEN` - Authenticated chat

## Message Format

### Client to Server

```json
{
  "message": "Your message or command here"
}
```

### Server to Client

Regular message:
```json
{
  "type": "message",
  "content": "AI response here"
}
```

Command response:
```json
{
  "type": "command_response",
  "command": "summarize",
  "result": {
    "status": "success",
    "summary": "Summary text here"
  }
}
```

System message:
```json
{
  "type": "system",
  "message": "System notification here"
}
```

Error message:
```json
{
  "type": "error",
  "message": "Error description here"
}
```

## Context Management

The service allows setting persistent context that will be included in all AI interactions. This is useful for:

- Giving the AI specific roles or personas (e.g., "You are a medical assistant")
- Providing background information that should be considered in all responses
- Setting constraints or guidelines for AI behavior
- Customizing the AI's knowledge domain

### Using Context Commands

1. **Set Context**:
   ```
   /setcontext text="You are a financial advisor with expertise in retirement planning. Always provide conservative investment advice and consider tax implications."
   ```

2. **View Current Context**:
   ```
   /getcontext
   ```

3. **Clear Context**:
   ```
   /clearcontext
   ```

Context is stored per client and persists across all conversations until explicitly cleared or the server restarts.

## Testing

A simple HTML client is included for testing. Open `client.html` in your browser to test the WebSocket API.

## Project Structure

```
DemoProject/
├── app/
│   ├── api/
│   │   └── websocket.py         # WebSocket API routes
│   ├── services/
│   │   ├── ai_service.py        # AI integration (OpenAI and Hugging Face)
│   │   └── message_processor.py # Message processing
│   ├── utils/
│   │   └── auth.py              # Authentication utilities
│   ├── websockets/
│   │   ├── chat.py              # WebSocket handlers
│   │   └── connection_manager.py # Connection management
│   ├── config.py                # Application configuration
│   └── main.py                  # Application entry point
├── tests/                       # Test suite
├── client.html                  # Simple test client
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Docker configuration
├── docker-compose.yml           # Docker Compose configuration
├── .env.example                 # Example environment variables
└── README.md                    # Project documentation
```

## AI Provider Configuration

The application supports two AI providers:

### OpenAI

To use OpenAI as the AI provider:

1. Set `AI_PROVIDER=openai` in your `.env` file
2. Provide your OpenAI API key as `OPENAI_API_KEY`
3. Optionally specify the model with `AI_MODEL` (defaults to "gpt-4")

### Hugging Face

To use Hugging Face as the AI provider:

1. Set `AI_PROVIDER=huggingface` in your `.env` file
2. Provide your Hugging Face API key as `HF_API_KEY`
3. Optionally specify the model with `HF_MODEL` (defaults to "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

The application uses TinyLlama-1.1B-Chat-v1.0 by default, which is a small but powerful chat model that:
- Is only 1.1B parameters (much smaller than most LLMs)
- Is specifically fine-tuned for chat applications
- Provides good quality responses for a model of its size
- Runs efficiently on limited hardware

Note: The application will automatically fall back to OpenAI if there's an issue initializing the Hugging Face models.

## Troubleshooting

If you encounter issues with the AI provider:

1. Check that your API keys are correctly set in the `.env` file
2. Ensure you have installed all required dependencies with `pip install -r requirements.txt`
3. For Hugging Face, make sure you have PyTorch installed correctly
4. Check the application logs for any error messages

## License

This project is licensed under the MIT License - see the LICENSE file for details.

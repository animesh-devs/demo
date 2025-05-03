import json
import re
from typing import Dict, Any, Tuple, Optional
import logging
from app.services.ai_service import ai_service
from app.websockets.connection_manager import manager
from app.config import settings

logger = logging.getLogger(__name__)

class MessageProcessor:
    def __init__(self):
        # Command pattern: /command param1=value1 param2="value with spaces"
        self.command_pattern = r'^/(\w+)(?:\s+(.*))?$'
        self.param_pattern = r'(\w+)=(?:"([^"]*)"|([\w\d]+))'

    def is_command(self, message: str) -> bool:
        """Check if a message is a command"""
        return message.startswith('/')

    def parse_command(self, message: str) -> Tuple[str, Dict[str, Any]]:
        """Parse a command message into command name and parameters"""
        match = re.match(self.command_pattern, message)
        if not match:
            return "", {}

        command = match.group(1)
        params_str = match.group(2) or ""

        # Parse parameters
        params = {}
        for param_match in re.finditer(self.param_pattern, params_str):
            key = param_match.group(1)
            # Group 2 is for quoted values, group 3 is for unquoted values
            value = param_match.group(2) if param_match.group(2) is not None else param_match.group(3)
            params[key] = value

        return command, params

    async def process_message(self, client_id: str, message: str) -> Dict[str, Any]:
        """Process an incoming message"""
        try:
            # Check if the message is a command
            if self.is_command(message):
                command, params = self.parse_command(message)
                if command:
                    # Check for context management commands
                    if command == "setcontext":
                        return await self._handle_set_context(client_id, params)
                    elif command == "getcontext":
                        return await self._handle_get_context(client_id)
                    elif command == "clearcontext":
                        return await self._handle_clear_context(client_id)

                    # Process other commands
                    history = manager.get_history(client_id)
                    result = await ai_service.process_command(command, params, history)

                    # Add command to history
                    manager.add_to_history(
                        client_id,
                        {"role": "user", "content": message, "type": "command"},
                        settings.MAX_HISTORY_LENGTH
                    )

                    # Add response to history
                    response_content = result.get("message", json.dumps(result))
                    manager.add_to_history(
                        client_id,
                        {"role": "assistant", "content": response_content, "type": "command_response"},
                        settings.MAX_HISTORY_LENGTH
                    )

                    return {
                        "type": "command_response",
                        "command": command,
                        "result": result
                    }
                else:
                    return {
                        "type": "error",
                        "message": "Invalid command format"
                    }
            else:
                # Regular message - process with AI
                history = manager.get_history(client_id)
                
                # Check if this is the first message in the conversation
                is_first_message = len(history) == 0
                logger.info("is_first_message: {is_first_message}")

                # Add user message to history
                manager.add_to_history(
                    client_id,
                    {"role": "user", "content": message, "type": "message"},
                    settings.MAX_HISTORY_LENGTH
                )

                # Get AI response with context
                response = await ai_service.process_message(message, history, client_id, is_first_message)

                # Add AI response to history
                manager.add_to_history(
                    client_id,
                    {"role": "assistant", "content": response, "type": "message"},
                    settings.MAX_HISTORY_LENGTH
                )

                return {
                    "type": "message",
                    "content": response
                }

        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            return {
                "type": "error",
                "message": f"Error processing message: {str(e)}"
            }

    async def _handle_set_context(self, client_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle the setcontext command"""
        context = params.get("text")
        if not context:
            return {
                "type": "error",
                "message": "Missing required parameter 'text'. Usage: /setcontext text=\"Your context here\""
            }

        # Set the context
        ai_service.set_context(client_id, context)

        return {
            "type": "system",
            "message": f"Context set successfully. All future interactions will include this context."
        }

    async def _handle_get_context(self, client_id: str) -> Dict[str, Any]:
        """Handle the getcontext command"""
        context = ai_service.get_context(client_id)

        if context:
            return {
                "type": "system",
                "message": f"Current context:\n\n{context}"
            }
        else:
            return {
                "type": "system",
                "message": "No context is currently set."
            }

    async def _handle_clear_context(self, client_id: str) -> Dict[str, Any]:
        """Handle the clearcontext command"""
        ai_service.clear_context(client_id)

        return {
            "type": "system",
            "message": "Context cleared successfully."
        }

    async def handle_special_commands(self, client_id: str, command: str) -> Dict[str, Any]:
        """Handle special system commands"""
        if command == "/clear":
            manager.clear_history(client_id)
            return {
                "type": "system",
                "message": "Conversation history cleared"
            }
        elif command == "/help":
            return {
                "type": "system",
                "message": """
Available commands:
/clear - Clear conversation history
/help - Show this help message
/summarize - Summarize the conversation
/analyze text="Your text here" - Analyze text sentiment and key points
/translate text="Your text here" target_language="Spanish" - Translate text
/setcontext text="Your context here" - Set a persistent context for all interactions
/getcontext - View the current context
/clearcontext - Clear the current context
                """
            }
        else:
            return None

# Create a global message processor instance
message_processor = MessageProcessor()

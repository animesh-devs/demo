from fastapi import WebSocket, WebSocketDisconnect, HTTPException, status
from typing import Dict, Any, Optional
import json
import uuid
import logging
import asyncio
from datetime import datetime

from app.websockets.connection_manager import manager
from app.services.message_processor import message_processor
from app.utils.auth import verify_token

logger = logging.getLogger(__name__)

async def handle_websocket_connection(websocket: WebSocket):
    """Handle a WebSocket connection for chat"""
    # Generate a unique client ID if not provided
    client_id = str(uuid.uuid4())
    
    try:
        # Accept the connection
        await manager.connect(websocket, client_id)
        
        # Send welcome message
        await manager.send_message(client_id, {
            "type": "system",
            "message": f"Connected to AI Chat Service. Type /help for available commands.",
            "client_id": client_id
        })
        
        # Main message loop
        while True:
            # Receive message from WebSocket
            data = await websocket.receive_text()
            
            try:
                # Parse the message
                message_data = json.loads(data)
                message_text = message_data.get("message", "")
                
                # Check for special system commands
                special_response = await message_processor.handle_special_commands(client_id, message_text)
                if special_response:
                    await manager.send_message(client_id, special_response)
                    continue
                
                # Process the message
                response = await message_processor.process_message(client_id, message_text)
                
                # Send response back to the client
                await manager.send_message(client_id, response)
                
            except json.JSONDecodeError:
                # Handle plain text messages
                response = await message_processor.process_message(client_id, data)
                await manager.send_message(client_id, response)
            
            except Exception as e:
                logger.error(f"Error processing message: {str(e)}")
                await manager.send_message(client_id, {
                    "type": "error",
                    "message": f"Error processing message: {str(e)}"
                })
    
    except WebSocketDisconnect:
        # Handle client disconnection
        manager.disconnect(client_id)
    
    except Exception as e:
        # Handle other exceptions
        logger.error(f"WebSocket error: {str(e)}")
        manager.disconnect(client_id)

async def handle_authenticated_websocket(websocket: WebSocket, token: str):
    """Handle an authenticated WebSocket connection"""
    try:
        # Verify the token
        payload = verify_token(token)
        user_id = payload.get("sub")
        
        if not user_id:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return
        
        # Accept the connection
        await manager.connect(websocket, user_id)
        
        # Send welcome message
        await manager.send_message(user_id, {
            "type": "system",
            "message": f"Connected to AI Chat Service. Type /help for available commands.",
            "user_id": user_id
        })
        
        # Main message loop
        while True:
            # Receive message from WebSocket
            data = await websocket.receive_text()
            
            try:
                # Parse the message
                message_data = json.loads(data)
                message_text = message_data.get("message", "")
                
                # Check for special system commands
                special_response = await message_processor.handle_special_commands(user_id, message_text)
                if special_response:
                    await manager.send_message(user_id, special_response)
                    continue
                
                # Process the message
                response = await message_processor.process_message(user_id, message_text)
                
                # Send response back to the client
                await manager.send_message(user_id, response)
                
            except json.JSONDecodeError:
                # Handle plain text messages
                response = await message_processor.process_message(user_id, data)
                await manager.send_message(user_id, response)
            
            except Exception as e:
                logger.error(f"Error processing message: {str(e)}")
                await manager.send_message(user_id, {
                    "type": "error",
                    "message": f"Error processing message: {str(e)}"
                })
    
    except WebSocketDisconnect:
        # Handle client disconnection
        manager.disconnect(user_id)
    
    except Exception as e:
        # Handle other exceptions
        logger.error(f"WebSocket error: {str(e)}")
        if 'user_id' in locals():
            manager.disconnect(user_id)

from typing import Dict, List, Set, Any, Optional
from fastapi import WebSocket, WebSocketDisconnect
import json
import uuid
import asyncio
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ConnectionManager:
    def __init__(self):
        # Map of client_id -> WebSocket connection
        self.active_connections: Dict[str, WebSocket] = {}
        # Map of client_id -> conversation history
        self.conversation_history: Dict[str, List[Dict[str, Any]]] = {}
        # Map of client_id -> last activity timestamp
        self.last_activity: Dict[str, datetime] = {}
        
    async def connect(self, websocket: WebSocket, client_id: str) -> None:
        """Accept a new WebSocket connection"""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.conversation_history[client_id] = []
        self.last_activity[client_id] = datetime.now()
        logger.info(f"Client {client_id} connected. Active connections: {len(self.active_connections)}")
        
    def disconnect(self, client_id: str) -> None:
        """Remove a WebSocket connection"""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.conversation_history:
            del self.conversation_history[client_id]
        if client_id in self.last_activity:
            del self.last_activity[client_id]
        logger.info(f"Client {client_id} disconnected. Active connections: {len(self.active_connections)}")
    
    async def send_message(self, client_id: str, message: Dict[str, Any]) -> None:
        """Send a message to a specific client"""
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_json(message)
            self.last_activity[client_id] = datetime.now()
    
    def add_to_history(self, client_id: str, message: Dict[str, Any], max_history: int = 10) -> None:
        """Add a message to the conversation history"""
        if client_id not in self.conversation_history:
            self.conversation_history[client_id] = []
        
        self.conversation_history[client_id].append(message)
        
        # Trim history if it exceeds max length
        if len(self.conversation_history[client_id]) > max_history:
            self.conversation_history[client_id] = self.conversation_history[client_id][-max_history:]
    
    def get_history(self, client_id: str) -> List[Dict[str, Any]]:
        """Get the conversation history for a client"""
        return self.conversation_history.get(client_id, [])
    
    def clear_history(self, client_id: str) -> None:
        """Clear the conversation history for a client"""
        if client_id in self.conversation_history:
            self.conversation_history[client_id] = []
    
    async def ping_inactive_connections(self, timeout_seconds: int = 300) -> None:
        """Ping connections that have been inactive for a while"""
        current_time = datetime.now()
        for client_id, last_time in list(self.last_activity.items()):
            time_diff = (current_time - last_time).total_seconds()
            if time_diff > timeout_seconds:
                try:
                    if client_id in self.active_connections:
                        await self.active_connections[client_id].send_json({"type": "ping"})
                        logger.info(f"Sent ping to inactive client {client_id}")
                except Exception as e:
                    logger.error(f"Error pinging client {client_id}: {str(e)}")
                    self.disconnect(client_id)

# Create a global connection manager instance
manager = ConnectionManager()

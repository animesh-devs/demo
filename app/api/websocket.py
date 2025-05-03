from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, Depends, HTTPException
from typing import Optional

from app.websockets.chat import handle_websocket_connection, handle_authenticated_websocket

router = APIRouter()

@router.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for anonymous chat.
    No authentication required.
    """
    await handle_websocket_connection(websocket)

@router.websocket("/ws/chat/auth")
async def authenticated_websocket_endpoint(
    websocket: WebSocket,
    token: Optional[str] = Query(None)
):
    """
    WebSocket endpoint for authenticated chat.
    Requires a valid JWT token as a query parameter.
    """
    if not token:
        await websocket.close(code=1008)  # Policy violation
        return
    
    await handle_authenticated_websocket(websocket, token)

import websocket
import json
import time

# Connect to the WebSocket server
ws = websocket.create_connection("ws://localhost:8003/ws/chat")

# Receive the initial connection message
initial_response = ws.recv()
print(f"Initial response: {initial_response}")

# Wait for the connection to establish
time.sleep(1)

# Send a message
message = {
    "message": "Hello, are you using Llama 3?"
}
ws.send(json.dumps(message))
print("Message sent, waiting for response...")

# Receive the response (this might take some time)
response = ws.recv()
print(f"Received: {response}")

# Close the connection
ws.close()

import websocket
import json
import time

# Connect to the WebSocket server
ws = websocket.create_connection("ws://localhost:8005/ws/chat")

# Receive the initial connection message
initial_response = ws.recv()
print(f"Initial response: {initial_response}")

# Wait for the connection to establish
time.sleep(1)

# Send first message
message = {
    "message": "Hello, I'm not feeling well today."
}
ws.send(json.dumps(message))
print("Message 1 sent, waiting for response...")
response = ws.recv()
print(f"Received response 1: {response}")
time.sleep(2)

# Send second message
message = {
    "message": "I have a headache and fever."
}
ws.send(json.dumps(message))
print("Message 2 sent, waiting for response...")
response = ws.recv()
print(f"Received response 2: {response}")
time.sleep(2)

# Send third message
message = {
    "message": "It started yesterday evening."
}
ws.send(json.dumps(message))
print("Message 3 sent, waiting for response...")
response = ws.recv()
print(f"Received response 3: {response}")
time.sleep(2)

# Send fourth message
message = {
    "message": "I'm taking Tylenol for the pain."
}
ws.send(json.dumps(message))
print("Message 4 sent, waiting for response...")
response = ws.recv()
print(f"Received response 4: {response}")
time.sleep(2)

# Send fifth message
message = {
    "message": "Yes, I can't concentrate on work."
}
ws.send(json.dumps(message))
print("Message 5 sent, waiting for response...")
response = ws.recv()
print(f"Received response 5: {response}")

# Send sixth message to get summary
message = {
    "message": "Can you summarize my condition?"
}
ws.send(json.dumps(message))
print("Message 6 sent, waiting for summary...")
response = ws.recv()
print(f"Received summary: {response}")

# Close the connection
ws.close()

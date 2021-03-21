#! python3
from flask import Flask
from flask_socketio import SocketIO
from inference import ModelManager, TextSampler
import threading

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

connected_clients = set()
connected_clients_lock = threading.Lock()

@socketio.on('new_job')
def new_job():
    pass

@socketio.on('connect')
def on_connect():
    with connected_clients_lock:
        connected_clients.add(request.sid)
    print(f"Client connected! sid={request.sid}")

@socketio.on('disconnect')
def on_disconnect():
    with connected_clients_lock:
        connected_clients.discard(request.sid)
    print(f"Client disconnected! sid={request.sid}")
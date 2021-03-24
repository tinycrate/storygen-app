#! python3
import time
from flask import Flask, request
from flask_socketio import SocketIO
from inference import ModelManager, TextSampler
import threading

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")
model_manager = ModelManager()

connected_clients = {}
connected_clients_lock = threading.RLock()

class ClientInfo:
    def __init__(self):
        self.samplers = {}
        self.lock = threading.RLock()

class SamplerInfo:
    def __init__(self, model_info, generator):
        self.model_info = model_info
        self.generator = generator
        self.lock = threading.RLock()

def get_client(sid):
    client = connected_clients.get(sid, None)
    if not client:
        print(f"WARN: Unable to get client for sid={sid}.")
    return client

@socketio.on('generate_text')
def generate_text(task_name, model_name, prefix, max_length, parameters={}, num=1):
    """
    Generates num number of sequences that is at most max_length tokens long

    Parameters:
        task_name:
            A name chosen by the client to identify the task.
        model_name:
            Name of model to be used
        prefix:
            The text before generation (Input prompt)
        length:
            Length of the text
        parameters:
            A dictionary with additional parameters given to the sampler.

    """
    num = min(num, 10) # Limit max number of sequences to 10
    max_length = min(max_length, 50) # Limit max_length to 50
    client = get_client(request.sid)
    if client == None: return
    with model_manager.use_model(model_name) as model_info:
        sampler = TextSampler(model_info, **parameters)
        result = [sampler.generate_text_atmost(prefix, max_length) for i in range(num)]
    socketio.emit('on_generate_completed',
        (task_name, result,),
        to=request.sid
    )

@socketio.on('start_new_sampler')
def start_new_sampler(sampler_name, model_name, prefix, parameters={}):
    """
    Create a new sampler for a client. Sampling starts immediately.

    Parameters:
        sampler_name:
            A name chosen by the client to identify the sampler.
            Duplicated name will overwrite previous sampler.
        model_name:
            Name of model to be used
        prefix:
            The text before generation (Input prompt)
        parameters:
            A dictionary with additional parameters given to the sampler.
    """
    client = get_client(request.sid)
    if client == None: return
    with client.lock:
        if sampler_name in client.samplers:
            model_manager.free_model(client.samplers[sampler_name].model_info.name)
        model_info = model_manager.acquire_model(model_name)
        sampler = TextSampler(model_info, **parameters)
        client.samplers[sampler_name] = SamplerInfo(model_info, sampler.sample_text(prefix))
    sampler_serve_next(request.sid, sampler_name)

def sampler_serve_next(sid, sampler_name):
    client = get_client(sid)
    if client == None: return
    sampler = client.samplers.get(sampler_name, None)
    if not sampler:
        print(f"WARN: Unable to get sampler {sampler_name} from client {sid}.")
    # Sample as much text in 0.5 seconds
    sampled_text = []
    with sampler.lock:
        start_time = time.monotonic()
        for text in sampler.generator:
            sampled_text.append(text)
            if time.monotonic() - start_time > 0.5:
                break
    if len(sampled_text) > 0:
        socketio.emit('on_text',
            (sampler_name, ''.join(sampled_text),),
            to=sid,
            callback=lambda continue_sampling=False, *args: after_text_sampled(sid, sampler_name, continue_sampling)
        )
    else:
        socketio.emit('on_text_sample_completed',
            sampler_name,
            to=sid
        )
        # Clean up sampler
        with client.lock:
            model_manager.free_model(sampler.model_info.name)
            del client.samplers[sampler_name]

def after_text_sampled(sid, sampler_name, continue_sampling):
    """
    Called when the client has acknoledged the sampled text
    The client should return a value continue_sampling to indicate
    whether the sampling should continue. If not, cleanups should be done
    """
    client = get_client(sid)
    if client == None: return
    if continue_sampling:
        sampler_serve_next(sid, sampler_name)
    else:
        # Clean up sampler
        sampler = client.samplers.get(sampler_name, None)
        if not sampler:
            print(f"WARN: Unable to get sampler {sampler_name} from client {sid}. Unable to cleanup.")
        with client.lock:
            model_manager.free_model(sampler.model_info.name)
            del client.samplers[sampler_name]

@socketio.on('connect')
def on_connect():
    print(f"Client connecting... sid={request.sid}")
    with connected_clients_lock:
        if request.sid in connected_clients:
            print(f"WARN: Client already existed for the same sid ({request.sid}). Might indicate serious memory leak. Attempting to cleanup.")
            _free_client_resources(request.sid)
        connected_clients[request.sid] = ClientInfo()
    print(f"Client connected!    sid={request.sid}")

@socketio.on('disconnect')
def on_disconnect():
    print(f"Client disconnecting... sid={request.sid}")
    with connected_clients_lock:
        if request.sid not in connected_clients:
            print(f"WARN: Client to be disconnected does not exist. Resources cannot be cleaned up.")
        else:
            _free_client_resources(request.sid)
            del connected_clients[request.sid]
    print(f"Client disconnected!    sid={request.sid}")


# This call is destructive, causes the client in an unusable state.
# The client should be removed immediately after the call
def _free_client_resources(sid):
    client = connected_clients[sid]
    with client.lock:
        for sampler in client.samplers.values():
            model_manager.free_model(sampler.model_info.name)
        client.samplers.clear()
        client.samplers = None # Causes further access to samplers crash early as it is not intended
    model_manager.free_resources()

if __name__ == '__main__':
    socketio.run(app, debug=True)

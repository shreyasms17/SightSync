#%%
import zmq
import sys
import uuid
import pickle
#%%
sys.argv = ["", "data/Queries/Videos/video6_1.mp4", "data/Queries/Audios/video6_1.wav"]
assert len(sys.argv) == 3, f"Usage python <name>.py <query-video-path>.mp4 <query-audio-path>.wav"
query_video_path = sys.argv[1]
query_audio_path = sys.argv[2]
#%%
context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5555")
#%%
input_data = dict(
    id=uuid.uuid4().hex,
    query_video_path=query_video_path,
    query_audio_path=query_audio_path
)
#%%
socket.send(pickle.dumps(input_data))
#%%
result = pickle.loads(socket.recv())
print(result)
#%%
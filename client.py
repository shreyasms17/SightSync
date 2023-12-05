#%%
import os
import zmq
import sys
import uuid
import pickle
#%%
context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5555")

sys.argv = ["", "data/Queries/Videos/video1_1.mp4", "data/Queries/Audios/video1_1.wav"]
if len(sys.argv) == 1:
    socket.send(pickle.dumps("quit"))
    sys.exit(0)

#%%
assert len(sys.argv) == 3, f"Usage python <name>.py <query-video-path>.mp4 <query-audio-path>.wav"
query_video_path = sys.argv[1]
query_audio_path = sys.argv[2]

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
#%%
video_name = f"{os.path.basename(result['audio_result'][0][2][0][2])}.mp4"
video_path = os.path.join("data\Videos", video_name)
time = result['audio_result'][0][2][0][3]
#%%
import sys
sys.path.append("src")
from player.media_player import play_video_from
#%%
play_video_from(video_path, time/954)
#%%

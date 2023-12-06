import zmq
import sys
import uuid
import pickle
from random import randint

from os import listdir
from os.path import isfile, join

onlyfiles = [(f.split('.')[0], f.split('.')[0].split('_')[0], f.split('.')[0].split('_')[1]) for f in listdir("data/testing/query_video") if isfile(join("data/testing/query_video", f))]

vid_id = randint(0, 97)
print("TESTING ", onlyfiles[vid_id])

sys.argv = ["", "data/testing/query_video/" + onlyfiles[vid_id][0] + ".mp4", "data/testing/query_audio/" + onlyfiles[vid_id][0] + ".wav"]
assert len(sys.argv) == 3, f"Usage python <name>.py <query-video-path>.mp4 <query-audio-path>.wav"
query_video_path = sys.argv[1]
query_audio_path = sys.argv[2]

context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5555")

input_data = dict(
    id=uuid.uuid4().hex,
    query_video_path=query_video_path,
    query_audio_path=query_audio_path
)

socket.send(pickle.dumps(input_data))

result = pickle.loads(socket.recv())
print(result)

import zmq
import sys
import uuid
import pickle
from random import randint

from os import listdir
from os.path import isfile, join

onlyfiles = [(f.split('.')[0], f.split('.')[0].split('_')[0], f.split('.')[0].split('_')[1]) for f in listdir("data/testing/query_video") if isfile(join("data/testing/query_video", f))]

# vid_id = randint(0, 97)
# print("TESTING ", onlyfiles[vid_id])

for vid_path, media_id, start_time in onlyfiles:
    query_video_path = "data/testing/query_video/" + vid_path + ".mp4"
    query_audio_path = "data/testing/query_audio/" + vid_path + ".wav"
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
    socket.close()
    print(vid_path, result)


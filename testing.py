import zmq
import sys
import uuid
import pickle
from random import randint
import pandas as pd

from os import listdir
from os.path import isfile, join

onlyfiles = [(f.split('.')[0], f.split('.')[0].split('_')[0], f.split('.')[0].split('_')[1]) for f in listdir("data/testing/query_audio") if isfile(join("data/testing/query_audio", f))]

# vid_id = randint(0, 97)
# print("TESTING ", onlyfiles[vid_id])
results = []
for vid_path, media_id, start_time in onlyfiles:
    # query_video_path = "data/testing/query_video/" + vid_path + ".mp4"
    query_audio_path = "data/testing/query_audio/" + vid_path + ".wav"
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:5555")

    input_data = dict(
        id=uuid.uuid4().hex,
        query_video_path="",
        query_audio_path=query_audio_path
    )

    socket.send(pickle.dumps(input_data))

    result = pickle.loads(socket.recv())
    socket.close()
    result["input_name"] = vid_path.split('_')[0]
    result["input_time"] = vid_path.split('_')[1]
    results.append(result)
df = pd.DataFrame.from_dict(results)
df.to_csv("AllTestResults.csv", index=False)


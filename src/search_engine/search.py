import sys
import os
import yaml

sys.path.append("src")
import subprocess

import zmq
import pickle
from pymilvus import db, connections, Collection
from towhee import ops, pipe, DataCollection
from glob import glob
from utils.time_utils import time_it
from feature_extraction.extractor import Extractor
import pandas as pd

from multiprocessing import Process, Queue

database_config = yaml.safe_load(open("config/database_config.yml"))
search_config = yaml.safe_load(open("config/search_config.yml"))
input_config = yaml.safe_load(open("config/input_config.yml"))


@time_it
def initialize_collections():
    global database_config
    connections.connect(alias="search_engine", host=database_config["host"], port=database_config["port"],
                        db_name=database_config["database_name"])
    video_collection = Collection(name=database_config["visual_collection"]["name"], using="search_engine")
    audio_collection = Collection(name=database_config["audio_collection"]["name"], using="search_engine")
    video_collection.load()
    audio_collection.load()
    return video_collection, audio_collection


def get_file(directory):
    for item in glob(directory):
        yield item


def extract_frames(query_video_path):
    global input_config
    extracted_frames_rmdir = [
        "rm", "-rf", input_config["query"]["frame_extraction_dir"]
    ]
    subprocess.run(extracted_frames_rmdir)

    for video_path in get_file(query_video_path):
        video = os.path.basename(video_path)

        extracted_frames_mkdir = [
            "mkdir", "-p", os.path.join(input_config["query"]["frame_extraction_dir"], video.split('.')[0])
        ]
        subprocess.run(extracted_frames_mkdir)
        ffmpeg_command = [
            'ffmpeg',
            '-i', video_path,
            "-vf", "fps=1",
            os.path.join(input_config["query"]["frame_extraction_dir"], video.split('.')[0], 'frame-%10d.jpg')
        ]
        subprocess.run(ffmpeg_command)


def search_server(socket):
    video_collection, audio_collection = initialize_collections()
    print("Server started")
    while True:
        input = pickle.loads(socket.recv())
        print(f"Got input {input}")
        if input == "quit":
            break
        result = search(input, video_collection, audio_collection)
        socket.send(pickle.dumps(result))


def collection_search(query, collection, **kwargs):
    milvus_result = collection.search(
        data=[query],
        **kwargs
    )

    result = []
    for hit in milvus_result[0]:
        row = []
        row.extend([hit.id, hit.score])
        if 'output_fields' in kwargs:
            for k in kwargs['output_fields']:
                row.append(hit.entity.get(k))
        result.append(row)
    return result


@time_it
def search_query_video(query_video_path, video_collection):
    global input_config
    global search_config
    extract_frames(query_video_path)

    search_video = (
        pipe.input('src')
        .flat_map('src', 'img_path', get_file)
        .map('img_path', 'img', ops.image_decode('rgb'))
        .map('img', 'visual_embedding',
             ops.image_embedding.timm(model_name=input_config["towhee_params"]["model"], device=None))
        .map('visual_embedding', 'search_res',
             lambda emb: collection_search(emb, video_collection, **search_config["visual"]))
        .output('search_res')
    )

    return DataCollection(search_video(f"{input_config['query']['frame_extraction_dir']}/*/*.jpg"))


@time_it
def search_query_audio(query_audio_path, audio_collection):
    global input_config
    global database_config
    global search_config

    input_config["query"]["input_audio_path"] = os.path.abspath(query_audio_path)
    extractor_obj = Extractor(input_config, database_config, "query")

    results = []
    for video_id, time, audio_embedding in extractor_obj.audio_feature_generator():
        result = collection_search(audio_embedding, audio_collection, **search_config["audio"])
        results.append((video_id, time, result))
    return results


def get_final_result(video_result, audio_result):
    best_match = None
    min_combined_score = float('inf')

    vid_search_res_list = []
    for i, d in enumerate(video_result.to_list()):
        search_res = d['search_res']

        for item in search_res:
            item.append(i)  # Append the index 'second'
            vid_search_res_list.append(item)

    # Create Pandas DataFrame
    video_df = pd.DataFrame(vid_search_res_list, columns=['hitid', 'score', 'media_id', 'start', 'second'])

    audio_search_res_list = []
    for i, d in enumerate(audio_result):
        search_res = d[2]

        for item in search_res:
            formatted_item = [item[0], item[1], item[2].split('/')[-1], item[-1], i]
            audio_search_res_list.append(formatted_item)

    # Create Pandas DataFrame
    audio_df = pd.DataFrame(audio_search_res_list, columns=['hitid', 'score', 'media_id', 'start', 'second'])

    vd_most_frequent_value = video_df['media_id'].value_counts().idxmax()
    aud_most_frequent_value = audio_df['media_id'].value_counts().idxmax()

    if vd_most_frequent_value == aud_most_frequent_value:
        vid_set = set(video_df[video_df['second'] == 0]['start'].tolist())
        aud_set = set(audio_df[audio_df['second'] == 0]['start'].tolist())
        print(vid_set, aud_set)
        common = vid_set.intersection(aud_set)
        if len(common) > 0:
            return {"media_id": vd_most_frequent_value, "time": sorted(list(common))[0]}
        else:
            # return the earliest time
            return {"media_id": vd_most_frequent_value, "time": sorted(list(vid_set.union(aud_set)))[0]}
    else:
        video_grouped = video_df.groupby(['media_id', 'second'])['start'].apply(set).reset_index()
        audio_grouped = audio_df.groupby(['media_id', 'second'])['start'].apply(set).reset_index()

        # Find matching 'media_id' values between video and audio results
        common_media_ids = set(video_grouped['media_id']).intersection(set(audio_grouped['media_id']))

        # Iterate through the common 'media_id' values and seconds to find potential matches
        for media_id in common_media_ids:
            video_seconds = set(video_grouped[video_grouped['media_id'] == media_id]['second'])
            audio_seconds = set(audio_grouped[audio_grouped['media_id'] == media_id]['second'])

            # Find common seconds between video and audio results for the same 'media_id'
            common_seconds = video_seconds.intersection(audio_seconds)

            if common_seconds:
                # If common seconds found, return the earliest common starting frame
                common_frames = set(video_grouped[(video_grouped['media_id'] == media_id) &
                                                  (video_grouped['second'].isin(common_seconds))]['start'])
                return {"media_id": media_id, "time": sorted(list(common_frames))[0]}

        return {"media_id": video_df.iloc[0]['media_id'], "time": video_df.iloc[0]['start']}

    # print(video_result.to_list()[0])
    # print(audio_result)
    print(len(audio_result), len(video_result.to_list()))
    return {
        "video_result": video_result.to_list(),
        "audio_result": audio_result
    }


@time_it
def search(input, video_collection, audio_collection):
    id = input["id"]
    query_video_path = input["query_video_path"]
    query_audio_path = input["query_audio_path"]

    video_result = search_query_video(query_video_path, video_collection)
    audio_result = search_query_audio(query_audio_path, audio_collection)
    return get_final_result(video_result=video_result, audio_result=audio_result)


if __name__ == "__main__":
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5555")
    search_server(socket)
    # message_queue = MessageQueue()
    # print(hex(id(message_queue)))
    # background_process = Process(target=search_server, args=(message_queue,))
    # background_process.start()
    # background_process.join()  # This will make the main program wait for the background process to finish

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

from multiprocessing import Process, Queue

database_config = yaml.safe_load(open("config/database_config.yml"))
search_config = yaml.safe_load(open("config/search_config.yml"))
input_config = yaml.safe_load(open("config/input_config.yml"))


@time_it
def initialize_collections():
    global database_config
    connections.connect(alias="search_engine", host=database_config["host"], port=database_config["port"], db_name=database_config["database_name"])
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
            'ffmpeg/bin/ffmpeg.exe',
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
        .map('img', 'visual_embedding', ops.image_embedding.timm(model_name=input_config["towhee_params"]["model"], device=None))
        .map('visual_embedding', 'search_res', lambda emb: collection_search(emb, video_collection, **search_config["visual"]))
        .output('search_res')
    )

    return DataCollection(search_video(f"{input_config['query']['frame_extraction_dir']}/*/*.jpg"))

def search_query_audio(query_audio_path, audio_collection):
    pass

def get_final_result(video_result, audio_result):
    print(video_result, type(video_result))
    return video_result.to_list()

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
    
import subprocess
import os
import cv2
import numpy as np
from vector_db_model import CollectionCreator
import zlib
import base64
from towhee import ops

class Extractor:
    def __init__(self, input_config, database_config, type):
        self.input_config = input_config
        self.database_config = database_config
        self.input_video_path = self.input_config[type]["input_video_path"]
        self.visual_collection_name = self.database_config["visual_collection"]["name"]
        self.extracted_frames_dir = self.input_config[type]["frame_extraction_dir"]
        self.type = type

    def extract_visual_frames(self):
        extracted_frames_rmdir = [
            "rm", "-rf", self.extracted_frames_dir
        ]

        subprocess.run(extracted_frames_rmdir)

        for video in os.listdir(self.input_video_path):
            if video.endswith('.mp4'):
                video_absolute_path = os.path.join(self.input_video_path, video)

                extracted_frames_mkdir = [
                    "mkdir", "-p", os.path.join(self.extracted_frames_dir, video.split('.')[0])
                ]

                subprocess.run(extracted_frames_mkdir)

                ffmpeg_command = [
                    'ffmpeg',
                    '-i', video_absolute_path,
                    "-vf", "fps=1",
                    os.path.join(self.extracted_frames_dir, video.split('.')[0], 'frame-%10d.jpg')
                ]
                subprocess.run(ffmpeg_command)
        

    def extract_and_index_visual_features(self):
        collection = CollectionCreator(self.database_config).create_visual_collection()
        i = 1
        data_dic = {'video_id':[], 'frame_id':[], 'visual_embedding': []}
        for video in os.listdir(self.extracted_frames_dir):
            for frame_id in os.listdir(os.path.join(self.extracted_frames_dir, video)):
                if i%50 == 0:
                    mr = collection.insert([data_dic[key] for key in ['video_id', 'frame_id', 'visual_embedding']])
                    data_dic = {'video_id':[], 'frame_id':[], 'visual_embedding': []}
                frame = ops.image_decode("rgb")(os.path.join(self.extracted_frames_dir, video, frame_id))
                visual_embedding = ops.image_embedding.timm(model_name=self.input_config["towhee_params"]["model"], device=None)(frame)
                data_dic['video_id'].append(video)
                data_dic['frame_id'].append(int(frame_id.split('-')[-1].split('.')[0]))
                data_dic['visual_embedding'].append(visual_embedding.tolist())
                i += 1
        if i%50 != 0:
            mr = collection.insert([data_dic[key] for key in ['video_id', 'frame_id', 'visual_embedding']])
        collection.flush()
        collection.create_index("visual_embedding", self.database_config["visual_collection"]["index"])
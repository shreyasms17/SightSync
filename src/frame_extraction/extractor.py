import subprocess
import os
import sys
import time
import cv2
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
import tensorflow as tf
from vector_db_model import CollectionCreator
import zlib
import base64

class Extractor:
    def __init__(self, input_config, database_config, type):
        self.input_config = input_config
        self.database_config = database_config
        self.input_video_path = self.input_config[type]["input_video_path"]
        self.collection_name = self.database_config["collection_name"]
        self.scene_changing_frames_dir = self.input_config[type]["frame_extraction_dir"]
        self.type = type

    def extract_scene_changing_frames(self):
        scene_changing_frames_rmdir = [
            "rm", "-rf", self.scene_changing_frames_dir
        ]

        subprocess.run(scene_changing_frames_rmdir)

        for video in os.listdir(self.input_video_path):
            if video.endswith('.mp4'):
                video_absolute_path = os.path.join(self.input_video_path, video)

                scene_changing_frames_mkdir = [
                    "mkdir", "-p", os.path.join(self.scene_changing_frames_dir, video.split('.')[0])
                ]
                # print(" ".join(scene_changing_frames_mkdir))

                subprocess.run(scene_changing_frames_mkdir)

                threshold = self.input_config[self.type]["scene_changing_threshold"]["default"] if (self.type=="query" or (self.type=="input" and not self.input_config[self.type]["scene_changing_threshold"].__contains__(video.split('.')[0]))) else self.input_config[self.type]["scene_changing_threshold"][video.split('.')[0]]

                ffmpeg_command = [
                    'ffmpeg',
                    '-i', video_absolute_path,
                    '-vf', f"""select='gt(scene,{threshold})'""",
                    '-frame_pts', 'true',
                    '-vsync', 'vfr',
                    os.path.join(self.scene_changing_frames_dir, video.split('.')[0], 'frame-%10d.jpg')
                ]
                subprocess.run(ffmpeg_command)
    
                ffmpeg_command = [
                    'ffmpeg',
                    '-i', video_absolute_path,
                    '-vf', f"""select='eq(pict_type,I)'""",
                    '-frame_pts', 'true',
                    '-vsync', 'vfr',
                    os.path.join(self.scene_changing_frames_dir, video.split('.')[0], 'frame-%10d.jpg')
                ]

                subprocess.run(ffmpeg_command)

                cap = cv2.VideoCapture(os.path.join(self.scene_changing_frames_dir, video))
                cap.set(cv2.CAP_PROP_POS_FRAMES, 1)
                ret, frame1 = cap.read()
                cv2.imwrite(os.path.join(self.scene_changing_frames_dir, video.split('.')[0], f'frame-{"0"*9}1.jpg'), frame1)
        

    def extract_and_index_features(self):
        collection = CollectionCreator(self.database_config).create_collection()
        i = 1
        data_dic = {'video_id':[], 'frame_id':[], 'embedding': [], 'descriptors':[]}
        orb = cv2.ORB_create()
        for video in os.listdir(self.scene_changing_frames_dir):
            for frame_id in os.listdir(os.path.join(self.scene_changing_frames_dir, video)):
                if i%50 == 0:
                    mr = collection.insert([data_dic[key] for key in ['video_id', 'frame_id', 'embedding', 'descriptors']])
                    data_dic = {'video_id':[], 'frame_id':[], 'embedding': [], 'descriptors':[]}
                frame = cv2.imread(os.path.join(self.scene_changing_frames_dir, video, frame_id))
                image_ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
                _, descriptors = orb.detectAndCompute(frame, None)
                hist = cv2.calcHist([image_ycrcb], [0, 1, 2], None, (8,8,8), [0, 256, 0, 256, 0, 256])
                data_dic['video_id'].append(video)
                data_dic['frame_id'].append(int(frame_id.split('-')[-1].split('.')[0]))
                data_dic['embedding'].append(hist.flatten().tolist())
                if descriptors is None:
                    descriptors = np.array([[0 for _ in range(32)]])
                data_dic['descriptors'].append(base64.b64encode(zlib.compress(str(descriptors.tolist()).encode())).decode('utf-8'))
                # reverse: np.array(eval(zlib.decompress(base64.b64decode(compressed_string))))
                i += 1
        if i%50 != 0:
            mr = collection.insert([data_dic[key] for key in ['video_id', 'frame_id', 'embedding', 'descriptors']])
        collection.flush()

        index = {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 1024},
        }
        collection.create_index("embedding", index)





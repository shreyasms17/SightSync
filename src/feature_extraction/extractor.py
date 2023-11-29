import subprocess
import os
import numpy as np
from vector_db_model import CollectionCreator
from towhee import ops
import scipy.io.wavfile as wavfile
from towhee.types.audio_frame import AudioFrame
from glob import glob

class Extractor:
    def __init__(self, input_config, database_config, type):
        self.input_config = input_config
        self.database_config = database_config
        self.input_video_path = self.input_config[type]["input_video_path"]
        self.visual_collection_name = self.database_config["visual_collection"]["name"]
        self.extracted_frames_dir = self.input_config[type]["frame_extraction_dir"]
        self.type = type
        self.video_embedding_model = ops.image_embedding.timm(model_name=self.input_config["towhee_params"]["model"], device=None)
        self.audio_embedding_model = ops.audio_embedding.nnfp()
        self.audio_window_size = self.input_config["audio_params"]["window_size"]

    def extract_visual_frames(self):
        extracted_frames_rmdir = [
            "rm", "-rf", self.extracted_frames_dir
        ]

        subprocess.run(extracted_frames_rmdir)

        for video_path in self.utils.get_files(self.input_video_path):
            video = video_path.split('/')[-1]

            extracted_frames_mkdir = [
                "mkdir", "-p", os.path.join(self.extracted_frames_dir, video.split('.')[0])
            ]

            subprocess.run(extracted_frames_mkdir)

            ffmpeg_command = [
                'ffmpeg',
                '-i', video_path,
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
                visual_embedding = self.video_embedding_model(frame)
                data_dic['video_id'].append(video)
                data_dic['frame_id'].append(int(frame_id.split('-')[-1].split('.')[0]))
                data_dic['visual_embedding'].append(visual_embedding.tolist())
                i += 1
        if i%50 != 0:
            mr = collection.insert([data_dic[key] for key in ['video_id', 'frame_id', 'visual_embedding']])
        collection.flush()
        collection.create_index("visual_embedding", self.database_config["visual_collection"]["index"])
    

    def audio_signal_segmentor(self, frame_rate, audio_signal):
        length_of_audio = audio_signal.shape[0] // frame_rate
        rounded_audio_signal_mono = audio_signal[: length_of_audio * frame_rate, 0].reshape(-1, frame_rate)
        assert rounded_audio_signal_mono.shape == (length_of_audio, frame_rate)
        for audio_signal_second in rounded_audio_signal_mono:
            yield audio_signal_second
    
    def audio_feature_generator(self):
        for audio_file in glob(self.input_config[self.type]["input_audio_path"]):
            frame_rate, signal = wavfile.read(audio_file)
            video_id = audio_file.split('.')[0]
            audio_embedded_second = []
            for segment in self.audio_signal_segmentor(frame_rate, signal):
                audio_embedding = self.audio_embedding_model([AudioFrame(segment, frame_rate, None, 'mono')])
                audio_embedded_second.append(audio_embedding)
            audio_embedded_second = np.array(audio_embedded_second)
            for i in range(len(audio_embedded_second) - self.audio_window_size):
                yield (video_id, i+1, audio_embedded_second[i: i + self.audio_window_size, :].reshape(-1))
    
    def extract_and_index_audio_features(self):
        collection = CollectionCreator(self.database_config).create_audio_collection()
        data_dic = {'video_id':[], 'time':[], 'audio_embedding': []}
        i = 1
        for video_id, time, audio_embedding in self.audio_feature_generator():
            if i%50 == 0:
                mr = collection.insert([data_dic[key] for key in ['video_id', 'time', 'audio_embedding']])
                data_dic = {'video_id':[], 'time':[], 'audio_embedding': []}
            data_dic["video_id"].append(video_id)
            data_dic["time"].append(time)
            data_dic["audio_embedding"].append(audio_embedding)
            i += 1
        if i%50 != 0:
            mr = collection.insert([data_dic[key] for key in ['video_id', 'time', 'audio_embedding']])
        collection.flush()
        collection.create_index("audio_embedding", self.database_config["audio_collection"]["index"])
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


class Selector:

    def __init__(self):
        self.video_metadata = yaml.safe_load(open("config/video_metadata.yml"))

    def select_best_match(self, audio_result, video_result):

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
                return {"media_id": vd_most_frequent_value,
                        "time": sorted(list(common))[0]-1,
                        "media_player_time": sorted(list(common))[0]/self.video_metadata[vd_most_frequent_value]['total_length']}
            else:
                # return the earliest time
                return {"media_id": vd_most_frequent_value,
                        "time": sorted(list(vid_set.union(aud_set)))[0]-1,
                        "media_player_time": sorted(list(vid_set.union(aud_set)))[0]/self.video_metadata[vd_most_frequent_value]['total_length']}
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
                    return {"media_id": media_id,
                            "time": sorted(list(common_frames))[0]-1,
                            "media_player_time": sorted(list(common_frames))[0]/self.video_metadata[video_grouped['media_id']]['total_length']}

            return {"media_id": video_df.iloc[0]['media_id'],
                    "time": video_df.iloc[0]['start']-1,
                    "media_player_time": video_df.iloc[0]['start']/self.video_metadata[video_df.iloc[0]['media_id']]['total_length']}

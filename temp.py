#%%
import yaml
import sys
import os
import time
sys.path.append("src")
from feature_extraction.extractor import Extractor
from towhee import ops, pipe, DataCollection
from pymilvus import db, connections, Collection
from glob import glob

#%%
input_config = yaml.safe_load(open("config/input_config.yml"))
database_config = yaml.safe_load(open("config/database_config.yml"))
search_config = yaml.safe_load(open("config/search_config.yml"))
#%%
sys.argv = ["", "data/Queries/Videos/video1_1.mp4", "data/Queries/Audios/video1_1.wav"]
assert len(sys.argv) == 3, f"Usage python <name>.py <query-video-path>.mp4 <query-audio-path>.wav"
query_video_path = sys.argv[1]
query_audio_path = sys.argv[2]
#%%
print(f"Abs path of input_video: {os.path.abspath(query_video_path)}, audio: {os.path.abspath(query_audio_path)}")
input_config["query"]["input_video_path"] = os.path.abspath(query_video_path)
input_config["query"]["input_audio_path"] = os.path.abspath(query_audio_path)
#%%
start_time = time.time()
extractor_obj = Extractor(input_config, database_config, "query")

extractor_obj.extract_visual_frames()
#%%
def get_file(directory):
        for item in glob(directory):
            yield item

connections.connect(host=database_config["host"], port=database_config["port"])
db.using_database(database_config["database_name"])
collection = Collection(database_config["visual_collection"]["name"])


p_embed = (
    pipe.input('src')
        .flat_map('src', 'img_path', get_file)
        .map('img_path', 'img', ops.image_decode('rgb'))
        .map('img', 'vec', ops.image_embedding.timm(model_name=input_config["towhee_params"]["model"], device=None))
)


p_search = (
      p_embed.map('vec', ('search_res'),  ops.ann_search.milvus_client(
            host=database_config["host"],
            port=database_config["port"],
            limit=search_config["visual"]["limit"],
            anns_field=search_config["visual"]["anns_field"], 
            collection_name=database_config["visual_collection"]["name"],
            **{'output_fields': search_config["visual"]["output_fields"], 'db_name': database_config["database_name"]}
      ))
      .output('img_path', 'search_res')
)

collection.load()
dc = p_search(f"{input_config['query']['frame_extraction_dir']}/*/*.jpg")
result = DataCollection(dc)
print(result)



# %%

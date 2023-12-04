import sys
import os
sys.path.append(os.path.join(os.path.split(os.path.abspath(__file__))[0], "feature_extraction"))
sys.path.append(os.getcwd())

from feature_extraction.extractor import Extractor
from pymilvus import db, connections, Collection
from towhee import ops, pipe, DataCollection
from glob import glob


class Search:
    def __init__(self, input_config, database_config, search_config):
        self.input_config = input_config
        self.database_config = database_config
        self.search_config = search_config
        self.extractor_obj = Extractor(input_config, database_config, "query")

        self.ext_object.extract_visual_frames()
        self.use_database_context()
        

    def use_database_context(self):
        conn = connections.connect(host=self.database_config["host"], port=self.database_config["port"])
        db.using_database(self.database_config["database_name"])

    def get_file(self, directory):
        for item in glob(directory):
            yield item

    def get_most_similar_visual_vectors(self):
        collection = Collection(self.database_config["visual_collection"]["name"])
        collection.load()
        p_embed = (
            pipe.input('src')
                .flat_map('src', 'img_path', self.get_file)
                .map('img_path', 'img', ops.image_decode('rgb'))
                .map('img', 'visual_embedding', ops.image_embedding.timm(model_name="resnet50", device=None))
        )
        p_search_pre = (
            p_embed.map('visual_embedding', 'search_res', ops.ann_search.milvus_client(host=self.database_config["config"], port=self.database_config["port"], 
            anns_field = self.search_config["visual_embedding"], collection_name=self.database_config["visual_collection"]["name"], limit=self.search_config["visual"]["limit"], 
            **{'output_fields': self.search_config["visual"]["output_fields"], 'db_name': self.database_config["database_name"]}))
        )
        p_search = p_search_pre.output('img_path', 'search_res')
        dc = p_search(f"{self.input_config['query']['frame_extraction_dir']}/*/*.jpg")
        visual_vectors = DataCollection(dc)
        return visual_vectors
    
    def get_most_similar_audio_vectors(self):
        collection = Collection(self.database_config["audio_collection"]["name"])
        collection.load()

        audio_search_operator = ops.ann_search.milvus_client(host=self.database_config["config"], port=self.database_config["port"], 
            anns_field = self.search_config["audio_embedding"], collection_name=self.database_config["audio_collection"]["name"], limit=self.search_config["audio"]["limit"], 
            **{'output_fields': self.search_config["audio"]["output_fields"], 'db_name': self.database_config["database_name"]})
        
        results = {}
        for video_id, time, audio_embedding in self.extractor_obj.audio_feature_generator():
            results[(video_id, time)] = audio_search_operator(audio_embedding)
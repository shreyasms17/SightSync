from pymilvus import (
    db,
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)

class CollectionCreator:
    def __init__(self, database_config):
        self.database_config = database_config
        self.visual_collection_name = self.database_config["visual_collection"]["name"]
        self.audio_collection_name = self.database_config["audio_collection"]["name"]
        self.visual_dim = self.database_config["visual_collection"]["embeddings_dim"]
        self.audio_dim = self.database_config["audio_collection"]["embeddings_dim"]
        self.conn = connections.connect(host=self.database_config["host"], port=self.database_config["port"])
        if self.database_config["database_name"] not in db.list_database():
            db.create_database(self.database_config["database_name"])
        db.using_database(self.database_config["database_name"])
    
    def create_visual_collection(self):
        if utility.has_collection(self.visual_collection_name):
            utility.drop_collection(self.visual_collection_name)
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="video_id", dtype=DataType.VARCHAR, max_length = 1000),
            FieldSchema(name="frame_id", dtype=DataType.INT64),
            FieldSchema(name="visual_embedding", dtype=DataType.FLOAT_VECTOR, dim=self.visual_dim)
        ]
        schema = CollectionSchema(
            fields = fields,
            description="Feature store"
        )
        collection = Collection(
            name=self.visual_collection_name,
            schema=schema,
            shards_num=4
        )
        return collection
    

    def create_audio_collection(self):
        if utility.has_collection(self.audio_collection_name):
            utility.drop_collection(self.audio_collection_name)
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="video_id", dtype=DataType.VARCHAR, max_length = 1000),
            FieldSchema(name="time", dtype=DataType.INT64),
            FieldSchema(name="audio_embedding", dtype=DataType.FLOAT_VECTOR, dim=self.audio_dim)
        ]
        schema = CollectionSchema(
            fields = fields,
            description="Feature store"
        )
        collection = Collection(
            name=self.audio_collection_name,
            schema=schema,
            shards_num=4
        )
        return collection
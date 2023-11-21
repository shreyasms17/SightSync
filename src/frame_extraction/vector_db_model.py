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
        self.collection_name = self.database_config["collection_name"]
        self.dim = self.database_config["embeddings_dim"]
        self.conn = connections.connect(host=self.database_config["host"], port=self.database_config["port"])
        if self.database_config["database_name"] not in db.list_database():
            db.create_database(self.database_config["database_name"])
        db.using_database(self.database_config["database_name"])
    
    def create_collection(self):
        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="video_id", dtype=DataType.VARCHAR, max_length = 1000),
            FieldSchema(name="frame_id", dtype=DataType.INT64),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
            FieldSchema(name="descriptors", dtype=DataType.VARCHAR, max_length = 65535)
        ]
        schema = CollectionSchema(
            fields = fields,
            description="Feature store"
        )
        collection = Collection(
            name=self.collection_name,
            schema=schema,
            shards_num=4
        )
    
        return collection
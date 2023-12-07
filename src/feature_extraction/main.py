import sys
sys.path.append("src")
from feature_extraction.extractor import Extractor
import yaml

input_config = yaml.safe_load(open("config/input_config.yml"))
database_config = yaml.safe_load(open("config/database_config.yml"))
type = "input" if ((len(sys.argv)==2 and sys.argv[1]=="input") or len(sys.argv)==1) else "query"

ext_object = Extractor(input_config, database_config, type)
if type == "input":
    # ext_object.extract_visual_frames()
    # ext_object.extract_and_index_visual_features()
    ext_object.extract_and_index_audio_features()
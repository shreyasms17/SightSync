from extractor import Extractor
import sys
import os
import yaml

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

input_config = yaml.safe_load(open("input_config.yml"))
database_config = yaml.safe_load(open("database_config.yml"))
type = "input" if ((len(sys.argv)==2 and sys.argv[1]=="input") or len(sys.argv)==1) else "query"

ext_object = Extractor(input_config, database_config, type)
ext_object.extract_visual_frames()
if type == "input":
    ext_object.extract_and_index_visual_features()
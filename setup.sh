virtualenv -p python3 .
source bin/activate
pip install -r requirements.txt
mkdir -p data/frames_database/
docker-compose up -d
docker port milvus-standalone 19530/tcp
python3 src/frame_extraction/main.py input
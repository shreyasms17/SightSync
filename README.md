# SightSync: Shazam for Video

SightSync is a project that implements a Shazam-like functionality for video and audio clips.  The system takes a database of videos, extracts frames and audio features, and stores them in Milvus vector database collections.  It then allows users to search for a specific video or audio clip, finding the closest matches in both the visual and audio databases.

## Features

1. **Video Processing:**
   - Reads videos from a specified folder.
   - Extracts frames at every second from the videos.
   - Utilizes a ResNet-50 model to obtain frame feature embeddings.
   - Stores feature vectors in a Milvus vector database collection with columns: `video_id`, `frame_id`, and `visual_embedding`.

2. **Audio Processing:**
   - Extracts audio features from a sliding window of 10 seconds for each video using Neural Audio Fingerprint.
   - Stores audio feature vectors in a Milvus vector database collection with columns: `video_id`, `time` (starting in seconds), and `audio_embedding`.

3. **Search and Playback:**
   - Given a video and audio clip, the framework performs a search on the visual and audio database.
   - Finds the closest vectors in both visual and audio collections.
   - Determines the starting second of the matching clip.
   - The server, which contains the loaded collections, performs the search.
   - The client submits a request to the server with the clip file names.
   - A media player plays the clip alongside the original video from the database, starting from the identified second.


## Requirements

- Docker
- Milvus Databasea
- ffmpeg tool
- Python 3.10
- Dependencies listed in `requirements.txt`


## Usage

To use SightSync, follow these steps:

1. **Setup:**
   - Ensure the required dependencies are installed.
   - Set up the Milvus vector database.

2. **Database Population:**
   - Add videos and audios to the specified folders in the input_config.yml under config/ folder.
   - Run the src/feature_extraction/main.py file to extract and store visual and audio features as follows:
     ```python3 src/feature_extraction/main.py input```

3. **Search and Playback:**
   - Start the server to load the collections.
     ```python3 src/search_engine/search.py```
   - Submit a request from the client with the clip file names.
     ```python3 client.py <video file path> <audio file path>```
   - The server performs the search, identifies the starting second, and client initiates playback.
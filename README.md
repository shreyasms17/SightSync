# Overview

SightSync is a project designed to provide a Shazam-like experience for video content. The system takes a database of videos, extracts frames and audio features, and stores them in Milvus vector database collections. It then allows users to search for a specific video or audio clip, finding the closest matches in both the visual and audio databases.

This project was part of the CSCI 576 - Multimedia Systems Design at the University of Southern California.

# Features

- **Video Processing**: Reads videos from a specified folder and extracts frames at every second.

- **Visual Feature Extraction**: Utilizes a ResNet-50 model to extract frame feature embeddings, creating a feature vector.

- **Audio Feature Extraction**: Extracts audio features from a sliding window of 10 seconds in the audio clips, storing them in the Milvus vector database.

- **Milvus Vector Database Integration**: Stores visual and audio feature vectors in Milvus vector database collections with columns for `video_id`, `frame_id`, `visual_embedding` (for visual data), and `video_id`, `time`, `audio_embedding` (for audio data).

- **Search and Matching**: Given a video and audio clip, performs a search on both visual and audio databases, finding the closest vectors in each collection.

- **Playback Integration**: Initiates a media player to play the original video from the specified starting second in the database.

# Requirements

- Docker
- Milvus Databasea
- ffmpeg tool
- Python 3.10
- Dependencies listed in `requirements.txt`
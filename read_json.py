import json
import os

# Read simplified semantic information
with open('outputs/video_name/video_name_semantic_info.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Get core file information
file_info = data['file_info']
output_dir = file_info['output_directory']
video_name = file_info['video_file_name']
tracks_name = file_info['tracks_file_name']

# Build complete paths
video_path = os.path.join(output_dir, video_name)
tracks_path = os.path.join(output_dir, tracks_name)

print(f"Output directory: {output_dir}")
print(f"Video file: {video_path}")
print(f"Tracks file: {tracks_path}")

# Get semantic information
for point_id, info in data['semantic_info'].items():
    if info['has_description']:
        print(f"Point {point_id}: {info['semantic_description']}")
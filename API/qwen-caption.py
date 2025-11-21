import openai
import base64
import os
import json
import sys
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
from pathlib import Path
from collections import defaultdict
import time
import traceback


IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')
VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv')
OPENAI_API_KEY = "Key"
OPENAI_API_BASE = "https://"
client = openai.OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_API_BASE,
)


def check_environment():
    return {
        "python_path": sys.executable,
        "conda_env": os.environ.get('CONDA_DEFAULT_ENV', 'None')
    }

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def generate_image_caption(local_image_path, prompt="What is in the image?"):
    if not os.path.exists(local_image_path):
        raise FileNotFoundError(f"No such files or directory：{local_image_path}")

    base64_image = encode_image(local_image_path)
    try:
        response = client.chat.completions.create(
            model="Qwen2.5-VL-3B-Instruct",
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        raise RuntimeError(f"{str(e)}")

def extract_video_frames(video_path, fps=2.0):
    frames = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"can not open file: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(video_fps / fps) if video_fps > fps else 1
    count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
        count += 1

    cap.release()
    if not frames:
        raise ValueError("cannot extract any frame from video")
    return frames, video_fps

def generate_video_caption(local_video_path, language='英文', fps=2.0):
    if not os.path.exists(local_video_path):
        raise FileNotFoundError(f"{local_video_path}")

    try:
        frames, video_fps = extract_video_frames(local_video_path, fps)
        base64_frames = []
        for frame in frames:
            buffer = BytesIO()
            frame.save(buffer, format="jpeg")
            base64_frames.append(base64.b64encode(buffer.getvalue()).decode("utf-8"))

        response = client.chat.completions.create(
            model="Qwen2.5-VL-3B-Instruct",
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": [
                    {"type": "text", "text": f"用{language}详细描述视频内容, 输出完整段落"},
                    {"type": "video_url", "video_url": {"url": f"data:video/jpeg;base64,{','.join(base64_frames)}"}}
                ]}
            ],
            extra_body={"mm_processor_kwargs": {"fps": [fps]}}
        )
        return response.choices[0].message.content, video_fps
    except Exception as e:
        raise RuntimeError(f"failed to generate caption for video: {str(e)}")

def get_file_type(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext in IMAGE_EXTENSIONS:
        return "image"
    elif ext in VIDEO_EXTENSIONS:
        return "video"
    return None

def generate_id(total_count):
    return f"{total_count + 1:05d}"

def get_bucket(id_str):
    return f"{int(id_str) // 1000:03d}"

def process_image(file_path, output_root, total_count, language):
    try:
        img_id = generate_id(total_count)
        bucket = get_bucket(img_id)
        img_dest_dir = os.path.join(output_root, "images", bucket)
        anno_dest_dir = os.path.join(output_root, "labels", bucket)
        os.makedirs(img_dest_dir, exist_ok=True)
        os.makedirs(anno_dest_dir, exist_ok=True)

        dest_path = os.path.join(img_dest_dir, f"{img_id}{os.path.splitext(file_path)[1].lower()}")
        with open(file_path, 'rb') as src, open(dest_path, 'wb') as dst:
            dst.write(src.read())
        file_size = os.path.getsize(dest_path) / (1024 **2)
      
        caption = generate_image_caption(file_path, prompt=f"Describe the image in {language}")
        with Image.open(file_path) as img:
            width, height = img.size
            mod = img.mode

        anno = {
            "image_id": img_id,
            "width": width,
            "height": height,
            "mode": mode,
            "source_url": "",
            "annotations": {
                "caption": caption,
                "spatial": [],
                "relations": [],
                "scene_graph": {}
            }
        }

        anno_path = os.path.join(anno_dest_dir, f"{img_id}.json")
        with open(anno_path, 'w', encoding='utf-8') as f:
            json.dump(anno, f, ensure_ascii=False, indent=2)

        return {
            "success": True,
            "id": img_id,
            "path": dest_path,
            "annotation_path": anno_path,
            "width": width,
            "height": height,
            "mode": mode,
            "format": os.path.splitext(file_path)[1].lower(),
            "format": os.path.splitext(file_path)[1].lower()
        }
    except Exception as e:
        return {
            "success": False,
            "path": file_path,
            "error": str(e)
        }

def process_video(file_path, output_root, total_count, language):
    try:
        video_id = generate_id(total_count)
        bucket = get_bucket(video_id)
        video_dest_dir = os.path.join(output_root, "videos", bucket)
        anno_dest_dir = os.path.join(output_root, "annotations", bucket)
        os.makedirs(video_dest_dir, exist_ok=True)
        os.makedirs(anno_dest_dir, exist_ok=True)

        dest_path = os.path.join(video_dest_dir, f"{video_id}{os.path.splitext(file_path)[1].lower()}")
        with open(file_path, 'rb') as src, open(dest_path, 'wb') as dst:
            dst.write(src.read())

        caption, video_fps = generate_video_caption(dest_path, language=language)
        cap = cv2.VideoCapture(dest_path)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frame_count / video_fps if video_fps > 0 else 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        file_size = os.path.getsize(dest_path) / (1024 ** 3)
        cap.release()

        anno = {
            "video_id": video_id,
            "duration": round(duration, 2),
            "fps": round(video_fps, 2),
            "width": width,
            "height": height,
            "source_url": "",
            "annotations": {
                "caption": caption,
                "temporal": [],
                "events": [],
                "camera_motion": {},
                "scene_graph_sequence": []
            }
        }

        anno_path = os.path.join(anno_dest_dir, f"{video_id}.json")
        with open(anno_path, 'w', encoding='utf-8') as f:
            json.dump(anno, f, ensure_ascii=False, indent=2)

        return {
            "success": True,
            "id": video_id,
            "path": dest_path,
            "annotation_path": anno_path,
            "duration": duration,
            "fps": video_fps,
            "width": width,
            "height": height,
            "format": os.path.splitext(file_path)[1].lower(),
            "file_size_gb": file_size
        }
    except Exception as e:
        return {
            "success": False,
            "path": file_path,
            "error": str(e)
        }

def collect_files(input_dir):
    files = []
    for root, _, file_names in os.walk(input_dir):
        for file_name in file_names:
            file_path = os.path.join(root, file_name)
            file_type = get_file_type(file_path)
            if file_type:
                files.append((file_path, file_type))
    return files

def update_metadata(output_root, processed_files):
    images = [f for f in processed_files if f["type"] == "image" and f["success"]]
    image_count = len(images)
    
    total_storage = sum(image.get("file_size_mb", 0) for image in images)
    avg_size = total_storage / image_count if image_count > 0 else 0
    
    format_counts = defaultdict(int)
    for image in images:
        fmt = image.get("format", "unknown")
        format_counts[fmt] += 1
    
    resolution_counts = defaultdict(int)
    for image in images:
        res = f"{image.get('width', 0)}x{image.get('height', 0)}"
        resolution_counts[res] += 1
    
    mode_counts = defaultdict(int)
    for image in images:
        mode = image.get("mode", "unknown")
        mode_counts[mode] += 1
    
    image_meta = {
        "dataset": {
            "name": "custom_image_dataset",
            "type": "image_captioning",
            "path": str(Path(output_root).resolve()),
            "image_dir": "images/",
            "annotation_dir": "labels/",
            "metadata_file": "metadata.json"
        },
        "data_stats": {
            "total_images": image_count,
            "total_storage_size_mb": round(total_storage, 4),
            "avg_file_size_mb": round(avg_size, 4) if image_count > 0 else 0,
            "image_formats": dict(format_counts),
            "resolution_distribution": dict(resolution_counts),
            "mode_distribution": dict(mode_counts)  
        },
        "data_format": {
            "image_extensions": list(IMAGE_EXTENSIONS),
            "supported_annotations": ["caption", "spatial", "relations", "scene_graph"]
        },
        "processing_info": {
            "generated_captions": True,
            "processing_date": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        }
    }
    with open(os.path.join(output_root, "metadata.json"), 'w', encoding='utf-8') as f:
        json.dump(image_meta, f, ensure_ascii=False, indent=2)

    videos = [f for f in processed_files if f["type"] == "video" and f["success"]]
    video_count = len(videos)
    
    total_duration = sum(video.get("duration", 0) for video in videos)
    total_duration_hours = total_duration / 3600  
    avg_duration = total_duration / video_count if video_count > 0 else 0
    
    format_counts = defaultdict(int)
    for video in videos:
        fmt = video.get("format", "unknown")
        format_counts[fmt] += 1
    
    resolution_counts = defaultdict(int)
    for video in videos:
        res = f"{video.get('width', 0)}x{video.get('height', 0)}"
        resolution_counts[res] += 1
    
    total_storage = sum(video.get("file_size_gb", 0) for video in videos)
    
    fps_ranges = defaultdict(int)
    for video in videos:
        fps = video.get("fps", 0)
        if fps < 10:
            fps_ranges["<10fps"] += 1
        elif fps < 25:
            fps_ranges["10-24fps"] += 1
        elif fps < 30:
            fps_ranges["25-29fps"] += 1
        else:
            fps_ranges["30+fps"] += 1

    os.makedirs(os.path.join(output_root, "meta_data"), exist_ok=True)
    stats = {
        "dataset": {
            "name": "custom_video_dataset",
            "type": "video_captioning",
            "path": str(Path(output_root).resolve()),
            "video_dir": "videos/",
            "annotation_dir": "annotations/",
            "metadata_file": "meta_data/stats.json"
        },
        "data_stats": {
            "total_videos": video_count,
            "total_duration_seconds": round(total_duration, 2),
            "total_duration_hours": round(total_duration_hours, 2),
            "avg_video_length_seconds": round(avg_duration, 2),
            "video_formats": dict(format_counts),
            "resolution_distribution": dict(resolution_counts),
            "fps_distribution": dict(fps_ranges),
            "total_storage_size_gb": round(total_storage, 4),
            "avg_file_size_gb": round(total_storage / video_count, 4) if video_count > 0 else 0
        },
        "data_format": {
            "video_extensions": list(VIDEO_EXTENSIONS),
            "supported_annotations": ["caption", "temporal", "events", "camera_motion"]
        },
        "processing_info": {
            "generated_captions": True,
            "extracted_frames": True,
            "processing_date": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        }
    }
    with open(os.path.join(output_root, "meta_data", "stats.json"), 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    splits = {
        "train": {"start_idx": 0, "end_idx": 0, "count": 0, "path_pattern": ""},
        "validation": {"start_idx": 0, "end_idx": 0, "count": 0, "path_pattern": ""},
        "test": {"start_idx": 0, "end_idx": 0, "count": 0, "path_pattern": ""}
    }
    with open(os.path.join(output_root, "meta_data", "splits.json"), 'w', encoding='utf-8') as f:
        json.dump(splits, f, ensure_ascii=False, indent=2)

def main():
    result = {
        "success": False,
        "output_path": None,
        "environment_info": check_environment(),
        "error": None,
        "details": {
            "processed_files": [],
            "skipped_files": [],
            "errors": []
        }
    }

    try:

        input_file = ""
        output_file = ""

        if not os.path.exists(input_file):
            raise FileNotFoundError(f"{input_file}")

        files = collect_files(input_dir)
        if not files:
            raise ValueError(f"{input_dir}")

        processed = []
        for idx, (file_path, file_type) in enumerate(files):
            if file_type == "image":
                res = process_image(file_path, output_root, idx, language)
            else:
                res = process_video(file_path, output_root, idx, language)
            
            res["type"] = file_type
            processed.append(res)
            print(f"idx: {idx}, file_type: {file_type}")
            print(f"res: {res}")

        result["success"] = all(item["success"] for item in processed)
        result["output_path"] = output_root
        result["details"]["processed_files"] = [
            {"id": item["id"], "path": item["path"], "type": item["type"]}
            for item in processed if item["success"]
        ]
        result["details"]["errors"] = [
            {"path": item["path"], "error": item["error"], "type": item["type"]}
            for item in processed if not item["success"]
        ]

        update_metadata(output_root, processed)
        print(f"result: {result}")

    except Exception as e:
        result["error"] = str(e)
        print(f"{str(e)}", file=sys.stderr)

if __name__ == "__main__":
    main()

'''
1.使用 Qwen2.5-VL 模型识别图像和视频中的名人、地标和电影角色
    在原有标注文档的annotations字段下添加新字段
    如果找不到，创建新的标注文档（遵循原脚本格式）并添加实体信息
2.脚本可以独立运行，也可以与原有的qwen-caption.py配合使用，先运行原脚本生成基础标注，再运行此脚本添加实体识别信息。
'''
import os
import json
import sys
import base64
import cv2
import shutil
import numpy as np
from PIL import Image
from io import BytesIO
from pathlib import Path
import time
import traceback

# 配置参数
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')
VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv')
OPENAI_API_KEY = "Key"  # 请替换为你的API密钥
OPENAI_API_BASE = "https://"  # 请替换为你的API基础地址
MODEL = "model name"

# 初始化OpenAI客户端
client = openai.OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_API_BASE,
)

def check_environment():
    """检查并返回环境信息"""
    return {
        "python_path": sys.executable,
        "conda_env": os.environ.get('CONDA_DEFAULT_ENV', 'None')
    }

def encode_image(image_path):
    """将图片编码为base64格式"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def extract_video_frames(video_path, fps=2.0):
    """从视频中提取帧"""
    frames = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")

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
        raise ValueError("无法从视频中提取任何帧")
    return frames, video_fps

def identify_entities(image_path=None, video_path=None, frames=None, language="英文"):
    """
    识别图像或视频中的实体（名人、地标、电影角色）

    Args:
        image_path: 图像路径
        video_path: 视频路径
        frames: 视频帧列表（如果已提取）
        language: 输出语言

    Returns:
        实体识别结果字符串
    """
    try:
        messages = [
            {"role": "system", "content": "你是一个图像识别专家，能够准确识别图像中的名人、地标和电影角色。"}
        ]

        user_content = []

        # 添加文本提示
        prompt_text = f"识别所有名人、地标和电影角色，如果都没有则输出None"
        user_content.append({"type": "text", "text": prompt_text})

        # 添加图像或视频
        if image_path:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"图像文件不存在：{image_path}")
            base64_image = encode_image(image_path)
            user_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}})
        elif video_path or frames:
            if not frames:
                frames, video_fps = extract_video_frames(video_path, fps=1.0)

            base64_frames = []
            for frame in frames:
                buffer = BytesIO()
                frame.save(buffer, format="jpeg")
                base64_frames.append(base64.b64encode(buffer.getvalue()).decode("utf-8"))

            user_content.append({
                "type": "video_url",
                "video_url": {"url": f"data:video/jpeg;base64,{','.join(base64_frames)}"}
            })
        else:
            raise ValueError("必须提供图像路径、视频路径或视频帧")

        messages.append({"role": "user", "content": user_content})

        # 调用API
        response = client.chat.completions.create(
            model="Qwen2.5-VL-3B-Instruct",
            messages=messages,
            extra_body={"mm_processor_kwargs": {"fps": [1.0]}} if (video_path or frames) else {}
        )

        return response.choices[0].message.content

    except Exception as e:
        raise RuntimeError(f"实体识别失败: {str(e)}")

def get_file_type(file_path):
    """判断文件类型（图像或视频）"""
    ext = os.path.splitext(file_path)[1].lower()
    if ext in IMAGE_EXTENSIONS:
        return "image"
    elif ext in VIDEO_EXTENSIONS:
        return "video"
    return None

def find_annotation_file(file_path, root_dir):
    """查找媒体文件对应的标注文件"""
    file_type = get_file_type(file_path)
    if not file_type:
        return None

    # 检查文件是否已经在输出目录中（即是否是之前处理过的文件）
    # 如果是，直接用其文件名作为file_id
    if file_path.startswith(os.path.join(root_dir, "images")) or file_path.startswith(os.path.join(root_dir, "videos")):
        file_name = os.path.basename(file_path)
        file_id = os.path.splitext(file_name)[0]
        if file_id.isdigit():
            bucket = f"{int(file_id) // 1000:03d}"
            if file_type == "image":
                anno_path = os.path.join(root_dir, "labels", bucket, f"{file_id}.json")
            else:
                anno_path = os.path.join(root_dir, "annotations", bucket, f"{file_id}.json")
            if os.path.exists(anno_path):
                return anno_path

    # 否则，按原始文件名查找（适用于处理已有标注的数据集）
    file_name = os.path.basename(file_path)
    file_id = os.path.splitext(file_name)[0]

    # 增加健壮性检查：确保 file_id 是有效的整数
    if not file_id.isdigit():
        # 如果文件名不是数字，则无法按照现有规则查找标注文件
        return None

    bucket = f"{int(file_id) // 1000:03d}"

    # 标注文件可能存在的位置
    possible_paths = []
    if file_type == "image":
        possible_paths.append(os.path.join(root_dir, "labels", bucket, f"{file_id}.json"))
    else:  # video
        possible_paths.append(os.path.join(root_dir, "annotations", bucket, f"{file_id}.json"))

    # 检查可能的路径
    for path in possible_paths:
        if os.path.exists(path):
            return path

    return None

def copy_and_rename_media_file(source_path, dest_dir, file_id):
    """
    将媒体文件复制到目标目录并重命名

    Args:
        source_path: 源文件路径
        dest_dir: 目标目录
        file_id: 新的文件ID（不含扩展名）

    Returns:
        新的文件路径
    """
    ext = os.path.splitext(source_path)[1]
    bucket = f"{int(file_id) // 1000:03d}"
    dest_path = os.path.join(dest_dir, bucket, f"{file_id}{ext}")
    os.makedirs(dest_dir, exist_ok=True)
    shutil.copy2(source_path, dest_path)  # 使用copy2保留文件元数据
    return dest_path

def get_max_id_in_dir(root_dir, file_type):
    """
    在指定的根目录下，查找特定类型文件（图片/视频）已存在的最大ID。
    ID是标注文件的文件名（不含扩展名），格式为00001, 00002等。

    Args:
        root_dir (str): 数据集根目录。
        file_type (str): 文件类型，"image" 或 "video"。

    Returns:
        int: 找到的最大ID，如果没有则返回-1。
    """
    if file_type == "image":
        search_dir = os.path.join(root_dir, "labels")
    elif file_type == "video":
        search_dir = os.path.join(root_dir, "annotations")
    else:
        return -1

    max_id = -1

    # 遍历所有分桶目录
    if os.path.exists(search_dir):
        for bucket_dir_name in os.listdir(search_dir):
            bucket_path = os.path.join(search_dir, bucket_dir_name)
            if os.path.isdir(bucket_path):
                # 遍历分桶目录下的所有.json文件
                for filename in os.listdir(bucket_path):
                    if filename.endswith('.json'):
                        file_id_str = os.path.splitext(filename)[0]
                        if file_id_str.isdigit():
                            current_id = int(file_id_str)
                            if current_id > max_id:
                                max_id = current_id

    return max_id

def create_new_annotation(file_path, root_dir, total_count):
    """创建新的标注文件（如果不存在），并复制媒体文件到相应位置"""
    try:
        file_type = get_file_type(file_path)
        if not file_type:
            raise ValueError(f"不支持的文件类型: {file_path}")

        file_id = f"{total_count:05d}"
        bucket = f"{int(file_id) // 1000:03d}"

        # 复制媒体文件到输出目录
        if file_type == "image":
            media_dest_dir = os.path.join(root_dir, "images")
            new_media_path = copy_and_rename_media_file(file_path, media_dest_dir, file_id)
            anno_dest_dir = os.path.join(root_dir, "labels", bucket)
            with Image.open(new_media_path) as img:
                width, height = img.size
                mode = img.mode

            anno = {
                "image_id": file_id,
                "width": width,
                "height": height,
                "mode": mode,
                "source_url": "",
                "annotations": {
                    "caption": "",
                    "spatial": [],
                    "relations": [],
                    "scene_graph": {},
                    "entity": []  # 统一使用entity字段
                }
            }
        else:  # video
            media_dest_dir = os.path.join(root_dir, "videos")
            new_media_path = copy_and_rename_media_file(file_path, media_dest_dir, file_id)
            anno_dest_dir = os.path.join(root_dir, "annotations", bucket)
            cap = cv2.VideoCapture(new_media_path)
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            duration = frame_count / video_fps if video_fps > 0 else 0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            anno = {
                "video_id": file_id,
                "duration": round(duration, 2),
                "fps": round(video_fps, 2),
                "width": width,
                "height": height,
                "source_url": "",
                "annotations": {
                    "caption": "",
                    "temporal": [],
                    "events": [],
                    "camera_motion": {},
                    "scene_graph_sequence": [],
                    "entity": []  # 统一使用entity字段
                }
            }

        os.makedirs(anno_dest_dir, exist_ok=True)
        anno_path = os.path.join(anno_dest_dir, f"{file_id}.json")

        with open(anno_path, 'w', encoding='utf-8') as f:
            json.dump(anno, f, ensure_ascii=False, indent=2)

        return anno_path, file_id, new_media_path

    except Exception as e:
        raise RuntimeError(f"创建新标注文件失败: {str(e)}")

def ensure_metadata_exists(root_dir):
    """确保元数据文件存在，如果不存在则创建"""
    # 确保meta_data目录存在
    meta_dir = os.path.join(root_dir, "meta_data")
    os.makedirs(meta_dir, exist_ok=True)

    # 确保images和videos目录存在
    os.makedirs(os.path.join(root_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(root_dir, "videos"), exist_ok=True)

    # 检查并创建图片元数据文件
    image_meta_path = os.path.join(root_dir, "metadata.json")
    if not os.path.exists(image_meta_path):
        image_meta = {
            "dataset": {
                "name": "custom_image_dataset",
                "type": "image_captioning",
                "path": str(Path(root_dir).resolve()),
                "image_dir": "images/",
                "annotation_dir": "labels/",
                "metadata_file": "metadata.json"
            },
            "data_stats": {
                "total_images": 0,
                "total_storage_size_mb": 0,
                "avg_file_size_mb": 0,
                "image_formats": {},
                "resolution_distribution": {},
                "mode_distribution": {}
            },
            "data_format": {
                "image_extensions": list(IMAGE_EXTENSIONS),
                "supported_annotations": ["caption", "spatial", "relations", "scene_graph", "entity"]
            },
            "processing_info": {
                "generated_captions": True,
                "processing_date": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            }
        }
        with open(image_meta_path, 'w', encoding='utf-8') as f:
            json.dump(image_meta, f, ensure_ascii=False, indent=2)

    # 检查并创建视频元数据文件
    video_meta_path = os.path.join(meta_dir, "stats.json")
    if not os.path.exists(video_meta_path):
        video_meta = {
            "dataset": {
                "name": "custom_video_dataset",
                "type": "video_captioning",
                "path": str(Path(root_dir).resolve()),
                "video_dir": "videos/",
                "annotation_dir": "annotations/",
                "metadata_file": "meta_data/stats.json"
            },
            "data_stats": {
                "total_videos": 0,
                "total_duration_seconds": 0,
                "total_duration_hours": 0,
                "avg_video_length_seconds": 0,
                "video_formats": {},
                "resolution_distribution": {},
                "fps_distribution": {},
                "total_storage_size_gb": 0,
                "avg_file_size_gb": 0
            },
            "data_format": {
                "video_extensions": list(VIDEO_EXTENSIONS),
                "supported_annotations": ["caption", "temporal", "events", "camera_motion", "entity"]
            },
            "processing_info": {
                "generated_captions": True,
                "extracted_frames": True,
                "processing_date": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            }
        }
        with open(video_meta_path, 'w', encoding='utf-8') as f:
            json.dump(video_meta, f, ensure_ascii=False, indent=2)

    # 检查并创建splits文件
    splits_path = os.path.join(meta_dir, "splits.json")
    if not os.path.exists(splits_path):
        splits = {
            "train": {"start_idx": 0, "end_idx": 0, "count": 0, "path_pattern": ""},
            "validation": {"start_idx": 0, "end_idx": 0, "count": 0, "path_pattern": ""},
            "test": {"start_idx": 0, "end_idx": 0, "count": 0, "path_pattern": ""}
        }
        with open(splits_path, 'w', encoding='utf-8') as f:
            json.dump(splits, f, ensure_ascii=False, indent=2)

def update_metadata_supported_annotations(root_dir):
    """更新元数据文件，添加entity到supported_annotations"""
    # 更新图片元数据
    image_meta_path = os.path.join(root_dir, "metadata.json")
    if os.path.exists(image_meta_path):
        with open(image_meta_path, 'r', encoding='utf-8') as f:
            image_meta = json.load(f)

        if "entity" not in image_meta["data_format"]["supported_annotations"]:
            image_meta["data_format"]["supported_annotations"].append("entity")
            # 更新处理时间
            image_meta["processing_info"]["processing_date"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

            with open(image_meta_path, 'w', encoding='utf-8') as f:
                json.dump(image_meta, f, ensure_ascii=False, indent=2)

    # 更新视频元数据
    video_meta_path = os.path.join(root_dir, "meta_data", "stats.json")
    if os.path.exists(video_meta_path):
        with open(video_meta_path, 'r', encoding='utf-8') as f:
            video_meta = json.load(f)

        if "entity" not in video_meta["data_format"]["supported_annotations"]:
            video_meta["data_format"]["supported_annotations"].append("entity")
            # 更新处理时间
            video_meta["processing_info"]["processing_date"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

            with open(video_meta_path, 'w', encoding='utf-8') as f:
                json.dump(video_meta, f, ensure_ascii=False, indent=2)

def update_metadata_statistics(root_dir):
    """
    根据处理后的实际数据，更新元数据文件中的统计信息。
    """
    import os
    from pathlib import Path

    # --- 更新图片元数据 (metadata.json) ---
    image_meta_path = os.path.join(root_dir, "metadata.json")
    if os.path.exists(image_meta_path):
        with open(image_meta_path, 'r', encoding='utf-8') as f:
            image_meta = json.load(f)

        image_dir = os.path.join(root_dir, "images")
        total_images = 0
        total_storage_size_mb = 0
        image_formats = {}
        resolution_distribution = {}

        if os.path.exists(image_dir):
            # 遍历所有图片分桶目录
            for bucket_dir in Path(image_dir).rglob('*'):
                if bucket_dir.is_dir():
                    for image_file in bucket_dir.iterdir():
                        if image_file.is_file() and get_file_type(str(image_file)) == "image":
                            total_images += 1

                            # 计算文件大小
                            size_mb = image_file.stat().st_size / (1024 * 1024)
                            total_storage_size_mb += size_mb

                            # 统计文件格式
                            ext = image_file.suffix.lower()
                            image_formats[ext] = image_formats.get(ext, 0) + 1

                            # 统计分辨率 (需要打开图片，可能会比较耗时)
                            try:
                                with Image.open(image_file) as img:
                                    res_key = f"{img.width}x{img.height}"
                                    resolution_distribution[res_key] = resolution_distribution.get(res_key, 0) + 1
                            except Exception:
                                # 如果图片损坏或无法打开，则跳过
                                pass

        image_meta["data_stats"]["total_images"] = total_images
        image_meta["data_stats"]["total_storage_size_mb"] = round(total_storage_size_mb, 2)
        image_meta["data_stats"]["avg_file_size_mb"] = round(total_storage_size_mb / total_images, 2) if total_images > 0 else 0
        image_meta["data_stats"]["image_formats"] = image_formats
        image_meta["data_stats"]["resolution_distribution"] = resolution_distribution
        # mode_distribution 统计比较复杂，这里暂时忽略

        with open(image_meta_path, 'w', encoding='utf-8') as f:
            json.dump(image_meta, f, ensure_ascii=False, indent=2)

    # --- 更新视频元数据 (meta_data/stats.json) ---
    video_meta_path = os.path.join(root_dir, "meta_data", "stats.json")
    if os.path.exists(video_meta_path):
        with open(video_meta_path, 'r', encoding='utf-8') as f:
            video_meta = json.load(f)

        video_dir = os.path.join(root_dir, "videos")
        total_videos = 0
        total_duration_seconds = 0
        total_storage_size_gb = 0
        video_formats = {}
        resolution_distribution = {}
        fps_distribution = {}

        if os.path.exists(video_dir):
            for bucket_dir in Path(video_dir).rglob('*'):
                if bucket_dir.is_dir():
                    for video_file in bucket_dir.iterdir():
                        if video_file.is_file() and get_file_type(str(video_file)) == "video":
                            total_videos += 1

                            # 计算文件大小
                            size_gb = video_file.stat().st_size / (1024 * 1024 * 1024)
                            total_storage_size_gb += size_gb

                            # 统计文件格式
                            ext = video_file.suffix.lower()
                            video_formats[ext] = video_formats.get(ext, 0) + 1

                            # 统计视频信息 (需要打开视频，可能会比较耗时)
                            try:
                                cap = cv2.VideoCapture(str(video_file))
                                if cap.isOpened():
                                    fps = cap.get(cv2.CAP_PROP_FPS)
                                    fps_key = f"{round(fps, 1)}"
                                    fps_distribution[fps_key] = fps_distribution.get(fps_key, 0) + 1

                                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                    res_key = f"{width}x{height}"
                                    resolution_distribution[res_key] = resolution_distribution.get(res_key, 0) + 1

                                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                                    duration = frame_count / fps if fps > 0 else 0
                                    total_duration_seconds += duration
                                cap.release()
                            except Exception:
                                # 如果视频损坏或无法打开，则跳过
                                pass

        video_meta["data_stats"]["total_videos"] = total_videos
        video_meta["data_stats"]["total_duration_seconds"] = round(total_duration_seconds, 2)
        video_meta["data_stats"]["total_duration_hours"] = round(total_duration_seconds / 3600, 2)
        video_meta["data_stats"]["avg_video_length_seconds"] = round(total_duration_seconds / total_videos, 2) if total_videos > 0 else 0
        video_meta["data_stats"]["total_storage_size_gb"] = round(total_storage_size_gb, 2)
        video_meta["data_stats"]["avg_file_size_gb"] = round(total_storage_size_gb / total_videos, 2) if total_videos > 0 else 0
        video_meta["data_stats"]["video_formats"] = video_formats
        video_meta["data_stats"]["resolution_distribution"] = resolution_distribution
        video_meta["data_stats"]["fps_distribution"] = fps_distribution

        with open(video_meta_path, 'w', encoding='utf-8') as f:
            json.dump(video_meta, f, ensure_ascii=False, indent=2)

def process_media_file(file_path, root_dir, total_count, language="英文"):
    """处理单个媒体文件，识别实体并更新标注"""
    try:
        file_type = get_file_type(file_path)
        if not file_type:
            return {
                "success": False,
                "path": file_path,
                "error": "不支持的文件类型"
            }

        # 查找或创建标注文件
        anno_path = find_annotation_file(file_path, root_dir)
        if not anno_path:
            # 创建新的标注文件，并复制媒体文件到输出目录
            # 为图片和视频分别计算新ID
            max_id = get_max_id_in_dir(root_dir, file_type)
            new_id = max_id + 1
            anno_path, file_id, new_media_path = create_new_annotation(file_path, root_dir, new_id)
            # 使用新复制的媒体文件路径进行实体识别
            process_path = new_media_path
        else:
            # 如果找到标注文件，使用原始路径进行实体识别
            file_id = os.path.splitext(os.path.basename(anno_path))[0]
            process_path = file_path

        # 加载现有标注
        with open(anno_path, 'r', encoding='utf-8') as f:
            annotation_data = json.load(f)

        # 识别实体
        if file_type == "image":
            entity_result = identify_entities(
                image_path=process_path,
                language=language
            )
        else:  # video
            # 为了提高效率，先提取关键帧
            frames, _ = extract_video_frames(process_path, fps=1.0)
            entity_result = identify_entities(
                frames=frames,
                language=language
            )

        # 更新标注中的实体信息
        if "annotations" not in annotation_data:
            annotation_data["annotations"] = {}

        # 将实体识别结果存入entity字段
        annotation_data["annotations"]["entity"] = entity_result

        # 保存更新后的标注
        with open(anno_path, 'w', encoding='utf-8') as f:
            json.dump(annotation_data, f, ensure_ascii=False, indent=2)

        return {
            "success": True,
            "id": file_id,
            "original_path": file_path,
            "processed_path": process_path,
            "annotation_path": anno_path,
            "type": file_type
        }

    except Exception as e:
        return {
            "success": False,
            "path": file_path,
            "error": str(e),
            "traceback": traceback.format_exc()
        }

def collect_files(input_dir):
    """收集目录中的所有图像和视频文件"""
    files = []
    for root, _, file_names in os.walk(input_dir):
        for file_name in file_names:
            file_path = os.path.join(root, file_name)
            file_type = get_file_type(file_path)
            if file_type:
                files.append((file_path, file_type))
    return files

def main():
    # 初始化结果字典
    result = {
        "success": False,
        "root_dir": None,
        "environment_info": check_environment(),
        "error": None,
        "details": {
            "processed_files": [],
            "errors": []
        }
    }

    try:
        # 解析命令行参数
        if len(sys.argv) < 3:
            raise ValueError("用法: python entity_caption.py <输入目录> <根目录> [语言]")

        input_dir = sys.argv[1]
        root_dir = sys.argv[2]
        language = sys.argv[3] if len(sys.argv) > 3 else "英文"

        if not os.path.exists(input_dir):
            raise FileNotFoundError(f"输入目录不存在: {input_dir}")

        os.makedirs(root_dir, exist_ok=True)
        result["root_dir"] = root_dir

        # 确保元数据文件和目录结构存在
        ensure_metadata_exists(root_dir)

        # 收集文件
        files = collect_files(input_dir)
        if not files:
            raise ValueError(f"在输入目录中未找到任何图像或视频文件: {input_dir}")

        # 处理文件
        processed = []
        for idx, (file_path, file_type) in enumerate(files):
            print(f"正在处理 {idx + 1}/{len(files)}: {file_path}")
            res = process_media_file(file_path, root_dir, idx, language)
            res["type"] = file_type
            processed.append(res)

            # 打印处理结果摘要
            if res["success"]:
                print(f"处理成功 -> 生成标注 ID: {res['id']}")
            else:
                print(f"处理失败: {res['error']}")

        # 更新结果统计
        result["success"] = all(item["success"] for item in processed)
        result["details"]["processed_files"] = [
            {
                "id": item["id"],
                "original_path": item["original_path"],
                "processed_path": item["processed_path"],
                "annotation_path": item["annotation_path"],
                "type": item["type"]
            }
            for item in processed if item["success"]
        ]
        result["details"]["errors"] = [
            {
                "path": item["path"],
                "error": item["error"],
                "type": item["type"]
            }
            for item in processed if not item["success"]
        ]

        # 更新元数据
        update_metadata_supported_annotations(root_dir)
        update_metadata_statistics(root_dir)

        print(f"\n处理完成 - 成功: {len(result['details']['processed_files'])}, 失败: {len(result['details']['errors'])}")
        print(f"结果摘要已保存至: {os.path.join(root_dir, 'entity_processing_result.json')}")

    except Exception as e:
        result["error"] = str(e)
        print(f"\n处理过程中发生严重错误: {str(e)}", file=sys.stderr)
        traceback.print_exc()

    # 保存处理结果摘要
    with open(os.path.join(root_dir, "entity_processing_result.json"), 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()

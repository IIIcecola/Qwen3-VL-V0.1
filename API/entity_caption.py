import os
import json
import sys
import base64
import cv2
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

def identify_entities(image_path=None, video_path=None, frames=None, entity_type="all", language="英文"):
    """
    识别图像或视频中的实体（名人、地标、电影角色）
    
    Args:
        image_path: 图像路径
        video_path: 视频路径
        frames: 视频帧列表（如果已提取）
        entity_type: 要识别的实体类型，可选值: "all", "celebrity", "landmark", "movie_character"
        language: 输出语言
    
    Returns:
        包含识别结果的字典
    """
    try:
        messages = [
            {"role": "system", "content": "你是一个图像识别专家，能够准确识别图像中的名人、地标和电影角色。请提供每个识别到的实体的位置（边界框或时间戳）和基本信息。"}
        ]
        
        user_content = []
        
        # 添加文本提示
        entity_prompt = {
            "all": "识别图像中的所有名人、地标和电影角色。",
            "celebrity": "识别图像中的所有名人，并提供他们的姓名、身份和在图像中的位置。",
            "landmark": "识别图像中的所有地标，并提供它们的名称、位置和基本信息。",
            "movie_character": "识别图像中的所有电影角色，并提供他们的姓名、出自的电影和在图像中的位置。"
        }
        
        prompt_text = f"用{language}详细描述以下内容中包含的{entity_prompt[entity_type]}对于每个实体，请明确说明其类型（名人/地标/电影角色）、位置（对于图像，使用边界框坐标[x1, y1, x2, y2]；对于视频，使用时间戳[开始时间, 结束时间]）以及基本信息。请以结构化的方式返回结果，便于解析。"
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
        
        return parse_entity_response(response.choices[0].message.content)
        
    except Exception as e:
        raise RuntimeError(f"实体识别失败: {str(e)}")

def parse_entity_response(response_text):
    """解析模型返回的实体识别结果"""
    # 这里可以根据实际返回格式进行更复杂的解析
    # 简化版本：将结果分类为不同实体类型
    entities = {
        "celebrity": [],
        "landmark": [],
        "movie_character": []
    }
    
    # 实际应用中应该根据模型返回的格式进行更精确的解析
    lines = response_text.split('\n')
    current_entity = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if "名人:" in line:
            current_entity = "celebrity"
        elif "地标:" in line:
            current_entity = "landmark"
        elif "电影角色:" in line:
            current_entity = "movie_character"
        elif current_entity and line:
            entities[current_entity].append(line)
    
    return entities

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
    
    file_name = os.path.basename(file_path)
    file_id = os.path.splitext(file_name)[0]
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

def create_new_annotation(file_path, root_dir, total_count):
    """创建新的标注文件（如果不存在）"""
    try:
        file_type = get_file_type(file_path)
        if not file_type:
            raise ValueError(f"不支持的文件类型: {file_path}")
        
        file_id = f"{total_count + 1:05d}"
        bucket = f"{int(file_id) // 1000:03d}"
        
        # 确定标注文件路径
        if file_type == "image":
            anno_dest_dir = os.path.join(root_dir, "labels", bucket)
            with Image.open(file_path) as img:
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
                    "celebrity": [],
                    "landmark": [],
                    "movie_character": []
                }
            }
        else:  # video
            anno_dest_dir = os.path.join(root_dir, "annotations", bucket)
            cap = cv2.VideoCapture(file_path)
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
                    "celebrity": [],
                    "landmark": [],
                    "movie_character": []
                }
            }
        
        os.makedirs(anno_dest_dir, exist_ok=True)
        anno_path = os.path.join(anno_dest_dir, f"{file_id}.json")
        
        with open(anno_path, 'w', encoding='utf-8') as f:
            json.dump(anno, f, ensure_ascii=False, indent=2)
        
        return anno_path, file_id
        
    except Exception as e:
        raise RuntimeError(f"创建新标注文件失败: {str(e)}")

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
            anno_path, file_id = create_new_annotation(file_path, root_dir, total_count)
        else:
            file_id = os.path.splitext(os.path.basename(anno_path))[0]
        
        # 加载现有标注
        with open(anno_path, 'r', encoding='utf-8') as f:
            annotation_data = json.load(f)
        
        # 识别实体
        if file_type == "image":
            entities = identify_entities(
                image_path=file_path,
                entity_type="all",
                language=language
            )
        else:  # video
            # 为了提高效率，先提取关键帧
            frames, _ = extract_video_frames(file_path, fps=1.0)
            entities = identify_entities(
                frames=frames,
                entity_type="all",
                language=language
            )
        
        # 更新标注中的实体信息
        if "annotations" not in annotation_data:
            annotation_data["annotations"] = {}
        
        annotation_data["annotations"]["celebrity"] = entities["celebrity"]
        annotation_data["annotations"]["landmark"] = entities["landmark"]
        annotation_data["annotations"]["movie_character"] = entities["movie_character"]
        
        # 保存更新后的标注
        with open(anno_path, 'w', encoding='utf-8') as f:
            json.dump(annotation_data, f, ensure_ascii=False, indent=2)
        
        return {
            "success": True,
            "id": file_id,
            "path": file_path,
            "annotation_path": anno_path,
            "type": file_type,
            "entities": {
                "celebrity_count": len(entities["celebrity"]),
                "landmark_count": len(entities["landmark"]),
                "movie_character_count": len(entities["movie_character"])
            }
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

def update_entity_metadata(root_dir, processed_files):
    """更新元数据，添加实体识别信息"""
    # 统计实体识别结果
    entity_stats = {
        "total_celebrity": 0,
        "total_landmark": 0,
        "total_movie_character": 0,
        "by_file_type": {
            "image": {
                "celebrity": 0,
                "landmark": 0,
                "movie_character": 0
            },
            "video": {
                "celebrity": 0,
                "landmark": 0,
                "movie_character": 0
            }
        }
    }
    
    for file_info in processed_files:
        if file_info["success"] and "entities" in file_info:
            entities = file_info["entities"]
            entity_stats["total_celebrity"] += entities["celebrity_count"]
            entity_stats["total_landmark"] += entities["landmark_count"]
            entity_stats["total_movie_character"] += entities["movie_character_count"]
            
            file_type = file_info["type"]
            entity_stats["by_file_type"][file_type]["celebrity"] += entities["celebrity_count"]
            entity_stats["by_file_type"][file_type]["landmark"] += entities["landmark_count"]
            entity_stats["by_file_type"][file_type]["movie_character"] += entities["movie_character_count"]
    
    # 保存实体统计信息
    meta_dir = os.path.join(root_dir, "meta_data")
    os.makedirs(meta_dir, exist_ok=True)
    
    with open(os.path.join(meta_dir, "entity_stats.json"), 'w', encoding='utf-8') as f:
        json.dump({
            "entity_recognition_stats": entity_stats,
            "processing_date": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        }, f, ensure_ascii=False, indent=2)

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
        # 解析命令行参数（实际应用中可以改为更灵活的参数解析）
        if len(sys.argv) < 3:
            raise ValueError("用法: python entity_caption.py <输入目录> <根目录> [语言]")
        
        input_dir = sys.argv[1]
        root_dir = sys.argv[2]
        language = sys.argv[3] if len(sys.argv) > 3 else "英文"
        
        if not os.path.exists(input_dir):
            raise FileNotFoundError(f"输入目录不存在: {input_dir}")
        
        os.makedirs(root_dir, exist_ok=True)
        result["root_dir"] = root_dir
        
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
                entities = res["entities"]
                print(f"处理成功 - 识别到 {entities['celebrity_count']} 个名人, {entities['landmark_count']} 个地标, {entities['movie_character_count']} 个电影角色")
            else:
                print(f"处理失败: {res['error']}")
        
        # 更新结果统计
        result["success"] = all(item["success"] for item in processed)
        result["details"]["processed_files"] = [
            {
                "id": item["id"], 
                "path": item["path"], 
                "type": item["type"],
                "entities": item["entities"]
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
        update_entity_metadata(root_dir, processed)
        print(f"处理完成 - 成功: {len(result['details']['processed_files'])}, 失败: {len(result['details']['errors'])}")

    except Exception as e:
        result["error"] = str(e)
        print(f"处理过程中发生错误: {str(e)}", file=sys.stderr)
        traceback.print_exc()

    # 保存处理结果摘要
    with open(os.path.join(root_dir, "entity_processing_result.json"), 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()

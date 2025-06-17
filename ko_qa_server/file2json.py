import os
import json
from pathlib import Path
from zhipuai import ZhipuAI


def extract_text_from_file(file_path, file_name, file_type):
    """
    从 file_path 指向的文件中提取文本内容，并将其以 JSON 格式保存到本地指定目录中。
    返回的结果包含提取到的文本和保存后的 JSON 文件路径。
    """
    json_data = {
        "名称": file_name,
        "content": ""
    }

    client = ZhipuAI(
        api_key="1e54498308567d4a66a210ea84420441.ieipu9oRxO8qO7W8",
        base_url="https://open.bigmodel.cn/api/paas/v4"
    )

    allowed_types = {'PDF', 'DOCX', 'DOC', 'XLS', 'XLSX', 'PPT', 'PPTX', 'PNG', 'JPG', 'JPEG', 'CSV'}

    if file_type.upper() not in allowed_types:
        json_data['content'] = "不支持的文件类型"
        return json_data, None

    file_size = os.path.getsize(file_path)
    size_limit = 50 * 1024 * 1024
    if file_size > size_limit:
        json_data['content'] = "文件内容过大"
        return json_data, None

    try:
        # 1. 上传文件到智谱api
        file_object = client.files.create(file=Path(file_path), purpose="file-extract")

        file_content_json_str = client.files.content(file_id=file_object.id).content.decode()
        file_content_json = json.loads(file_content_json_str)
        file_content = file_content_json.get("content", "")

        # 2. 删除远端文件
        client.files.delete(file_id=file_object.id)

        # 3. 更新 json_data 并保存到本地
        json_data['content'] = file_content
        save_dir = Path("./upload_files_json_all").resolve()
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        save_path = os.path.join(save_dir, f"{file_name}.json")

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)

        # 函数返回 (json_data, save_path)
        return json_data, save_path

    except Exception as e:
        json_data['content'] = f"文件解析失败: {str(e)}"
        print(f"文件解析失败: {str(e)}")
        return json_data, None


def extract_text_from_TXT(file_path, file_name):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    json_data = {"名称": file_name, 'content': content}
    save_dir = Path("./upload_files_json_all").resolve()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, f"{file_name}.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    return json_data, save_path



# extract_text_from_file(r"C:\Users\wsco22\Desktop\交通运输部关于印发《城市轨道交通设施设备运行维护管理办法》的通知.pdf",
#                        "交通运输部关于印发《城市轨道交通设施设备运行维护管理办法》的通知.pdf", "PDF")

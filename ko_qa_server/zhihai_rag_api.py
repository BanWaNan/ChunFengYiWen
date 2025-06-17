import asyncio

from openai import OpenAI
from pydantic import BaseModel
from fastapi import APIRouter
from fastapi import UploadFile, File, Form, Request
from fastapi.responses import StreamingResponse, JSONResponse
from chatzxl import *
import os

zhihai_rag_router = APIRouter()

kochatbot = Chatzxl()
kochatbot.init_database()


# 定义请求体
class QueryRequest1(BaseModel):
    query: str
    userid: str


async def get_stream(q_history):
    _qwen = OpenAI(
            api_key="sk-1c8ca6ab254349c6a17fecf753d2dd15",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
    response = _qwen.chat.completions.create(
        model="deepseek-v3",
        messages=q_history,
        temperature=0.1,
        stream=True,
    )
    # print(response)
    answer_content = ""  # 定义完整回复
    for msg in response:
        # 先确保 msg.choices 不为空
        if not msg.choices or len(msg.choices) == 0:
            # 可能是空的分片，直接跳过
            continue

        delta = msg.choices[0].delta
        # 还可以判断 delta 是否有 content 字段
        if hasattr(delta, "content") and delta.content:
            text_delta = delta.content.replace('\n', 'StReAmNeWlInE').replace(' ', 'StReAmSpAcE')
            yield f"data: {text_delta}\n\n"
            answer_content += text_delta

        await asyncio.sleep(0.1)
    print(answer_content)


@zhihai_rag_router.post("/upload")
async def handle_upload(userId: str = Form(...), file: UploadFile = File(...)):
    try:
        print("接收到的用户ID：", userId)

        user_vector_stores_path = Path(f"./user_vector_stores/{userId}").resolve()

        if not os.path.exists(user_vector_stores_path):
            os.makedirs(user_vector_stores_path, exist_ok=True)
            kochatbot.create_new_vector_stroes(user_vector_stores_path)

            # 同时创建doc_rangs.json文件
            doc_rangs_path = os.path.join(user_vector_stores_path, "doc_rangs.json")
            with open(doc_rangs_path, 'w', encoding='utf-8') as f:
                json.dump({}, f, ensure_ascii=False, indent=4)

            print(f"为用户 {userId} 创建新的向量库及doc_rangs.json: {user_vector_stores_path}")
        else:
            print("该用户的向量库已经存在。")

        contents = await file.read()
        save_dir = Path(f"./upload_file_user/{userId}").resolve()
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, file.filename)
        with open(save_path, "wb") as f:
            f.write(contents)

        # 获取文件类型
        filetype = file.filename.split('.')[-1].upper()
        print(f"文件类型: {filetype}")
        # if filetype in ["PPT", "PPTX"]:
        #     filepath_pdf = ppt_to_pdf(save_path, save_dir)
        #     save_path = filepath_pdf
        # 使用线程锁

        kochatbot.update_vector_stores(save_path, file.filename, userId, filetype)

        return JSONResponse({'msg': "文件接收成功"}, status_code=200)
    except Exception as e:
        return JSONResponse({'msg': str(e)}, status_code=500)


@zhihai_rag_router.post("/receive_question")
async def receive_question(request: Request):
    """
    接收前端传来的问题并返回
    :param request:
    :return: 返回接收到的问题
    """
    query_data = await request.json()
    userid = query_data['userid']
    question = query_data['query']
    query = question
    prompt = kochatbot.koquery2kb(query, userid)
    q_history = [{"role": "user", "content": prompt}]
    return StreamingResponse(
        get_stream(q_history),
        media_type="text/event-stream"
    )

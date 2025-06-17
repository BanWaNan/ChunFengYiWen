from datetime import timedelta
from queue import Queue, Empty
from pydantic import BaseModel
from fastapi import APIRouter, Request
from starlette.responses import StreamingResponse
from types import SimpleNamespace
from zhihaiQA import *
from flask import Flask, request, jsonify
import os
import time
from zhipuai import ZhipuAI

app = Flask(__name__)
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
zhihaiQA_router = APIRouter()

zhqa = ZhiHaiQA()


class Result(BaseModel):
    response: object = None
    stream: bool = False
    time: float = None
    status: int = 200
    errorMessage: str = None
    docs: object = None

    def __init__(self, **kwargs):
        if "time" not in kwargs:
            kwargs["time"] = time.time()
        super().__init__(**kwargs)


class UserQAHistory:
    def __init__(self):
        self.user_data = {}  # 存储问答框ID及其问答历史和最后更新时间

    def add_or_update_user_qa(self, qa_box_id, qa_pair):
        """
        添加或更新用户的问答历史，并更新最后更新时间
        :param qa_box_id: 问答框ID
        :param qa_pair: 问答对，格式为
            [{"role": "user", "content": "问题"},
            {"role": "assistant", "content": "回答"}]
        """
        if qa_box_id not in self.user_data:
            self.user_data[qa_box_id] = {'qa_list': [], 'last_updated': datetime.now()}

        self.user_data[qa_box_id]['qa_list'].extend(qa_pair)
        self.user_data[qa_box_id]['last_updated'] = datetime.now()

    def remove_old_users(self):
        """
        删除超过一天没有更新的用户及其问答历史
        """
        one_day_ago = datetime.now() - timedelta(days=1)
        users_to_remove = [qa_box_id for qa_box_id, data in self.user_data.items() if
                           data['last_updated'] < one_day_ago]

        for qa_box_id in users_to_remove:
            del self.user_data[qa_box_id]

    def get_user_qa_history(self, qa_box_id):
        """
        获取用户的问答历史
        :param qa_box_id: 问答框ID
        :return: 用户的问答历史列表，如果没有找到用户则返回空列表
        """
        return self.user_data.get(qa_box_id, {}).get('qa_list', [])

with open("./sensitive.json", 'r', encoding='utf-8') as file:
    ban_words = json.load(file)

kdzn_history = UserQAHistory()
ds_history = UserQAHistory()


def get_model(model_name, difficulty=None):
    model_mapping = {
        "Qwen": "qwen-turbo-1101",
        "DeepSeekV3": "deepseek-v3",
        "DeepSeekR1": "deepseek-r1",
        "Kimi": "moonshot-v1-8k",
        "QwQ": "qwq-32b",
        "Spark": "spark",
        "ZhipuAI": "glm-4-flash"
    }
    model_value = model_mapping.get(model_name, model_mapping["DeepSeekR1"])

    return model_value


def allmodel_stream_time(web, bigbase, model_name, messages, streaming_flag, qid=None, timeout=10):
    state = SimpleNamespace(
        reasoning_content="",  # 完整的思考过程
        answer_content="",  # 完整的回答
        is_answering=False  # 标记是否从思考过程进入回答阶段
    )

    def data_generator():

        # —— 新增：如果需要流式标志，就先输出一次 ——
        if streaming_flag:
            print(111111)
            yield f"data: zxl\n\n"

        try:
            # "Kimi": "moonshot-v1-8k",
            # "QwQ": "qwq-32b-preview",
            # "star": "spark"
            answer_content = ""  # 定义完整回复
            if model_name == "spark":
                # 拿到最后一条 user 消息做为 query
                query = messages[-1]['content']
                print("query:")
                print(query)
                for delta in zhqa.spark(
                        appid="9947c2c7",
                        api_key="430b795ff304ad85fcdba389cf8d09b0",
                        api_secret="NzAyZWNiMTRiYjliNTY2OTQxOTYyMDJk",
                        gpt_url="wss://spark-api.xf-yun.com/v3.5/chat",
                        domain="generalv3.5",
                        query=query):
                    # 直接是干净的星火输出，不要再替换\n、空格
                    print(delta)
                    yield f"data: {delta}\n\n"
                    answer_content += delta

            elif model_name == "glm-4-flash":
                client = ZhipuAI(api_key="a693ae0ec512459a8eabf7a8d27b019c.jTMZ4ZQlyJXHouwG")
                # response = client.chat.completions.create(
                #     model=model_name,
                #     messages=messages,
                #     temperature=0.1,
                #     stream=True,
                # )
                response = client.chat.completions.create(
                    model="glm-4-flash",
                    messages=messages,
                    top_p=0.7,
                    temperature=0.95,
                    max_tokens=1024,
                    tools=[{"type": "web_search", "web_search": {"search_result": True}}],
                    stream=True
                )
                for msg in response:
                    delta = msg.choices[0].delta
                    if delta.content:
                        text_delta = delta.content.replace('\n', 'StReAmNeWlInE').replace(' ', 'StReAmSpAcE')
                        yield f"data: {text_delta}\n\n"
                        # print(f"data: {text_delta}\n\n")
                        answer_content = answer_content + text_delta
                        time.sleep(0.01)

            elif model_name == "moonshot-v1-8k":
                response = zhqa._kimi.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0.1,
                    stream=True,
                )
                for msg in response:
                    delta = msg.choices[0].delta
                    if delta.content:
                        text_delta = delta.content.replace('\n', 'StReAmNeWlInE').replace(' ', 'StReAmSpAcE')
                        yield f"data: {text_delta}\n\n"
                        # print(f"data: {text_delta}\n\n")
                        answer_content = answer_content + text_delta
                        time.sleep(0.01)
            else:
                response = zhqa._qwen2.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0.1,
                    stream=True,
                )

                if model_name == "deepseek-r1":
                    reasoning_content = ""  # 定义完整思考过程
                    is_answering = False  # 判断是否结束思考过程并开始回复
                    yield f"data: > **思考内容：**StReAmNeWlInEStReAmNeWlInE > \n\n"
                    i_sleep = 0
                    for chunk in response:
                        # if not getattr(chunk, 'choices', None):  # Token使用情况
                        #     print("\n" + "\033[92m=" * 20 + f"最终{model}流式Token使用情况" + str(chunk.usage) + "\033[0m")
                        #     continue
                        delta = chunk.choices[0].delta
                        if not hasattr(delta, 'reasoning_content'):  # 检查是否有reasoning_content属性
                            continue
                        if not getattr(delta, 'reasoning_content', None) and not getattr(delta, 'content',
                                                                                         None):  # 处理空内容情况
                            continue
                        if not getattr(delta, 'reasoning_content', None) and not is_answering:  # 处理开始回答的情况
                            # print("\n" + "=" * 20 + "完整回复" + "=" * 20 + "\n")
                            is_answering = True
                            yield "data: StReAmNeWlInE --- StReAmNeWlInE\n\n"
                            yield "data: **开始回答：** StReAmNeWlInEStReAmNeWlInE \n\n"

                        if getattr(delta, 'reasoning_content', None):  # 处理思考过程
                            # print(delta.reasoning_content, end='', flush=True)
                            text_delta = delta.reasoning_content.replace('\n', 'StReAmNeWlInE > ').replace(' ',
                                                                                                           'StReAmSpAcE')
                            i_sleep += 1
                            yield f"data: {text_delta}\n\n"
                            reasoning_content += delta.reasoning_content
                        elif getattr(delta, 'content', None):  # 处理回复内容
                            # print(delta.content, end='', flush=True)
                            i_sleep += 1
                            text_delta = delta.content.replace('\n', 'StReAmNeWlInE').replace(' ', 'StReAmSpAcE')

                            yield f"data: {text_delta}\n\n"
                            answer_content += delta.content
                        time.sleep(0.01)
                        # if i_sleep > 100:
                        #     time.sleep(6)
                elif model_name == "qwq-32b":
                    reasoning_content = ""  # 定义完整思考过程
                    answer_content = ""  # 定义完整回复
                    is_answering = False  # 判断是否结束思考过程并开始回复
                    yield f"data: > **思考内容：**StReAmNeWlInEStReAmNeWlInE > \n\n"
                    for chunk in response:
                        # 如果chunk.choices为空，则打印usage
                        if not chunk.choices:
                            print("\nUsage:")
                            print(chunk.usage)
                        else:
                            delta = chunk.choices[0].delta
                            # 打印思考过程
                            if hasattr(delta, 'reasoning_content') and delta.reasoning_content != None:
                                yield f"data: {delta.reasoning_content}\n\n"
                                print(delta.reasoning_content, end='', flush=True)
                                reasoning_content += delta.reasoning_content
                            else:
                                # 开始回复
                                if delta.content != "" and is_answering is False:
                                    yield "data: **开始回答：**\n\n"
                                    print("\n" + "=" * 20 + "完整回复" + "=" * 20 + "\n")
                                    is_answering = True
                                # 打印回复过程
                                yield f"data: {delta.content}\n\n"
                                print(delta.content, end='', flush=True)
                                answer_content += delta.content
                            time.sleep(0.01)
                else:
                    for msg in response:
                        delta = msg.choices[0].delta
                        if delta.content:
                            text_delta = delta.content.replace('\n', 'StReAmNeWlInE').replace(' ', 'StReAmSpAcE')
                            yield f"data: {text_delta}\n\n"
                            # print(f"data: {text_delta}\n\n")
                            answer_content = answer_content + text_delta
                            time.sleep(0.01)
            state.answer_content = answer_content
            print("大模型回答为：")
            print(state.answer_content)

        except Exception as e:
            # 一定要先用 getattr 拿 code，否则 TypeError/其它异常会没有 .code
            err_code = getattr(e, 'code', None)
            if err_code == 'data_inspection_failed':
                text = "我们换个话题聊聊吧。"
                yield f"data: {text}\n\n"

        if not web and not bigbase:
            yield f"data: StReAmNeWlInEStReAmNeWlInE**您可以在输入栏左边选择开启更大图谱或者联网搜索，以获取更多相关信息。**\n\n"
        elif not web and bigbase:
            yield f"data: StReAmNeWlInEStReAmNeWlInE**您可以在输入栏左边选择开启联网搜索，以获取更多相关信息。**\n\n"
        elif web and not bigbase:
            yield f"data: StReAmNeWlInEStReAmNeWlInE**您可以在输入栏左边选择开启更大图谱，以获取更多相关信息。**\n\n"

        yield f"data: StReAmNeWlInEStReAmNeWlInE**如果您对回答结果不满意，您可以选择[上传文件]、或[上传表格录入]知识图谱、或在现有知识上[增设规则]，以改善回答效果。**\n\n"

    def alternative_method():
        """
        超时后调用的备选方法。
        """
        error_text = "【KO-请大模型求异常出现，重试稍后请】"
        yield f"data: {error_text}\n\n"

    def run_generator_with_timeout(gen, timewaitout=timeout):
        """
        从生成器 gen 中获取数据，如果超过 timewaitout 秒还没有数据，则超时。
        """
        q = Queue()

        def worker():
            try:
                for item in gen:
                    q.put(item)
            except Exception as e:
                q.put(e)
            finally:
                q.put(None)

        thread = threading.Thread(target=worker)
        thread.daemon = True
        thread.start()
        ans_over = False
        while True:
            try:
                item = q.get(timeout=timewaitout)
                if item is None:
                    # print("R1生成器已结束")
                    ans_over = True
                    break
                elif isinstance(item, Exception):
                    raise item
                else:
                    yield item
            except Empty:
                yield from alternative_method()
                break
        if qid and ans_over:
            messages.append({"role": "assistant", "content": state.answer_content})
            kdzn_history.add_or_update_user_qa(qid, messages[-2:])
            kdzn_history.remove_old_users()

    return run_generator_with_timeout(data_generator(), timewaitout=timeout)


# 文件上传接口
@app.route('/uploadFile', methods=['POST'])
def upload_files():
    if 'files' not in request.files:
        return jsonify({'success': False, 'message': 'No files part'}), 400

    files = request.files.getlist('files')  # 获取多个文件
    uploaded = []

    for file in files:
        if file.filename == '':
            continue

        timestamp = str(int(time.time() * 1000))
        filename = f"{timestamp}-{file.filename}"
        save_path = os.path.join(UPLOAD_FOLDER, filename)

        try:
            file.save(save_path)
            print(f"{filename} -- 上传成功")
            uploaded.append(filename)
            time.sleep(0.001)  # 确保时间戳唯一
        except Exception as e:
            print(f"{file.filename} -- 上传失败: {e}")

    if uploaded:
        return jsonify({'success': True, 'uploaded': uploaded}), 200
    else:
        return jsonify({'success': False, 'message': 'No files uploaded'}), 500


@zhihaiQA_router.post('/ko4csg_hci')
async def ko4csg_hci(request: Request):
    print("\033[93m" + "ko4csg_hci" + "\033[0m")
    csg_res = Result()
    query_data = await request.json()
    print(query_data)
    qid = query_data['qid']
    query = query_data['query']
    qa_llm_model = query_data.get("model_choose")
    qa_llm_model = get_model(qa_llm_model)
    web = query_data.get('web', False)
    bigbase = query_data.get('bigbase', False)
    print("web:", web)
    print("bigbase:", bigbase)
    if query == "" or qid == "":
        csg_res.status = 400
        csg_res.errorMessage = "输入最新问答对应为user"
        return csg_res
    else:  # 正常处理-------
        q_history = kdzn_history.get_user_qa_history(qid)
        q_history.append({"role": "user", "content": query})
        print(q_history)
        print("问答请求模型", qa_llm_model)
        getQuery_mess, get_path, _, _, streaming_flag = zhqa.csg_port_stream(web, bigbase, q_history)
        print(streaming_flag)
        return StreamingResponse(
            allmodel_stream_time(web, bigbase, qa_llm_model, getQuery_mess, streaming_flag, qid=qid, timeout=45),
            media_type="text/event-stream"
        )


@zhihaiQA_router.post('/ycyx')
async def ycyx(request: Request):
    print("\033[93m" + "ycyx" + "\033[0m")
    query_data = await request.json()
    data_source = query_data['data_source']
    status = query_data['status']
    print("数据来源:", data_source)
    print("数据:", status)
    filtered_formatted_paths = zhqa.ycyx_stream(data_source, status)
    return filtered_formatted_paths


import _thread as thread
import base64
import concurrent.futures
import datetime
import hashlib
import hmac
import json
import os
import queue
import ssl
import threading
import time
from datetime import datetime
from time import mktime
from urllib.parse import urlencode
from urllib.parse import urlparse
from wsgiref.handlers import format_date_time

import requests
import websocket
import yaml
from openai import OpenAI

from prompt_utils import build_prompt, prompt_list
from web_search_api import web_search_pachong_with_memo, web_search_baidubaike_with_memo
from zhihai_qa.neo4j_def import KO_tupu, all_paths_from_entity
from zhihai_qa.neo4j_hpt import find_kdzn, getYiChang, find_most_similar
from zhihai_qa.path2json import transform_paths_to_hierarchy

with open("config.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)


# temp_gpt35_api_key = ["sk-SxFJyauOcVOO1Lxu6WG6D0Xl2sUlP3OuuXa3LftB2wFOvlsg",
#                       "sk-l6kYtoQjX9t3tLtLcWrJaFaPTxHm4X9ULcgGciYwvqY1xawV"]
# temp_gpt35_base_url = "https://api.chatanywhere.tech/v1"

# 星火大模型

class Ws_Param(object):
    # 初始化
    def __init__(self, APPID, APIKey, APISecret, gpt_url):
        self.APPID = APPID
        self.APIKey = APIKey
        self.APISecret = APISecret
        self.host = urlparse(gpt_url).netloc
        self.path = urlparse(gpt_url).path
        self.gpt_url = gpt_url

    # 生成url
    def create_url(self):
        # 生成RFC1123格式的时间戳
        now = datetime.now()
        date = format_date_time(mktime(now.timetuple()))

        # 拼接字符串
        signature_origin = "host: " + self.host + "\n"
        signature_origin += "date: " + date + "\n"
        signature_origin += "GET " + self.path + " HTTP/1.1"

        # 进行hmac-sha256进行加密
        signature_sha = hmac.new(self.APISecret.encode('utf-8'), signature_origin.encode('utf-8'),
                                 digestmod=hashlib.sha256).digest()

        signature_sha_base64 = base64.b64encode(signature_sha).decode(encoding='utf-8')

        authorization_origin = f'api_key="{self.APIKey}", algorithm="hmac-sha256", headers="host date request-line", ' \
                               f'signature="{signature_sha_base64}" '

        authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')

        # 将请求的鉴权参数组合为字典
        v = {
            "authorization": authorization,
            "date": date,
            "host": self.host
        }
        # 拼接鉴权参数，生成url
        url = self.gpt_url + '?' + urlencode(v)
        # 此处打印出建立连接时候的url,参考本demo的时候可取消上方打印的注释，比对相同参数时生成的url与自己代码生成的url是否一致
        return url


# 收到websockets错误的处理
def on_error(ws, error):
    print("### error:", error)


# 收到websockets关闭的处理
def on_close(ws):
    print("### closed ###")


# 收到websockets连接建立的处理
def on_open(ws):
    thread.start_new_thread(run, (ws,))


def run(ws, *args):
    data = json.dumps(gen_params(appid=ws.appid, query=ws.query, domain=ws.domain))
    ws.send(data)


def gen_params(appid, query, domain):
    """
    通过appid和用户的提问来生成请参数
    """

    data = {
        "header": {
            "app_id": appid,
            "uid": "1234",
            # "patch_id": []    #接入微调模型，对应服务发布后的resourceid
        },
        "parameter": {
            "chat": {
                "domain": domain,
                "temperature": 0.5,
                "max_tokens": 4096,
                "auditing": "default",
            }
        },
        "payload": {
            "message": {
                "text": [{"role": "user", "content": query}]
            }
        }
    }
    return data


class ZhiHaiQA:
    def __init__(self):

        self._gpt = OpenAI(api_key=cfg["small_models"]["gpt"]["api_key"],
                           base_url=cfg["small_models"]["gpt"]["base_url"])
        self._qwen2 = OpenAI(
            api_key=cfg["small_models"]["qwen"]["api_key"],  # 如果您没有配置环境变量，请在此处用您的API Key进行替换
            base_url=cfg["small_models"]["qwen"]["base_url"],  # 填写DashScope SDK的base_url
        )
        self._kimi = OpenAI(
            api_key=cfg["small_models"]["kimi"]["api_key"],
            base_url=cfg["small_models"]["kimi"]["base_url"],
        )
        self.prompt_list = prompt_list

    def small_model(self, model_name, prompt):
        if model_name == 'qwen':
            return self.qwen2(None, prompt)
        elif model_name == 'gpt':
            return self.gpt_predict(None, prompt)
        elif model_name == 'kimi':
            return self.kimi(None, prompt)
        else:
            print("暂无" + model_name + "模型")

    def embedding_model(self, model_name):
        if model_name == 'qwen':
            return self._qwen2

    def gpt_predict(self, head, prompt):
        """output:answer:str"""
        messages = [{'role': 'user', 'content': prompt}]
        if head:
            messages.insert(0, {'role': 'system', 'content': head})
        for _ in range(5):
            try:
                completion = self._gpt.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    temperature=0.0
                )
                return completion.choices[0].message.content
            except Exception:
                time.sleep(1)

    def qwen(self, mess, model_name="qwen2"):
        quest_model = "qwen2-72b-instruct"
        if model_name == "qwen2.5":
            quest_model = "qwen2.5-72b-instruct"
        try:
            completion = self._qwen2.chat.completions.create(
                model=quest_model,
                messages=mess,
                temperature=0.0
            )
            # print(f"\033[96mQwen2使用token量：{completion.usage} \033[0m")
            return completion.choices[0].message.content
        except Exception as e:
            if e.code == 'data_inspection_failed':
                return "我们换个话题聊聊吧。"

    def get_ans_with_retry(self, message, model="qwen2", try_num=5):
        for _ in range(try_num):
            try:
                res_ans = self.qwen(message, model)
                if res_ans and res_ans != '{}':
                    return res_ans
            except Exception:
                time.sleep(1)
        return None

    def qwen2(self, head, prompt):
        """output:answer:str"""
        messages = [{'role': 'user', 'content': prompt}]
        if head:
            messages.insert(0, {'role': 'system', 'content': head})
        result_ans = self.get_ans_with_retry(messages)
        if not result_ans:
            result_ans = self.get_ans_with_retry(messages, "qwen2.5")
        return result_ans

    def kimi(self, head, prompt):
        messages = [{'role': 'user', 'content': prompt}]
        if head:
            messages.insert(0, {'role': 'system', 'content': head})
        try:
            completion = self._kimi.chat.completions.create(
                model="moonshot-v1-8k",
                messages=messages,
                temperature=0.3,
            )
            result = completion.choices[0].message.content
            return result
        except Exception:
            print("KIMI错误")

    def spark(self, appid, api_key, api_secret, gpt_url, domain, query, timeout=30):
        q = queue.Queue()

        # 改写回调，把 content 放入队列
        def on_message(ws, message):
            data = json.loads(message)
            if data["header"]["code"] != 0:
                q.put(Exception(f"星火返回错误：{data['header']}"))
                ws.close()
                return
            choices = data["payload"]["choices"]
            content = choices["text"][0]["content"]
            q.put(content)
            if choices["status"] == 2:  # 最后一条
                q.put(None)  # 标志结束
                ws.close()

        # 其它回调保持不变
        def on_error(ws, error):
            q.put(Exception(error))
            ws.close()

        def on_close(ws, *args):
            q.put(None)

        def on_open(ws):
            payload = gen_params(appid, query, domain)
            ws.send(json.dumps(payload))

        # 启动 WebSocket 在另一个线程
        wsParam = Ws_Param(appid, api_key, api_secret, gpt_url)
        wsUrl = wsParam.create_url()
        ws_app = websocket.WebSocketApp(wsUrl,
                                        on_message=on_message,
                                        on_error=on_error,
                                        on_close=on_close,
                                        on_open=on_open)
        t = threading.Thread(target=lambda: ws_app.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE}))
        t.daemon = True
        t.start()

        # 从队列里不断拿数据并 yield
        start = time.time()
        while True:
            try:
                chunk = q.get(timeout=0.1)
            except queue.Empty:
                # 超时机制
                if time.time() - start > timeout:
                    raise TimeoutError("星火流式超时")
                continue

            if isinstance(chunk, Exception):
                raise chunk
            if chunk is None:
                break

            yield chunk

    def local_glm(self, prompt):
        url = f"http://114.213.232.140:6289"
        data = {
            "prompt": prompt,
            "history": [],
        }
        try:
            response = requests.post(url, data=json.dumps(data))
            return response.json()["response"]
        except Exception as e:
            print('error: ', e)

    def csg_port_stream(self, web, bigbase, query_history: list):
        # tishi_flag = False
        wenti = query_history[-1]["content"]
        # 意图匹配：闲聊+所有文件本体
        yituprompt = self.prompt_list["kdzn_yitu_file"].replace("__input__", wenti)
        # 事件抽取
        event_extract_prompt = self.prompt_list["kdzn_input_extract"].replace("__input__", wenti)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            print("\033[94m 意图/场景识别 \033[0m")
            future_yitu = executor.submit(self.small_model, cfg.get('choose_small_model'), yituprompt)
            yitu_file_res = future_yitu.result()
            print("\033[94m 事件抽取 \033[0m")
            event_extract_llmRes = executor.submit(self.small_model, cfg.get('choose_small_model'),
                                                   event_extract_prompt)
            event_extract_llmRes = event_extract_llmRes.result()
        print("csg_sys_port:意图匹配：", yitu_file_res)
        print(event_extract_llmRes)
        file_ontology_list = []
        _yitu = "Chitchat"
        if yitu_file_res:
            yitu_file_res = json.loads(yitu_file_res)
            if yitu_file_res.get("意图") != "Chitchat":
                _yitu = yitu_file_res.get("意图", "Chitchat")
                file_ontology_list = yitu_file_res.get("相关文件")
        # 使用正则提取 “意图” 字段的值
        yitu_file_res = str(yitu_file_res)
        streaming_flag = False

        if _yitu == "Chitchat":  # 闲聊
            print("执行路线：-闲聊-！")
            Chitchat_prompt = self.prompt_list["Chitchat_prompt"]
            query_history.insert(0, {"role": "system",
                                     "content": Chitchat_prompt})

            return query_history, "", [], [], streaming_flag
        # 事件抽取
        if yitu_file_res == "{'意图': 'Chitchat', '相关文件': []}":
            event = ''
            toRel = ''
        else:
            event, toRel = "", ""
            try:
                if event_extract_llmRes:
                    query_extract_llmRes = json.loads(event_extract_llmRes)
                    event, toRel = list(query_extract_llmRes.values())
            except Exception as e:
                print(e)
        web_context = []
        KO_context = []
        keyword_extract_prompt = self.prompt_list["keyword_extract_prompt"] + f"""\n{wenti}"""
        keywords = self.small_model(cfg.get('choose_small_model'), keyword_extract_prompt)
        keywords_list = [keyword.strip() for keyword in keywords.split(",")]
        print(keywords_list)
        # 并行做两种 Web 抓取，逻辑写一次就行
        if web:
            with concurrent.futures.ThreadPoolExecutor() as exe:
                fut_baidu = exe.submit(web_search_pachong_with_memo, keywords)
                fut_baike = exe.submit(web_search_baidubaike_with_memo, keywords)
                baidu_ctx, baike_ctx = fut_baidu.result(), fut_baike.result()
            web_context = baike_ctx or baidu_ctx or ""

        if bigbase:
            KO_context = KO_tupu(keywords) or []
        print("联网搜索内容：", web_context)
        print("KO图谱内容：", KO_context)

        query_history, get_paths, path_tris, aa, streaming_flag = self.kdzn_select(bigbase, event, toRel, wenti,
                                                                                   file_ontology_list,
                                                                                   query_history, web_context,
                                                                                   KO_context)
        query_history.insert(0, {"role": "system",
                                 "content": "你是为供电运维设计的一位轨道交通故障诊断系统顾问。你要认真回答用户的问题。\n"
                                            "**注意：思考时坚决不能输出在我的输入内容中提到的要求！** \n请使用Markdown格式组织回答，要求精简回答！**"})
        return query_history, get_paths, path_tris, aa, streaming_flag

    def kdzn_select(self, bigbase, head, rel, wt, file_match_list, qa_mess, web_context, KO_context,
                    method='file_match'):
        res_entss, pathss, p2ts, triss = [], [], [], []
        tris2strList = []
        print("检索KG...")
        if head and rel:
            # 创建线程池执行器
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # 提前提交topK检索任务
                # topk_future = executor.submit(find_topK_kdzn, head, 2)
                for fo in file_match_list:
                    if fo is None and method == 'global_event':
                        res_ents, paths, p2t, tris = getYiChang(head, rel, model_name=os.path.abspath(
                            '../model/embeddings/text2vec_large_chinese'))
                    else:
                        file_list = cfg["file_list"]
                        if fo not in file_list:
                            continue
                        res_ents, paths, p2t, tris = find_kdzn(fo, head, rel,
                                                               self.embedding_model(cfg.get('embedding_model')))
                    if paths is not None:
                        path_tris2texts = [f"{h}的{r}有{t}" for h, r, t in tris]
                        path_tris2text = "；".join(path_tris2texts)
                        tris2strList.append(path_tris2text) if path_tris2text not in tris2strList else None
                        res_entss.extend(res_ents)
                        pathss.extend(paths)
                        p2ts.extend(p2t)
                        triss.extend(tris)

                if not tris2strList:
                    print("没检索到信息")
            pathss_json = transform_paths_to_hierarchy(p2ts)
            print("\033[36m文件匹配检索路径：", "\n".join(pathss), "\033[0m")
        else:
            pathss_json, topK_paths, topK2triplet, topK_tris = "", [], [], []
        web_context = "\n\n".join(web_context)
        streaming_flag = False
        # 全局匹配路径topK_paths，暂时设置为空
        topK_paths = []
        if pathss == [] and topK_paths == [] and bigbase:
            streaming_flag = True
        knowledge_res = f"### 规则匹配结果：{pathss_json}" if pathss_json else ""
        global_refs = topK_paths + KO_context
        if global_refs:
            knowledge_res += "\n### 全局匹配参考：" + "\n\n".join(
                "\n".join(p) if isinstance(p, list) else p for p in global_refs
            )
        prompt = build_prompt(
            web_context=web_context.strip() if web_context else "",
            ko_ctx=KO_context,
            knowledge_res=knowledge_res,
            question=wt,
        )

        all_paths = pathss + topK_paths
        all_p2ts = p2ts
        all_tris = triss
        qa_mess[-1]["content"] = prompt
        res_paths = json.dumps(all_paths, ensure_ascii=False)

        return qa_mess, res_paths, all_p2ts, all_tris, streaming_flag

    def ycyx_stream(self, data_source, status):
        yxyc_data_list = cfg.get(['yxyc_data_list'])
        # 匹配最相似的实体
        best_match, score = find_most_similar(data_source, yxyc_data_list,
                                              self.embedding_model(cfg.get('embedding_model')))
        formatted_paths, paths_tris = all_paths_from_entity(best_match)
        print(formatted_paths)
        print(f"匹配到的数据：{best_match}。")
        # print(f"当前数值：{data}，阈值上限：{threshold_up}，阈值下限：{threshold_down}")
        print(f"分析结果：{status}")
        filtered_formatted_paths = [p for p in formatted_paths if status in p]
        print("匹配到的路径：", filtered_formatted_paths)
        entities = []
        for p in filtered_formatted_paths:
            # 方法一：用 split
            entity = p.split('可能原因->')[-1]
            entities.append(entity)
        print("可能的原因：", entities[-1])
        return filtered_formatted_paths

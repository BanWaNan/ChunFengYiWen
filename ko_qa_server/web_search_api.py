import urllib.parse

import pymysql
import requests
from duckduckgo_search import DDGS
import asyncio
from openai import OpenAI
from pydantic import BaseModel
from fastapi import APIRouter
from fastapi import Request
from fastapi.responses import StreamingResponse
import urllib.parse
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import time
import json
from http import HTTPStatus
from dashscope import Application
from selenium.webdriver.support.ui import WebDriverWait  # 添加导入
from selenium.webdriver.support import expected_conditions as EC  # 添加导入
from selenium.webdriver.common.by import By
import re

web_of_search_router = APIRouter()


# 定义请求体
class QueryRequest1(BaseModel):
    query: str
    userid: str


async def get_stream(q_history):
    _gpt = OpenAI(
        api_key="sk-1c8ca6ab254349c6a17fecf753d2dd15",
        base_url="https://api.chatanywhere.tech/v1",
    )
    response = _gpt.chat.completions.create(
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

        await asyncio.sleep(0.01)


def parse_search_results(formatted_str):
    """
    解析 format_search_results 格式化后的字符串，反向生成原始结果列表
    """
    items = []
    # 使用正则按标题分割块，标题是"标题: "开头的行
    # 先按换行分割，逐行处理
    lines = formatted_str.splitlines()
    current_item = {}
    for line in lines:
        line = line.strip()
        if line.startswith("标题: "):
            # 新条目开始，先保存前一个
            if current_item:
                items.append(current_item)
                current_item = {}
            current_item["title"] = line[len("标题: "):]
        elif line.startswith("链接: "):
            current_item["url"] = line[len("链接: "):]
        elif line.startswith("内容: "):
            current_item["description"] = line[len("内容: "):]
        elif line == "":
            # 空行代表条目分割，保存当前条目
            if current_item:
                items.append(current_item)
                current_item = {}
    # 最后一个条目添加
    if current_item:
        items.append(current_item)

    return items


def format_search_results(results, method):
    lines = []
    # 智能体联网搜索
    if method == 'dashscope':
        for idx, result in enumerate(results, start=1):
            title = result.get("title", "无标题")
            url = result.get("url", "无链接")
            description = result.get("description", "无描述")
            lines.append(f"标题: {title}")
            lines.append(f"链接: {url}")
            lines.append(f"内容: {description}\n")
    # duckduckgo联网搜索
    elif method == 'DDG':
        for idx, result in enumerate(results, start=1):
            title = result.get("title", "无标题")
            href = result.get("href", "无链接")
            body = result.get("body", "无描述")
            lines.append(f"标题: {title}")
            lines.append(f"链接: {href}")
            lines.append(f"内容: {body}\n")
    # 爬虫联网搜索
    elif method == 'pachong':
        for idx, result in enumerate(results, start=1):
            title = result.get("title", "无标题")
            href = result.get("url", "无链接")
            body = result.get("description", "无描述")
            lines.append(f"标题: {title}")
            lines.append(f"链接: {href}")
            lines.append(f"内容: {body}\n")
    # 百度百科
    elif method == 'baidubaike':
        for idx, result in enumerate(results, start=1):
            title = result.get("title", "无标题")
            href = result.get("url", "无链接")
            body = result.get("description", "无描述")
            lines.append(f"标题: {title}")
            lines.append(f"链接: {href}")
            lines.append(f"内容: {body}\n")
    return "\n".join(lines)


def get_prompt(context, query):
    # 遍历所有搜索结果，将其格式化后添加到 prompt 中
    prompt = f"""
- 目标 -
根据提供的联网搜索内容，准确回答用户的问题。如果搜索到的内容中没有相关信息，请明确告知。

- 角色 -
您扮演的角色是“变压器故障诊断小助手”，专门利用联网搜索中获取到的内容为用户提供专业解答。

- 注意事项 -
1. 确保回答与问题完全对应，避免偏离主题。
2. 匹配到的知识存在与问题无关的内容，请注意筛选。充分利用对问题有用的内容。
3. 如果搜索到的知识中没有相关信息，请明确告知用户“由于缺乏相关信息，当前无法准确回答您的问题”。
4. 信息来源展示给用户，即将网页连接也输出给用户。
5. 可以适当增加你自己的思考内容，作为总结。

- 知识库内容 -
{context.strip()}

- 问题 -
"{query}"
"""
    return prompt


# prompt = web_search_pachong("python")
# print(prompt)
def web_search_DDG(query):
    search_results = DDGS().text(query, max_results=5)
    context = format_search_results(search_results, 'DDG')
    # prompt = get_prompt(context, query)
    return context


def web_search_dashscope(query):
    try:
        response = Application.call(
            api_key="sk-e4f34f0c88d8467c9cafe7c1d1f7774a",
            app_id='4a5ecea0707346658f90a2e19a5a92ec',
            prompt=query)

        # 检查响应状态码
        if response.status_code != HTTPStatus.OK:
            print(f'Error: request_id={response.request_id}')
            print(f'Error Code: {response.status_code}')
            print(f'Error Message: {response.message}')
            print(
                f'Please refer to the documentation for more details: https://help.aliyun.com/zh/model-studio/developer-reference/error-code')
            return None

        # 尝试解析返回的 JSON 数据
        try:
            # print(response.output.text)
            response_data = json.loads(response.output.text)
            print(response_data)
            context = format_search_results(response_data, 'dashscope')
            prompt = get_prompt(context, query)
            return prompt
        except json.JSONDecodeError as e:
            print(f"Failed to parse response as JSON: {e}")
            return None

    except requests.exceptions.RequestException as e:
        # 捕获请求异常
        print(f"Network error: {e}")
        return None
    except Exception as e:
        # 捕获其他可能的异常
        print(f"An unexpected error occurred: {e}")
        return None


def baidu_baike_scraper(query):
    # 对 query 进行 URL 编码
    query_encoded = urllib.parse.quote(query)
    search_url = f"https://baike.baidu.com/search?enc=utf8&word={query_encoded}"

    # 设置 Selenium 使用无头模式（即不弹出浏览器窗口）
    options = Options()
    options.add_argument("--headless")  # 启用无头模式
    options.add_argument("--disable-extensions")  # 禁用扩展
    options.add_argument("--disable-gpu")  # 禁用 GPU，加速渲染
    options.add_argument("--no-sandbox")  # 沙盒模式
    options.add_argument("--disable-dev-shm-usage")  # 解决 DevToolsActivePort 文件不存在的错误
    options.add_argument("start-maximized")  # 窗口最大化
    options.add_argument("disable-infobars")  # 禁止浏览器提示条
    options.add_argument('--disable-web-security')  # 禁用安全性
    options.add_argument('--disable-features=IsolateOrigins,site-per-process')  # 禁用 CSP
    options.add_argument("--no-proxy-server")
    options.add_argument("--proxy-server='direct://'")
    # 启动 Selenium WebDriver（Chrome）
    try:
        driver = webdriver.Chrome(options=options)
    except Exception as e:
        print(f"Error launching Chrome WebDriver: {e}")
        return None

    # 访问百度搜索结果页面
    driver.get(search_url)
    driver.get(search_url)
    # 等待页面加载完成，确保结果完全加载
    try:
        # 等待页面中显示以 'container_' 开头的元素加载完成
        WebDriverWait(driver, 3).until(
            EC.presence_of_element_located((By.XPATH, "//*[starts-with(@class, 'container_6csHE')]"))
        )

    except Exception as e:
        print(f"Error waiting for page to load: {e}")
        driver.quit()
        return None

    # 获取页面内容
    soup = BeautifulSoup(driver.page_source, 'html.parser')

    # 获取搜索结果项（根据新的 HTML 结构更新类名）
    # result_items = soup.find_all('div', class_='container_EB0wZ')  # 使用正确的类名
    result_items = soup.find_all('div', class_=re.compile('^container_6csHE'))

    if not result_items:
        print("未找到搜索结果项，可能是页面结构改变了")
        driver.quit()  # 关闭浏览器
        return None

    results = []
    for item in result_items[:3]:  # 获取前五个结果
        # title = item.find('a', class_='title_Qq6En')  # 更新为正确的类名
        title = item.find('a', class_=re.compile('^title_'))
        title_text = title.text if title else "No title"
        url = title['href'] if title else "No URL"

        # 查找两个可能的类名
        # description = item.find('p', class_='abstract_rC2eN')
        description = item.find('p', class_=re.compile('^(abstract_|summary_)'))

        description_text = description.text if description else "No description"

        results.append({
            "title": title_text,
            "url": url,
            "description": description_text
        })
    context = format_search_results(results, 'baidubaike')

    driver.quit()  # 关闭浏览器
    return context


def pachong(query):
    # 对 query 进行 URL 编码
    query_encoded = urllib.parse.quote(query)
    search_url = f"https://www.baidu.com/s?wd={query_encoded}"

    # 设置 Selenium 使用无头模式（即不弹出浏览器窗口）
    options = Options()
    options.add_argument("--headless")  # 启用无头模式
    options.add_argument("--disable-extensions")  # 禁用扩展
    options.add_argument("--disable-gpu")  # 禁用 GPU，加速渲染
    options.add_argument("--no-sandbox")  # 沙盒模式
    options.add_argument("--disable-dev-shm-usage")  # 解决 DevToolsActivePort 文件不存在的错误
    options.add_argument("start-maximized")  # 窗口最大化
    options.add_argument("disable-infobars")  # 禁止浏览器提示条
    options.add_argument('--no-proxy-server')
    options.add_argument('--proxy-server=')

    # 启动 Selenium WebDriver（Chrome）
    try:
        driver = webdriver.Chrome(options=options)
    except Exception as e:
        print(f"Error launching Chrome WebDriver: {e}")
        return None

    # 访问百度搜索结果页面
    driver.get(search_url)

    # 等待页面加载完成
    time.sleep(1)  # 可以根据网络情况调整等待时间

    # 获取页面内容
    soup = BeautifulSoup(driver.page_source, 'html.parser')

    # 获取搜索结果项
    result_items = soup.find_all('div', class_='result c-container xpath-log new-pmd')

    if not result_items:
        print("未找到搜索结果项，可能是页面结构改变了")
        driver.quit()  # 关闭浏览器
        return None

    results = []
    for item in result_items[:3]:  # 获取前五个结果
        title = item.find('h3').text if item.find('h3') else "No title"
        url = item.find('a')['href'] if item.find('a') else "No URL"

        # 查找两个可能的类名
        description = item.find('span', class_='content-right_1THTn') or item.find('span', class_='content-right_2s-H4')
        description_text = description.text if description else "No description"

        results.append({
            "title": title,
            "url": url,
            "description": description_text
        })

    driver.quit()  # 关闭浏览器
    context = format_search_results(results, 'pachong')
    # prompt = get_prompt(context, query)
    # return prompt
    return context


def web_search_pachong(keywords_list):
    keyword_results = []
    for keyword in keywords_list:
        print(f"正在爬取百度搜索关键词: {keyword}")
        results = pachong(keyword)
        keyword_results.append(results)
    return keyword_results


def web_search_baidubaike(keywords_list):
    keyword_results = []
    for keyword in keywords_list:
        print(f"正在爬取百度百科关键词: {keyword}")
        results = baidu_baike_scraper(keyword)
        keyword_results.append(results)
    return keyword_results


def connect_to_db():
    try:
        connection = pymysql.connect(
            host='www.zhonghuapu.com',  # 数据库主机
            user='koroot',  # 数据库用户名
            password='DMiC-4092',  # 数据库密码
            database='db_hp',
            charset='utf8'
        )
        return connection
    except pymysql.MySQLError as err:
        print(f"Error: {err}")
        return None


def check_wname_exists(connection, wname, ksource):
    """检查指定的 wname-ksource是否已存在于数据库中。"""
    try:
        with connection.cursor() as cursor:
            query = "SELECT COUNT(*) FROM tb_zhihaiqa_bdwebsearch WHERE wname = %s AND ksource = %s"
            cursor.execute(query, (wname, ksource,))
            result = cursor.fetchone()
            return result[0] > 0  # 如果存在记录，返回 True
    except pymysql.MySQLError as err:
        print(f"Error while checking wname: {err}")
        return False


def get_kdescription_by_wname(connection, wname, ksource):
    try:
        with connection.cursor() as cursor:
            # 查询语句，查找所有符合 name 的记录
            query = """
                    SELECT ktitle, kurl, kdescription
                    FROM tb_zhihaiqa_bdwebsearch
                    WHERE wname = %s AND ksource = %s \
                    """
            cursor.execute(query, (wname, ksource,))
            result = cursor.fetchall()  # 获取所有匹配的结果

            if result:
                all_record = [{
                    "title": row[0],
                    "url": row[1],
                    "description": row[2]
                } for row in result]  # 获取每行的记录
                return all_record
            else:
                print(f"No records found for name '{wname}' in source {ksource}.")
                return []
    except pymysql.MySQLError as err:
        print(f"Error while querying data: {err}")
        return []


def insert_bdwebsearch(connection, data_list, keyword, ksource):
    try:
        with connection.cursor() as cursor:
            # 插入语句，不再涉及id
            insert_query = """
                           INSERT INTO tb_zhihaiqa_bdwebsearch (wname, ksource, ktitle, kurl, kdescription)
                           VALUES (%s, %s, %s, %s, %s) \
                           """
            for data in data_list:
                # 插入数据
                new_data = (keyword, ksource, data["title"], data["url"], data["description"])
                cursor.execute(insert_query, new_data)
            connection.commit()  # 提交所有插入操作
            print("All data inserted successfully.")
    except pymysql.MySQLError as err:
        print(f"Error while inserting data: {err}")
        connection.rollback()  # 如果出错，回滚事务


def web_search_pachong_with_memo(keywords_list):
    keyword_results = []
    connection = connect_to_db()
    for keyword in keywords_list:
        if check_wname_exists(connection, keyword, "bdsearch"):
            print(f"关键词 {keyword} 命中缓存")
            raw_data = get_kdescription_by_wname(connection, keyword, "bdsearch")
            results = format_search_results(raw_data, "pachong")
            keyword_results.append(results)
        else:
            print(f"正在爬取百度搜索关键词: {keyword}")
            results = pachong(keyword)
            if results:
                raw_data = parse_search_results(results)
                insert_bdwebsearch(connection, raw_data, keyword, "bdsearch")
                keyword_results.append(results)
    connection.close()
    return keyword_results


def web_search_baidubaike_with_memo(keywords_list):
    keyword_results = []
    connection = connect_to_db()
    for keyword in keywords_list:
        if check_wname_exists(connection, keyword, "bdbaike"):
            print(f"关键词 {keyword} 命中缓存")
            raw_data = get_kdescription_by_wname(connection, keyword, "bdbaike")
            results = format_search_results(raw_data, "baidubaike")
            keyword_results.append(results)
        else:
            print(f"正在爬取百度百科关键词: {keyword}")
            results = baidu_baike_scraper(keyword)
            if results:
                raw_data = parse_search_results(results)
                insert_bdwebsearch(connection, raw_data, keyword, "bdbaike")
                keyword_results.append(results)
    connection.close()
    return keyword_results


@web_of_search_router.post("/web_of_search")
async def web_of_search(request: Request):
    """
    接收前端传来的问题并返回
    :param request:
    :param req_data: 包含问题的请求数据
    :return: 返回接收到的问题
    """
    query_data = await request.json()
    userid = query_data['userid']
    question = query_data['query']
    query = question
    prompt = web_search_pachong(query)
    print(prompt)
    q_history = [{"role": "user", "content": prompt}]
    return StreamingResponse(
        get_stream(q_history),
        media_type="text/event-stream"
    )


if __name__ == '__main__':
    # 调用函数获取数据
    query = ['变压器']
    # query = ['吴信东和吴共庆有什么区别？']
    # prompt = web_search_pachong(query)
    # print(prompt)
    context = web_search_baidubaike(query)
    # context = pachong("吴信东")
    print(context)

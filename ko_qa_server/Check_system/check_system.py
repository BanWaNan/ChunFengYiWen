import pymysql
import requests
import smtplib
from email.mime.text import MIMEText
from email.header import Header
from datetime import datetime
from neo4j import GraphDatabase
from email.utils import formataddr
from fastapi import APIRouter
from fastapi.responses import StreamingResponse, JSONResponse

check_system_router = APIRouter()
# ---------- 接口配置 ----------
API_URL = "http://ko.zhonghuapu.com:8989/getTT"
API_METHOD = "post"  # "get" 或 "post"
EXPECTED_FIELD_PATH = ["response", "answer"]


# ---------- 数据库配置 ----------
DB_CONFIG = {
    "host": "www.zhonghuapu.com",
    "user": "koroot",
    "password": "DMiC-4092",
    "database": "db_hp",
    "charset": "utf8"
}


# ---------- 邮件配置 ----------
SMTP_HOST = "smtp.163.com"
MAIL_USER = "zhangxiaolongwork@163.com"
MAIL_PASS = "KJSK9fXXDb3chSh7"
TO_ADDR = "3010064861@qq.com"


# ---------- Neo4j 配置 ----------
uri = "bolt://114.213.232.140:7687"
username = "neo4j"
password = "DMiChao"
kdzn_uri = "bolt://114.213.232.140:37687"
kdzn_username = "neo4j"
kdzn_password = "123456"


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


# ---------- 邮件发送 ----------
def send_alert_email(subject, message):
    msg = MIMEText(message, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = MAIL_USER
    msg["To"] = TO_ADDR
    try:
        smtp = smtplib.SMTP_SSL(SMTP_HOST, 465)
        smtp.login(MAIL_USER, MAIL_PASS)
        smtp.sendmail(MAIL_USER, [TO_ADDR], msg.as_string())
        smtp.quit()
        print("✅ 邮件发送成功")
    except Exception as e:
        print(f"❌ 邮件发送失败: {str(e)}")




# ---------- 功能函数 ----------
def check_api():
    try:
        if API_METHOD.lower() == "post":
            response = requests.post(API_URL, json={})
        else:
            response = requests.get(API_URL)
        if response.status_code != 200:
            raise Exception(f"接口状态码异常: {response.status_code}")
        data = response.json()
        # 按层次判断字段是否存在
        val = data
        for key in EXPECTED_FIELD_PATH:
            val = val[key]
        if not val:
            raise Exception("接口返回字段值异常：answer 为 false")
        return None  # 无异常
    except Exception as e:
        return f"接口异常：{str(e)}"


def check_db():
    try:
        connection = pymysql.connect(**DB_CONFIG)
        connection.close()
        return None
    except pymysql.MySQLError as e:
        return f"数据库异常：{str(e)}"


# ---------- 执行 Neo4j 查询 ----------
def query_neo4j(query, timeout=30, kg='kdzn'):
    """执行neo4j查询"""
    try:
        if kg == 'kdzn':
            driver = GraphDatabase.driver(kdzn_uri, auth=(kdzn_username, kdzn_password))
        else:
            driver = GraphDatabase.driver(uri, auth=(username, password))
        query_ = f"""CALL apoc.cypher.runTimeboxed('{query}',NULL,{timeout * 1000})"""
        res = []
        with driver.session() as session:
            result = session.run(query_)
            for record in result:
                res.append(record)
        driver.close()
        return res
    except Exception as e:
        raise RuntimeError(f"Neo4j 查询异常：{str(e)}")


# ---------- 检查 Neo4j ----------
def check_neo4j():
    try:
        query_neo4j("RETURN 1", timeout=10)
        return None
    except Exception as e:
        return str(e)


@check_system_router.get("/check_system")
def check_system():
    errors = []
    api_result = check_api()
    if api_result:
        errors.append(api_result)

    db_result = check_db()
    if db_result:
        errors.append(db_result)

    neo4j_result = check_neo4j()
    if neo4j_result:
        errors.append(f"Neo4j 异常：{neo4j_result}")

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if errors:
        msg = f"""❌ 系统检测异常
时间: {now}
问题如下：
{chr(10).join(f'- {err}' for err in errors)}
"""
        print(msg)
        send_alert_email("系统异常告警", msg)
        return JSONResponse(content={"status": "error", "time": now, "errors": errors}, status_code=500)
    else:
        return {"status": "ok", "time": now}
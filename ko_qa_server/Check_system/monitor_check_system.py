import requests
import smtplib
from email.mime.text import MIMEText
from datetime import datetime

# --------- 监控配置 ----------
CHECK_SYSTEM_URL = "http://localhost:8989/check_system"  # 改成实际部署地址
TIMEOUT = 10  # 超时时间

# --------- 邮件配置 ----------
SMTP_HOST = "smtp.163.com"
MAIL_USER = "zhangxiaolongwork@163.com"
MAIL_PASS = "KJSK9fXXDb3chSh7"
TO_ADDR = "3010064861@qq.com"


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
        print("✅ 邮件已发送")
    except Exception as e:
        print(f"❌ 邮件发送失败: {str(e)}")


def check_health_endpoint():
    try:
        resp = requests.get(CHECK_SYSTEM_URL, timeout=TIMEOUT)
        if resp.status_code != 200:
            raise Exception(f"接口状态码异常：{resp.status_code}")
        print("✅ /check_system 接口正常")
    except Exception as e:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        msg = f"""🚨 接口 /check_system 异常！
时间: {now}
错误信息: {str(e)}
说明: 健康检查接口本身不可用，可能 FastAPI 服务未启动或已崩溃，请立即排查。
"""
        print(msg)
        send_alert_email("🚨 系统健康检查接口异常", msg)


if __name__ == "__main__":
    check_health_endpoint()

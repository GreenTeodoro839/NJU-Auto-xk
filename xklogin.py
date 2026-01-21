import requests
import json
import base64
import time
import os
import io
from PIL import Image  # pip install pillow
from serverchan import send_serverchan_notification

OCR = 0 # 1为超级鹰，0为YOLO
if OCR == 1:
    from captcha_cjy import solve_captcha_from_base64
else:
    from captcha_yolo import solve_captcha_from_base64

# ================= 配置加载 =================
def load_config(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"配置文件 {path} 不存在")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ================= 核心登录逻辑 =================
def login(conf_path="xk.conf"):
    # 1. 读取基础配置
    conf = load_config(conf_path)
    username = conf.get("USER")
    password = conf.get("PWD_ENCRYPT")
    max_retries = int(conf.get("MAX_RETRIES", 3))

    # === 新增：读取代理配置 ===
    proxy_url = conf.get("PROXY")

    if not username or not password:
        raise ValueError("配置文件中缺少 USER 或 PWD_ENCRYPT")

    BASE_URL = "https://xk.nju.edu.cn/xsxkapp/sys/xsxkapp"
    INDEX_URL = f"{BASE_URL}/*default/index.do"
    VCODE_API = f"{BASE_URL}/student/4/vcode.do"
    LOGIN_API = f"{BASE_URL}/student/check/login.do"

    # 初始化 Session
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": INDEX_URL,
        "Origin": "https://xk.nju.edu.cn",
        "X-Requested-With": "XMLHttpRequest"
    })

    # === 新增：应用代理设置 ===
    if proxy_url and proxy_url.strip():
        print(f">>> 启用代理: {proxy_url}")
        session.proxies.update({
            "http": proxy_url,
            "https": proxy_url
        })
    else:
        print(">>> 未配置代理，使用直连模式")

    # 开始循环尝试
    for attempt in range(max_retries):
        try:
            print(f"\n====== 尝试第 {attempt + 1}/{max_retries} 次登录 ======")

            # Step 1. 访问主页 (Session 初始化)
            print(">>> 1. 初始化 Session...")
            session.get(INDEX_URL, timeout=10)

            # Step 2. 获取验证码 (GIF Base64)
            print(">>> 2. 获取验证码...")
            ts = str(int(time.time() * 1000))
            vcode_resp = session.post(f"{VCODE_API}", timeout=10)
            vcode_json = vcode_resp.json()

            data_node = vcode_json.get("data", {})
            server_uuid = data_node.get("uuid")
            img_b64_raw = data_node.get("vode") or data_node.get("vcode")

            if not server_uuid or not img_b64_raw:
                print(f"❌ 响应数据不完整: {vcode_json}")
                continue

            # 去除前缀
            if "," in img_b64_raw:
                img_gif_b64_body = img_b64_raw.split(",")[1]
            else:
                img_gif_b64_body = img_b64_raw

            # Step 3. 格式转换: GIF -> JPG
            # print(">>> 3. 正在转换图片格式 (GIF -> JPG)...")
            try:
                gif_bytes = base64.b64decode(img_gif_b64_body)
                img = Image.open(io.BytesIO(gif_bytes))

                if img.mode != 'RGB':
                    img = img.convert('RGB')

                jpg_buffer = io.BytesIO()
                img.save(jpg_buffer, format='JPEG', quality=95)
                img_jpg_b64 = base64.b64encode(jpg_buffer.getvalue()).decode('utf-8')

            except Exception as e:
                print(f"❌ 图片转换失败: {e}")
                continue

            # Step 4. 外部识别
            print(">>> 3. 识别验证码 (JPG Base64)...")
            points = solve_captcha_from_base64(img_jpg_b64)

            if not points:
                print(f"❌ 识别失败")
                continue

            coord_str_list = [f"{int(p[0])}-{int(p[1] * 5 / 6)}" for p in points]
            verify_code = ",".join(coord_str_list)
            print(f"    提交坐标: {verify_code}")

            # Step 5. 登录
            payload = {
                "loginName": username,
                "loginPwd": password,
                "verifyCode": verify_code,
                "vtoken": "",
                "uuid": server_uuid
            }

            print(">>> 4. 发送登录请求...")
            login_resp = session.post(LOGIN_API, data=payload, timeout=15)
            login_json = login_resp.json()

            # Step 6. 结果校验
            resp_code = login_json.get("code")
            resp_data = login_json.get("data") or {}

            returned_number = resp_data.get("number")

            if str(resp_code) == "1" and str(returned_number) == str(username):
                token = resp_data.get("token")
                print(f"✅ 登录成功! Token: {token}")

                return session.cookies.get_dict(), token
            else:
                msg = login_json.get("msg", "未知错误")
                print(f"❌ 登录失败: {msg} (Code: {resp_code})")

        except Exception as e:
            # 打印详细错误信息
            import traceback
            print(f"❌ 异常: {e}")
            # traceback.print_exc() # 调试时可开启
            time.sleep(1)

    send_serverchan_notification("❌ 登录失败", "🚫 登录失败，已达最大重试次数。")
    print("🚫 登录失败，已达最大重试次数。")
    return None, None


if __name__ == "__main__":
    # 确保当前目录下有 xk.conf，并且包含 USER, PWD, (可选 PROXY)
    c, t = login()
    if t:
        print("Final Token:", t)
import os
import base64
import io
import re
import json
import requests
from hashlib import md5
from PIL import Image
from openai import OpenAI

# ================= 常量定义 =================

# 【配置文件】
CONF_FILE_NAME = "xk.conf"

# 【超级鹰】
CJY_CODETYPE_WORD_LOCATE = 9801
# 超级鹰文档建议分辨率不超过 500px（越大正确率越低）
CJY_TOP_MAX_PX = 500
CAPTCHA_SPLIT_RATIO = 0.83  # 切图比例：Top(点选区):Bottom(题目区) = 8:2


# ================= 类与辅助函数 =================

class ChaojiyingClient:
    """超级鹰 HTTP 标准接口封装"""
    API_PROCESSING = "https://upload.chaojiying.net/Upload/Processing.php"

    def __init__(self, username: str, password: str, soft_id: str | int):
        self.username = username
        self.password_md5 = md5(password.encode("utf-8")).hexdigest()
        self.soft_id = str(soft_id)
        self.base_params = {
            "user": self.username,
            "pass2": self.password_md5,
            "softid": self.soft_id
        }
        self.headers = {
            "Connection": "Keep-Alive",
            "User-Agent": "Mozilla/5.0",
        }
        self.session = requests.Session()

    def post_pic(self, img_bytes: bytes, codetype: int, *, str_debug: str = "") -> dict:
        params = {"codetype": str(codetype)}
        if str_debug:
            params["str_debug"] = str_debug
        params.update(self.base_params)
        files = {"userfile": ("captcha.jpg", img_bytes)}
        # 超级鹰偶尔响应慢，设置较大超时
        r = self.session.post(
            self.API_PROCESSING,
            data=params,
            files=files,
            headers=self.headers,
            timeout=60
        )
        return r.json()


def load_xk_config(conf_path: str | None = None) -> dict:
    """读取配置文件"""
    if conf_path is None:
        conf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), CONF_FILE_NAME)
    if not os.path.exists(conf_path):
        raise FileNotFoundError(f"配置文件未找到: {conf_path}")
    with open(conf_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _conf_required(conf: dict, key: str):
    val = conf.get(key)
    if not val:
        raise ValueError(f"配置文件缺少参数: {key}")
    return str(val).strip()


def process_captcha_images(full_image: Image.Image):
    """
    切图逻辑：
    1. Top(83%): 点选区 -> 缩放到 CJY_TOP_MAX_PX 限制内 -> 转 bytes 给超级鹰
    2. Bottom(17%): 题目区 -> 放大2倍 -> 转 base64 给大模型
    """
    w, h = full_image.size
    split_y = int(h * CAPTCHA_SPLIT_RATIO)

    # --- 1) Top: 点选区 ---
    img_top = full_image.crop((0, 0, w, split_y))
    top_w_orig, top_h_orig = img_top.size

    # 缩放逻辑
    max_dim = max(top_w_orig, top_h_orig)
    if max_dim > CJY_TOP_MAX_PX:
        scale = CJY_TOP_MAX_PX / max_dim
        new_w = max(1, int(top_w_orig * scale))
        new_h = max(1, int(top_h_orig * scale))
        img_top = img_top.resize((new_w, new_h), Image.Resampling.LANCZOS)

    top_w_resized, top_h_resized = img_top.size

    buf_top = io.BytesIO()
    img_top.save(buf_top, format="JPEG", quality=95)
    top_img_bytes = buf_top.getvalue()

    # --- 2) Bottom: 题目区 ---
    img_bottom = full_image.crop((0, split_y, w, h))
    w_bot, h_bot = img_bottom.size
    img_bottom = img_bottom.resize((w_bot * 2, h_bot * 2), Image.Resampling.LANCZOS)

    buf_bot = io.BytesIO()
    img_bottom.save(buf_bot, format="JPEG", quality=90)
    bottom_b64 = base64.b64encode(buf_bot.getvalue()).decode("utf-8")

    return top_img_bytes, (top_w_resized, top_h_resized), bottom_b64


def get_prompt_text(client: OpenAI, llm_model: str, bottom_b64: str) -> str:
    """使用大模型识别题目文字"""
    prompt = "直接输出括号【】内的汉字。例如‘依次点击【事 论】’->输出‘事论’。不要其他符号。"
    try:
        completion = client.chat.completions.create(
            model=llm_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{bottom_b64}"}},
                    ],
                }
            ],
        )
        text = completion.choices[0].message.content.strip()
        # 清洗非汉字字符
        clean_text = re.sub(r"[^一-龥]", "", text)
        return clean_text
    except Exception as e:
        raise Exception(f"大模型读题失败: {e}")


def _build_cjy_8a_str_debug(target_chars: str) -> str:
    """构造 9801 的 8a 指令"""
    chars = [c for c in target_chars if "\u4e00" <= c <= "\u9fff"]
    return "{8a:" + ",".join(chars) + "/8a}"


def get_coordinate_selection_cjy(chaojiying: ChaojiyingClient, top_img_bytes: bytes, target_chars: str):
    """请求超级鹰获取坐标（相对于 top_img_bytes 的尺寸）"""
    str_debug = _build_cjy_8a_str_debug(target_chars)
    if str_debug in ("{8a:/8a}", ""):
        raise Exception("识别到的目标汉字为空")

    resp = chaojiying.post_pic(top_img_bytes, CJY_CODETYPE_WORD_LOCATE, str_debug=str_debug)

    if not isinstance(resp, dict):
        raise Exception(f"超级鹰返回异常: {resp}")

    err_no = resp.get("err_no")
    if err_no != 0:
        raise Exception(f"超级鹰识别失败: err_no={err_no}, err_str={resp.get('err_str')}")

    pic_str = str(resp.get("pic_str", ""))

    # 解析坐标
    points = []
    for part in pic_str.split("|"):
        part = part.strip()
        if not part or "," not in part:
            continue
        x_str, y_str = part.split(",", 1)
        points.append((float(x_str), float(y_str)))

    expected_len = len([c for c in target_chars if "\u4e00" <= c <= "\u9fff"])
    if len(points) != expected_len:
        print(f"⚠️ 警告: 题目文字长度({expected_len})与识别坐标数({len(points)})不一致")

    return points


# ================= 核心暴露接口 =================

def solve_captcha_from_base64(base64_img: str, conf_path: str | None = None) -> list[tuple[int, int]]:
    """
    核心功能函数：传入Base64图片，返回识别后的坐标列表。

    Args:
        base64_img: JPG图片的Base64编码字符串（不带 data:image/jpeg;base64, 前缀）
        conf_path: 配置文件路径，默认读取同级目录下的 xk.conf

    Returns:
        list[tuple[int, int]]: 坐标列表，例如 [(120, 50), (300, 80)]。
                               坐标系原点为传入图片左上角。
    """

    # 1. 加载配置
    conf = load_xk_config(conf_path)

    llm_key = _conf_required(conf, "LLM_KEY")
    llm_model = _conf_required(conf, "LLM_MODEL")
    llm_base_url = _conf_required(conf, "LLM_BASE_URL")

    cjy_user = _conf_required(conf, "CJY_USER")
    cjy_pass = _conf_required(conf, "CJY_PASS")
    cjy_softid = _conf_required(conf, "CJY_SOFTID")

    # 2. 初始化客户端
    client = OpenAI(api_key=llm_key, base_url=llm_base_url)
    chaojiying = ChaojiyingClient(cjy_user, cjy_pass, cjy_softid)

    # 3. 处理图片
    try:
        img_data = base64.b64decode(base64_img)
        full_image = Image.open(io.BytesIO(img_data))
    except Exception as e:
        raise ValueError(f"无效的 Base64 图片数据: {e}")

    # 获取原图尺寸（用于最后映射回原图坐标）
    orig_w, orig_h = full_image.size

    # 切分并缩放（返回的 bytes 是缩放后的 top 区，尺寸也是缩放后的）
    top_img_bytes, (resized_top_w, resized_top_h), bottom_b64 = process_captcha_images(full_image)

    # 4. 大模型读题
    target_text = get_prompt_text(client, llm_model, bottom_b64)
    print(f"识别题目: {target_text}")

    if not target_text:
        return []

    # 5. 超级鹰找坐标 (返回的是相对于 resized_top_w/h 的坐标)
    raw_points = get_coordinate_selection_cjy(chaojiying, top_img_bytes, target_text)

    # 6. 坐标映射还原 (Resized Top -> Original Top -> Original Image)
    # 计算缩放比例
    # 注意：top 区域的高度在原图中是 orig_h * CAPTCHA_SPLIT_RATIO
    orig_top_h = int(orig_h * CAPTCHA_SPLIT_RATIO)
    orig_top_w = orig_w

    scale_x = orig_top_w / float(resized_top_w)
    scale_y = orig_top_h / float(resized_top_h)

    final_coords = []
    for (rx, ry) in raw_points:
        # 映射回原图尺寸
        x_orig = rx * scale_x
        y_orig = ry * scale_y

        # 转换为整数像素坐标
        final_coords.append((int(round(x_orig)), int(round(y_orig))))

    return final_coords


if __name__ == "__main__":
    # 简单的本地测试逻辑
    import sys

    if len(sys.argv) < 2:
        print("Usage: python script.py <path_to_captcha.jpg>")
    else:
        img_path = sys.argv[1]
        with open(img_path, "rb") as f:
            b64_str = base64.b64encode(f.read()).decode("utf-8")

        try:
            results = solve_captcha_from_base64(b64_str)
            print(f"Result Coords: {results}")
        except Exception as e:
            print(f"Error: {e}")
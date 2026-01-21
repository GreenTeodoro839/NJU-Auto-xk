import os
import json
import io
import re
import base64
import numpy as np
from typing import Any, List, Tuple, Dict
from PIL import Image

# 尝试导入 YOLO，如果未安装给出提示
try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("请先安装 YOLO 依赖: pip install ultralytics")

try:
    from openai import OpenAI
except ImportError:
    raise ImportError("请先安装 OpenAI 依赖: pip install openai")

# ================= 常量定义 =================

CONF_FILE_NAME = "xk.conf"
# Top(点选区):Bottom(题目区) = 5:1 => 约 0.83
CAPTCHA_SPLIT_RATIO = 0.83
TOP_MAX_PX = 500  # YOLO检测时的缩放基准
SLICE_PAD_PX = 8
SLICE_UPSCALE = 3
YOLO_DEFAULT_CONF = 0.25

# 全局变量用于缓存 YOLO 模型，避免每次调用都重新加载
_CACHED_YOLO_MODEL = None


# ================= 辅助函数 =================

def load_xk_config(conf_path: str | None = None) -> dict:
    if conf_path is None:
        # 默认在当前文件目录下寻找 xk.conf
        conf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), CONF_FILE_NAME)

    if not os.path.exists(conf_path):
        raise FileNotFoundError(f"配置文件未找到: {conf_path}")

    with open(conf_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_yolo_model(model_path: str):
    global _CACHED_YOLO_MODEL
    if _CACHED_YOLO_MODEL is None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YOLO 模型文件未找到: {model_path}")
        print(f"⏳ 正在加载 YOLO 模型: {model_path} ...")
        _CACHED_YOLO_MODEL = YOLO(model_path)
    return _CACHED_YOLO_MODEL


def extract_json_from_text(text: str):
    """从大模型返回的文本中提取 JSON"""
    try:
        match = re.search(r"```json\s*(.*?)```", text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        match = re.search(r"(\{.*\})", text, re.DOTALL) or re.search(r"(\[.*\])", text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        return None
    except Exception:
        return None


def _resize_keep_aspect(img: Image.Image, max_dim: int) -> Tuple[Image.Image, float]:
    """调整图片大小并返回缩放比例 (scale = new / old)"""
    w, h = img.size
    if max(w, h) <= max_dim:
        return img, 1.0
    scale = max_dim / float(max(w, h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return img.resize((new_w, new_h), Image.Resampling.LANCZOS), scale


def _unique_chinese_chars(text: str) -> List[str]:
    chars = [c for c in text if "\u4e00" <= c <= "\u9fff"]
    out: List[str] = []
    for c in chars:
        if c not in out:
            out.append(c)
    return out


# ================= 核心处理逻辑 =================

def process_captcha_split(full_image: Image.Image):
    """分割验证码为 上部(点选区) 和 下部(题目区)"""
    w, h = full_image.size
    split_y = int(h * CAPTCHA_SPLIT_RATIO)

    # 上部图片处理
    img_top_original = full_image.crop((0, 0, w, split_y))
    # 缩放上部图片用于 YOLO 检测，并记录缩放比例以便还原坐标
    img_top_resized, scale_factor = _resize_keep_aspect(img_top_original, TOP_MAX_PX)

    buf_top = io.BytesIO()
    img_top_resized.save(buf_top, format="JPEG", quality=95)
    top_img_bytes = buf_top.getvalue()

    # 下部图片处理 (题目)
    img_bottom = full_image.crop((0, split_y, w, h))
    w_bot, h_bot = img_bottom.size
    # 放大下部图片以提高 OCR/LLM 识别率
    img_bottom = img_bottom.resize((w_bot * 2, h_bot * 2), Image.Resampling.LANCZOS)
    buf_bot = io.BytesIO()
    img_bottom.save(buf_bot, format="JPEG", quality=90)
    bottom_b64 = base64.b64encode(buf_bot.getvalue()).decode("utf-8")

    return top_img_bytes, scale_factor, bottom_b64


def get_prompt_text(client: OpenAI, llm_model: str, bottom_b64: str) -> str:
    """利用 LLM 识别题目文字"""
    prompt = "直接输出括号【】内的汉字。例如‘依次点击【事 论】’->输出‘事论’。不要其他符号。"
    try:
        completion = client.chat.completions.create(
            model=llm_model,
            temperature=0,
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
        clean_text = re.sub(r"[^一-龥]", "", text)
        return clean_text
    except Exception as e:
        print(f"读题失败: {e}")
        return ""


def yolo_detect_boxes(yolo_model, top_img_bytes: bytes, conf_thres: float):
    """运行 YOLO 检测汉字框"""
    img = Image.open(io.BytesIO(top_img_bytes)).convert("RGB")
    arr_rgb = np.array(img)
    arr_bgr = arr_rgb[:, :, ::-1]  # OpenCV/YOLO 使用 BGR

    results = yolo_model.predict(source=arr_bgr, save=False, conf=conf_thres, verbose=False)
    if not results:
        return []

    r0 = results[0]
    if r0.boxes is None or len(r0.boxes) == 0:
        return []

    xyxy = r0.boxes.xyxy.cpu().numpy().tolist()
    confs = r0.boxes.conf.cpu().numpy().tolist()

    dets = []
    for bbox, c in zip(xyxy, confs):
        x1, y1, x2, y2 = [float(v) for v in bbox]
        dets.append({"bbox": [x1, y1, x2, y2], "conf": float(c)})

    # 按置信度降序，Y坐标，X坐标排序
    dets.sort(key=lambda d: (-d["conf"], d["bbox"][1], d["bbox"][0]))
    return dets


def crop_slices(top_img_bytes: bytes, dets: List[dict], pad: int = SLICE_PAD_PX) -> List[dict]:
    """根据检测框切出单个汉字图片"""
    img = Image.open(io.BytesIO(top_img_bytes)).convert("RGB")
    w, h = img.size
    slices = []

    for i, d in enumerate(dets, start=1):
        x1, y1, x2, y2 = d["bbox"]
        # 增加一点 padding
        x1i = max(0, int(np.floor(x1)) - pad)
        y1i = max(0, int(np.floor(y1)) - pad)
        x2i = min(w, int(np.ceil(x2)) + pad)
        y2i = min(h, int(np.ceil(y2)) + pad)

        if x2i <= x1i or y2i <= y1i:
            continue

        crop = img.crop((x1i, y1i, x2i, y2i))
        # 放大切片以便 LLM 识别
        if SLICE_UPSCALE > 1:
            cw, ch = crop.size
            crop = crop.resize((max(1, cw * SLICE_UPSCALE), max(1, ch * SLICE_UPSCALE)), Image.Resampling.LANCZOS)

        buf = io.BytesIO()
        crop.save(buf, format="JPEG", quality=95)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        # 计算在 Resized Image 上的中心点
        cx = (float(x1) + float(x2)) / 2.0
        cy = (float(y1) + float(y2)) / 2.0

        slices.append({
            "i": i,
            "bbox": [x1, y1, x2, y2],
            "center_resized": [cx, cy],  # 这里的坐标还是基于 resized 图片的
            "conf": d["conf"],
            "b64": b64
        })
    return slices


def llm_match_slices(client: OpenAI, llm_model: str, target_text: str, slices: List[dict]) -> Dict[int, str]:
    """让 LLM 识别切片属于哪个候选字"""
    candidates_unique = _unique_chinese_chars(target_text)
    if not candidates_unique or not slices:
        return {}

    n = len(slices)
    prompt = (
        "你将看到 N 张验证码中的单个汉字切片图片，编号为 1..N。\n"
        f"题目要求依次点击的汉字序列为：{target_text}\n"
        f"候选汉字集合：{''.join(candidates_unique)}\n"
        "任务：对每个切片 i，判断它是哪一个候选汉字；如果不属于候选集合，输出 '未知'。\n"
        "输出格式：{\"items\":[{\"i\":1,\"char\":\"…\"},...],\"n\":N}"
    )

    content: List[dict] = [{"type": "text", "text": prompt.replace("N", str(n))}]
    for s in slices:
        content.append({"type": "text", "text": f"切片#{s['i']}"})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{s['b64']}"}})

    try:
        completion = client.chat.completions.create(
            model=llm_model,
            temperature=0,
            messages=[{"role": "user", "content": content}],
        )
        raw = completion.choices[0].message.content.strip()
        parsed = extract_json_from_text(raw)

        mapping = {}
        items = parsed.get("items") if isinstance(parsed, dict) else parsed

        if isinstance(items, list):
            for item in items:
                if not isinstance(item, dict): continue
                try:
                    idx = int(item.get("i"))
                    ch = str(item.get("char", "")).strip()
                    ch = re.sub(r"[^一-龥]", "", ch)
                    mapping[idx] = ch if ch in candidates_unique else "未知"
                except:
                    continue
        return mapping
    except Exception as e:
        print(f"切片识别错误: {e}")
        return {}


# ================= 对外暴露的接口 =================

def solve_captcha_from_base64(image_base64: str) -> List[Tuple[float, float]]:
    """
    接收 Base64 编码的验证码图片，返回按点击顺序排列的坐标列表。
    坐标以图片左上角为原点 (x, y)，单位为像素。
    """
    # 1. 加载配置
    conf = load_xk_config()
    llm_key = conf.get("LLM_KEY")
    llm_model = conf.get("LLM_MODEL")
    llm_base = conf.get("LLM_BASE_URL")
    yolo_path = conf.get("YOLO_MODEL_PATH", "best.pt")

    # 修正相对路径
    if not os.path.isabs(yolo_path):
        yolo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), yolo_path)

    # 2. 初始化模型和客户端
    yolo = _get_yolo_model(yolo_path)
    client = OpenAI(api_key=llm_key, base_url=llm_base)

    # 3. 图像预处理
    try:
        image_data = base64.b64decode(image_base64)
        full_image = Image.open(io.BytesIO(image_data)).convert("RGB")
    except Exception as e:
        raise ValueError(f"无效的 Base64 图片数据: {e}")

    # 分割图片，获取缩放因子 (用于后续坐标还原)
    top_img_bytes, scale_factor, bottom_b64 = process_captcha_split(full_image)

    # 4. LLM 读题
    target_text = get_prompt_text(client, llm_model, bottom_b64)
    target_chars = [c for c in target_text if "\u4e00" <= c <= "\u9fff"]
    if not target_chars:
        raise RuntimeError("未能识别出题目中的有效汉字")

    print(f"识别目标: {target_text}")

    # 5. YOLO 检测
    dets = yolo_detect_boxes(yolo, top_img_bytes, conf_thres=YOLO_DEFAULT_CONF)
    # 如果检测数量不足，尝试降低阈值
    if len(dets) < len(target_chars):
        dets = yolo_detect_boxes(yolo, top_img_bytes, conf_thres=0.10)

    if len(dets) < len(target_chars):
        raise RuntimeError(f"检测到的文字数量不足 (检测:{len(dets)} < 目标:{len(target_chars)})")

    # 6. 切片与识别
    slices = crop_slices(top_img_bytes, dets)
    mapping = llm_match_slices(client, llm_model, target_text, slices)

    # 7. 匹配逻辑与坐标还原
    # 将切片按识别出的字符归类
    char_map: Dict[str, List[dict]] = {}
    for s in slices:
        identified_char = mapping.get(s["i"], "未知")
        if identified_char != "未知":
            char_map.setdefault(identified_char, []).append(s)

    # 按置信度排序
    for ch in char_map:
        char_map[ch].sort(key=lambda x: -x["conf"])

    result_coords = []

    for ch in target_chars:
        pool = char_map.get(ch, [])
        if not pool:
            raise RuntimeError(f"无法在验证码中找到目标字符: {ch}")

        # 取出置信度最高的切片
        best_slice = pool.pop(0)

        # === 关键步骤：坐标还原 ===
        # best_slice['center_resized'] 是基于 resized 图片 (宽=TOP_MAX_PX) 的坐标
        # 我们需要将其除以 scale_factor 还原回原始图片的像素坐标
        cx_resized, cy_resized = best_slice["center_resized"]

        real_x = cx_resized / scale_factor
        real_y = cy_resized / scale_factor

        result_coords.append((real_x, real_y))

    return result_coords
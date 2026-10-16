# -*- coding: utf-8 -*-
"""
core/comfy_client.py
调用本机 ComfyUI 对证件照做生成式编辑（换装 / 换发型 / 换背景 / 精修）。

已按本机实际环境校准：
  ComfyUI 0.39.2  @ http://127.0.0.1:8166
  UNET : qwen_image_edit_2511_int8_convrot.safetensors   (Qwen-Image-Edit 2511)
  LoRA : Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors  (4 步加速)
  CLIP : qwen_2.5_vl_7b_fp8_scaled.safetensors  (type=qwen_image)
  VAE  : qwen_image_vae.safetensors

支持：
  1. 模型/LoRA/步数自定义（启动时服务端实际存在哪些就列哪些）
  2. websocket 流式生成：实时推送采样过程预览帧（需服务端 --preview-method latent2rgb）
"""

import json
import uuid
import time
import io
import struct
import urllib.request
import urllib.error
import urllib.parse
from typing import Optional, List, Callable

import numpy as np
import cv2

try:
    import websocket  # websocket-client
    _HAS_WS = True
except Exception:
    _HAS_WS = False

# --------------------------------------------------------------------------- #
# 服务地址
# --------------------------------------------------------------------------- #
DEFAULT_PORTS = [8166, 8188, 8000, 8189, 8888]
COMFY_URL = "http://127.0.0.1:8166"

_cached_url: Optional[str] = None


def _probe(base: str, timeout: float = 2.0) -> bool:
    try:
        req = urllib.request.Request(f"{base}/system_stats", method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def find_server(force: bool = False) -> Optional[str]:
    global _cached_url
    if _cached_url and not force and _probe(_cached_url):
        return _cached_url
    for p in DEFAULT_PORTS:
        base = f"http://127.0.0.1:{p}"
        if _probe(base):
            _cached_url = base
            return base
    _cached_url = None
    return None


def set_server(url: str) -> None:
    global COMFY_URL, _cached_url
    COMFY_URL = url.rstrip("/")
    _cached_url = COMFY_URL


def server_url() -> str:
    return _cached_url or COMFY_URL


def is_available() -> bool:
    return find_server() is not None


# --------------------------------------------------------------------------- #
# 模型清单（已对照 object_info 校验）
# --------------------------------------------------------------------------- #
UNET_CANDIDATES = [
    "qwen_image_edit_2511_int8_convrot.safetensors",
    "qwen_image_edit_2509_fp8_e4m3fn.safetensors",
]
LORA_2511 = "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
LORA_2509 = "Qwen-Image-Edit-Lightning-8steps-V1.0-bf16.safetensors"
CLIP_CANDIDATES = [
    "qwen_2.5_vl_7b_fp8_scaled.safetensors",
    "qwen_3_8b_fp8mixed.safetensors",
]
VAE_CANDIDATES = [
    "qwen_image_vae.safetensors",
    "qwen_image_2.1_vae_bf16.safetensors",
]

QWEN_UNET = UNET_CANDIDATES[0]
QWEN_CLIP = CLIP_CANDIDATES[0]
QWEN_VAE = VAE_CANDIDATES[0]

_resolved: Optional[dict] = None


def _object_info(timeout: int = 60) -> dict:
    with urllib.request.urlopen(f"{server_url()}/object_info", timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _choices(info: dict, node: str, field: str) -> List[str]:
    try:
        v = info[node]["input"]["required"][field][0]
        return v if isinstance(v, list) else [v]
    except Exception:
        return []


def resolve_models(force: bool = False) -> dict:
    """挑出真实可用的一组模型 + 采样参数，并带出可选清单供 UI 下拉。"""
    global _resolved
    if _resolved and not force:
        return _resolved

    res = {
        "unet": QWEN_UNET, "lora": LORA_2511, "clip": QWEN_CLIP, "vae": QWEN_VAE,
        "steps": 4, "cfg": 1.0, "shift": 3.0, "ok": False, "note": "",
        "unet_options": [], "lora_options": [],
    }
    if not find_server():
        res["note"] = "未检测到 ComfyUI 服务"
        return res

    try:
        info = _object_info()
    except Exception as e:
        res["note"] = f"读取 object_info 失败: {e}"
        return res

    unets = _choices(info, "UNETLoader", "unet_name")
    clips = _choices(info, "CLIPLoader", "clip_name")
    vaes = _choices(info, "VAELoader", "vae_name")
    loras = _choices(info, "LoraLoaderModelOnly", "lora_name")

    def first(cands, pool):
        for c in cands:
            if c in pool:
                return c
        return None

    unet = first(UNET_CANDIDATES, unets)
    clip = first(CLIP_CANDIDATES, clips)
    vae = first(VAE_CANDIDATES, vaes)
    if not unet or not clip or not vae:
        res["note"] = "没有找到 Qwen-Image-Edit 主模型/CLIP/VAE"
        return res

    if "2511" in unet:
        lora = LORA_2511 if LORA_2511 in loras else \
            (LORA_2511.replace("-bf16", "-fp32") if LORA_2511.replace("-bf16", "-fp32") in loras else None)
        steps = 4
    else:
        lora = LORA_2509 if LORA_2509 in loras else None
        steps = 8

    # 供 UI 下拉的选项：只挑 qwen 相关的
    res.update({
        "unet": unet, "clip": clip, "vae": vae, "lora": lora,
        "steps": steps if lora else 20,
        "cfg": 1.0 if lora else 4.0,
        "ok": True, "note": "OK",
        "unet_options": [u for u in unets if "qwen" in u.lower()],
        "lora_options": [l for l in loras
                         if ("qwen" in l.lower() and ("lightning" in l.lower() or "consis" in l.lower() or "outfit" in l.lower()))],
    })
    _resolved = res
    return res


# --------------------------------------------------------------------------- #
# Prompt 预设
# --------------------------------------------------------------------------- #
_NEG_DEFAULT = "模糊, 低质量, 变形, 多余的手指, 多余肢体, 文字, 水印, 卡通, 插画, 绘画"

PRESET_PROMPTS = {
    "换正装（深色西装）": "把人物身上的衣服换成合体的深色商务西装搭配白衬衫，肩线与身体自然贴合，保持人脸、发型和背景完全不变",
    "换白衬衫": "把人物身上的衣服换成干净的白色衬衫，领口整齐，保持人脸、发型和背景完全不变",
    "换正装（女士）": "把人物身上的衣服换成合体的女士职业西装外套搭配浅色衬衫，保持人脸、发型和背景完全不变",
    "换毛衣": "把人物身上的衣服换成纯色圆领针织毛衣，保持人脸、发型和背景完全不变",
    "换护士服": "把人物身上的衣服换成白色护士服，保持人脸、发型和背景完全不变",
    "换短发": "给人物换成整洁利落的短发发型，发际线自然，保持人脸五官、衣服和背景完全不变",
    "换长发": "给人物换成柔顺的长发发型，发际线自然，保持人脸五官、衣服和背景完全不变",
    "换刘海": "给人物添加自然的齐刘海发型，保持人脸五官、衣服和背景完全不变",
    "换背头": "给人物换成向后梳理的背头发型，保持人脸五官、衣服和背景完全不变",
    "染黑发": "把人物的头发颜色染成自然纯黑色，保持发型形状、人脸五官和背景完全不变",
    "染棕发": "把人物的头发颜色染成自然的栗棕色，保持发型形状、人脸五官和背景完全不变",
    "一键微笑": "让人物自然地微笑，微微露出牙齿，保持五官特征、发型、衣服和背景完全不变",
    "闭眼修复": "让人物双眼自然睁开平视镜头，眼神有神，保持五官特征、发型、衣服和背景完全不变",
    "去除眼镜": "去除人物脸上的眼镜，还原自然的眼睛和眼部周围皮肤，保持五官特征、发型、衣服和背景完全不变",
    "磨皮美颜": "对人物皮肤做专业级柔化磨皮和肤色提亮，保留真实毛孔质感，保持五官特征、发型、衣服和背景完全不变",
    "去油光": "去除人物面部的油光和反光，让皮肤呈现自然的哑光质感，保持五官特征、衣服和背景完全不变",
    "换纯白背景": "把背景替换成纯净的白色影棚背景，边缘干净，保持人物、发型和衣服完全不变",
    "换蓝底背景": "把背景替换成证件照标准的均匀蓝色背景，边缘干净，保持人物、发型和衣服完全不变",
    "换红底背景": "把背景替换成证件照标准的均匀红色背景，边缘干净，保持人物、发型和衣服完全不变",
    "换灰底背景": "把背景替换成均匀的浅灰色背景，边缘干净，保持人物、发型和衣服完全不变",
    "证件照整体生成": "请参考图中人物特征，生成一张标准的证件照：正面免冠、面部居中、眼神平视镜头、表情自然、肩线水平、影棚级柔光、无噪点，保持面部特征一致",
    "职业形象照": "生成一张专业职业形象照：柔和影棚布光、浅灰渐变背景、合体深色西装、正面半身、自然微笑，保持面部特征一致",
}

PRESET_PROMPTS_REF = {
    "按参考图换装": "把图1中人物的衣服换成图2中的服装，款式颜色完全一致，比例协调自然贴合身体，保持图1人物的脸、发型和背景不变",
    "按参考图换发型": "把图1中人物的发型换成图2中的发型，形状颜色一致，发际线自然，保持图1人物的脸、衣服和背景不变",
    "按参考图换背景": "把图1的背景替换成图2的背景，光影协调，保持图1人物、发型和衣服不变",
}


# --------------------------------------------------------------------------- #
# 上传 / 工作流
# --------------------------------------------------------------------------- #
def _upload_image(image_bgr: np.ndarray, prefix: str = "zsphoto_in") -> str:
    png_ok, buf = cv2.imencode(".png", image_bgr)
    if not png_ok:
        raise ValueError("图像编码失败")
    png = buf.tobytes()
    name = f"{prefix}_{uuid.uuid4().hex[:10]}.png"

    boundary = uuid.uuid4().hex
    body = bytearray()

    def w_text(k: str, v: str):
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(f'Content-Disposition: form-data; name="{k}"\r\n\r\n'.encode())
        body.extend(v.encode("utf-8"))
        body.extend(b"\r\n")

    def w_file(k: str, fn: str, ct: str, data: bytes):
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(f'Content-Disposition: form-data; name="{k}"; filename="{fn}"\r\n'.encode())
        body.extend(f"Content-Type: {ct}\r\n\r\n".encode())
        body.extend(data)
        body.extend(b"\r\n")

    w_text("type", "input")
    w_text("overwrite", "true")
    w_text("subfolder", "")
    w_file("image", name, "image/png", png)
    body.extend(f"--{boundary}--\r\n".encode())

    req = urllib.request.Request(
        f"{server_url()}/upload/image", data=bytes(body),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            resp = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise ConnectionError(f"上传图像失败 HTTP {e.code}: {e.read()[:300]!r}")
    n = resp.get("name", name)
    sub = resp.get("subfolder") or ""
    return f"{sub}/{n}" if sub else n


def _build_workflow(models: dict, image_names: List[str], prompt: str,
                    negative: str, seed: int, steps: int, cfg: float,
                    shift: float, megapixels: float) -> dict:
    clips = [n for n in image_names if n]
    wf = {
        "1": {"class_type": "UNETLoader",
              "inputs": {"unet_name": models["unet"], "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": models["clip"], "type": "qwen_image", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": models["vae"]}},
    }
    model_src = ["1", 0]
    if models.get("lora"):
        wf["4"] = {"class_type": "LoraLoaderModelOnly",
                   "inputs": {"model": model_src, "lora_name": models["lora"], "strength_model": 1.0}}
        model_src = ["4", 0]
    wf["6"] = {"class_type": "ModelSamplingAuraFlow",
               "inputs": {"model": model_src, "shift": float(shift)}}

    for i, nm in enumerate(clips):
        nid = str(10 + i * 2)
        wf[nid] = {"class_type": "LoadImage", "inputs": {"image": nm}}
        wf[str(int(nid) + 1)] = {
            "class_type": "ImageScaleToTotalPixels",
            "inputs": {"image": [nid, 0], "upscale_method": "lanczos",
                       "megapixels": float(megapixels), "resolution_steps": 1},
        }

    te_in = {"clip": ["2", 0], "vae": ["3", 0], "prompt": prompt}
    for i in range(len(clips)):
        te_in[f"image{i + 1}"] = [str(11 + i * 2), 0]
    wf["30"] = {"class_type": "TextEncodeQwenImageEditPlus", "inputs": dict(te_in)}
    wf["31"] = {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["30", 0]}}
    wf["32"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["11", 0], "vae": ["3", 0]}}
    wf["40"] = {"class_type": "KSampler",
                "inputs": {"model": ["6", 0], "positive": ["30", 0], "negative": ["31", 0],
                           "latent_image": ["32", 0], "seed": int(seed), "steps": int(steps),
                           "cfg": float(cfg), "sampler_name": "euler",
                           "scheduler": "simple", "denoise": 1.0}}
    wf["50"] = {"class_type": "VAEDecode", "inputs": {"samples": ["40", 0], "vae": ["3", 0]}}
    wf["60"] = {"class_type": "SaveImage", "inputs": {"images": ["50", 0], "filename_prefix": "zsphoto"}}
    return wf


def _submit(wf: dict) -> str:
    payload = {"prompt": wf, "client_id": str(uuid.uuid4())}
    req = urllib.request.Request(
        f"{server_url()}/prompt", data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read().decode("utf-8"))["prompt_id"]
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "ignore")
        try:
            j = json.loads(raw)
            msg = j.get("error", {})
            detail = msg.get("message") if isinstance(msg, dict) else str(msg)
            nodes = j.get("node_errors", {})
            if nodes:
                detail = f"{detail} | 节点错误: {json.dumps(nodes, ensure_ascii=False)[:600]}"
        except Exception:
            detail = raw[:600]
        raise ConnectionError(f"提交失败 HTTP {e.code}: {detail}")


def _fetch_result_image(prompt_id: str) -> np.ndarray:
    with urllib.request.urlopen(f"{server_url()}/history/{prompt_id}", timeout=15) as r:
        hist = json.loads(r.read().decode("utf-8"))
    imgs = []
    for _nid, out in hist.get(prompt_id, {}).get("outputs", {}).items():
        imgs.extend(out.get("images", []))
    if not imgs:
        return None
    info = imgs[0]
    q = urllib.parse.urlencode({"filename": info.get("filename", ""),
                                "subfolder": info.get("subfolder", ""),
                                "type": info.get("type", "output")})
    with urllib.request.urlopen(f"{server_url()}/view?{q}", timeout=60) as r:
        data = r.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("结果图像解码失败")
    return img


def _decode_ws_preview(data: bytes) -> Optional[np.ndarray]:
    """解析 ComfyUI websocket 二进制预览帧。"""
    if len(data) < 8:
        return None
    event = struct.unpack(">I", data[:4])[0]
    payload = data[4:]
    try:
        if event == 1:            # PREVIEW_IMAGE: [4B img_type][jpeg/png]
            if len(payload) <= 4:
                return None
            img = cv2.imdecode(np.frombuffer(payload[4:], np.uint8), cv2.IMREAD_COLOR)
            return img
        if event == 4:            # PREVIEW_IMAGE_WITH_METADATA
            mlen = struct.unpack(">I", payload[:4])[0]
            rest = payload[4 + mlen:]
            if len(rest) <= 4:
                return None
            return cv2.imdecode(np.frombuffer(rest[4:], np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None
    return None


# --------------------------------------------------------------------------- #
# 流式生成（实时预览）
# --------------------------------------------------------------------------- #
def edit_image_stream(image_bgr: np.ndarray, prompt: str,
                      negative: str = "", steps: Optional[int] = None,
                      seed: Optional[int] = None, timeout: int = 1800,
                      ref_image_bgr: Optional[np.ndarray] = None,
                      unet: Optional[str] = None, lora: Optional[str] = None,
                      cfg: Optional[float] = None, shift: Optional[float] = None,
                      megapixels: float = 1.0):
    """
    生成式编辑（生成器版本）。
    依次 yield:
        ("progress", 文本)
        ("preview",  BGR ndarray)   -- 采样过程预览帧
        ("done",     BGR ndarray)   -- 最终结果（此后生成器结束）
    """
    models = dict(resolve_models())
    if not models.get("ok"):
        raise ConnectionError(f"ComfyUI 不可用或模型缺失：{models.get('note')}")
    if not prompt or not prompt.strip():
        raise ValueError("prompt 不能为空")

    # UI 传入的覆盖项
    if unet: models["unet"] = unet
    if lora: models["lora"] = lora
    if steps: models["steps"] = int(steps)
    if cfg is not None: models["cfg"] = float(cfg)
    if shift is not None: models["shift"] = float(shift)
    if models["lora"] in ("", None, "无（不用加速LoRA）"):
        models["lora"] = None
        models["steps"] = int(steps) if steps else 20
        models["cfg"] = float(cfg) if cfg is not None else 4.0

    st = int(models["steps"])
    sd = int(seed) if seed is not None else int(time.time() * 1000) % (2 ** 32)

    yield ("progress", "上传图像…")
    names = [_upload_image(image_bgr, "zsphoto_a")]
    if ref_image_bgr is not None:
        names.append(_upload_image(ref_image_bgr, "zsphoto_b"))

    wf = _build_workflow(models, names, prompt, negative or _NEG_DEFAULT,
                         sd, st, models["cfg"], models["shift"], megapixels)
    yield ("progress", f"提交任务（{st} 步，模型 {models['unet']}）…")
    pid = _submit(wf)

    ws_url = f"ws://{urllib.parse.urlparse(server_url()).netloc}/ws?clientId={uuid.uuid4().hex}"
    got_done = False
    got_frame = False

    if _HAS_WS:
        try:
            ws = websocket.create_connection(ws_url, timeout=4, enable_multithread=True)
            t0 = time.time()
            last_hist_check = 0.0
            while time.time() - t0 < timeout:
                try:
                    op, frame = ws.recv_data()
                except websocket.WebSocketTimeoutException:
                    frame = None
                    op = None
                if op == 1:  # 文本帧
                    try:
                        msg = json.loads(frame.decode("utf-8", "ignore"))
                    except Exception:
                        msg = None
                    if msg:
                        mtype = msg.get("type")
                        data = msg.get("data", {}) or {}
                        if mtype == "progress":
                            yield ("progress", f"采样中 {data.get('value', 0)}/{data.get('max', st)} 步")
                        elif mtype == "executing" and data.get("node") is None:
                            break  # 整个队列执行完毕
                        elif mtype == "execution_error":
                            raise RuntimeError(json.dumps(data, ensure_ascii=False)[:600])
                elif op == 2:  # 二进制帧
                    img = _decode_ws_preview(frame)
                    if img is not None:
                        got_frame = True
                        yield ("preview", img)

                # 定期轮询 history 兜底（防止漏掉结束信号；结果一到立刻返回）
                now = time.time()
                if now - last_hist_check > 4:
                    last_hist_check = now
                    try:
                        img = _fetch_result_image(pid)
                    except Exception:
                        img = None
                    if img is not None:
                        yield ("progress", "完成")
                        yield ("done", img)
                        got_done = True
                        break
            try:
                ws.close()
            except Exception:
                pass
        except Exception:
            pass  # ws 不可用则走轮询

    if not got_done:
        # 轮询拿最终结果
        t0 = time.time()
        while time.time() - t0 < timeout:
            time.sleep(0.8)
            try:
                img = _fetch_result_image(pid)
            except Exception:
                continue
            if img is not None:
                yield ("progress", "完成")
                yield ("done", img)
                got_done = True
                break
    if not got_done:
        raise TimeoutError(f"等待 ComfyUI 结果超时（{timeout}s）")


def edit_image(image_bgr: np.ndarray, prompt: str,
               negative: str = "", steps: Optional[int] = None,
               seed: Optional[int] = None, timeout: int = 1800,
               ref_image_bgr: Optional[np.ndarray] = None,
               unet: Optional[str] = None, lora: Optional[str] = None,
               cfg: Optional[float] = None, shift: Optional[float] = None,
               megapixels: float = 1.0,
               progress: Optional[Callable[[str], None]] = None) -> np.ndarray:
    """阻塞版本：内部走 edit_image_stream，取最终结果。"""
    result = None
    for kind, payload in edit_image_stream(
            image_bgr, prompt, negative=negative, steps=steps, seed=seed,
            timeout=timeout, ref_image_bgr=ref_image_bgr, unet=unet, lora=lora,
            cfg=cfg, shift=shift, megapixels=megapixels):
        if kind == "progress" and progress:
            progress(payload)
        elif kind == "done":
            result = payload
    if result is None:
        raise RuntimeError("ComfyUI 未返回结果")
    return result


def free_memory() -> None:
    try:
        req = urllib.request.Request(f"{server_url()}/free",
                                     data=json.dumps({"unload_models": True, "free_memory": True}).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        urllib.request.urlopen(req, timeout=10).read()
    except Exception:
        pass


def interrupt() -> None:
    try:
        req = urllib.request.Request(f"{server_url()}/interrupt", data=b"{}",
                                     headers={"Content-Type": "application/json"}, method="POST")
        urllib.request.urlopen(req, timeout=5).read()
    except Exception:
        pass


def server_models_report() -> str:
    m = resolve_models(force=True)
    if not m["ok"]:
        return f"未连接 ComfyUI（{m['note']}）"
    return (f"{server_url()} | {m['unet']} | {m['lora'] or '无加速LoRA'} | "
            f"{m['steps']}步 cfg{m['cfg']}")

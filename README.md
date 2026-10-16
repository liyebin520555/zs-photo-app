# ZSPhotoApp · 证件照 / 人像美颜精修工具

一个本地运行的桌面端照片处理工具：抠图换底、证件照规格与多图排版、皮肤级美颜、面部微整形，并可接入本地 ComfyUI 做生成式 AI 编辑（换装/改发色等）。

界面基于 PySide6，全部处理在本地完成，不依赖任何云服务。

## 功能

| 模块 | 说明 |
| --- | --- |
| 抠图换底 | rembg 提取人像，支持纯色底（红/蓝/白等）与自定义底色 |
| 证件照规格 | 一寸/二寸/小一寸/美国签证等常用规格，含自定义像素与 DPI |
| 排版打印 | 单张成品 + 6 寸/A4 多张拼版，直接送冲印 |
| 磨皮美白锐化 | 基于 YCrCb 肤色掩码，只作用于脸部与颈部皮肤，色度不偏移 |
| 面部精修 | mediapipe 478 关键点局部形变：瘦脸、大眼、微笑、牙齿美白、自动摆正头部、头部左右转/上下仰、眼球左右/上下 |
| AI 编辑 | 接入本地 ComfyUI（Qwen-Image-Edit 2511 + Lightning LoRA），支持提示词换装、双图参考换装，采样过程实时预览 |

## 目录结构

```
ZSPhotoApp/
├─ main.py                 程序入口（PySide6 主窗口）
├─ ui/
│  ├─ photo_tab.py         人像主界面：预览、抠图、美颜、AI 编辑、生成结果
│  └─ scan_tab.py          文档扫描页（当前入口未挂载）
├─ core/
│  ├─ matting.py           rembg 抠图与底色合成
│  ├─ beautify.py          磨皮/美白/锐化（肤色遮罩）
│  ├─ retouch.py           478 关键点面部形变（瘦脸/大眼/微笑/牙齿/发色）
│  ├─ photo_specs.py       证件照规格与拼版合成
│  ├─ layout.py            画布排版工具
│  ├─ garment.py           服装模板相关工具
│  ├─ comfy_client.py      ComfyUI 客户端（HTTP + WebSocket，流式预览帧）
│  ├─ scanner.py           文档扫描处理
│  └─ spec_data.py         规格数据表
├─ models/
│  └─ face_landmarker.task mediapipe 人脸关键点模型（478 点，多人脸）
├─ prompts/                各模块的提示词与规格文本
└─ 一键启动（含ComfyUI）.bat / 启动证件照工具.bat    Windows 双击启动
```

## 环境要求

- Python 3.10+
- 依赖：`pip install -r requirements.txt`（PySide6 / opencv-python / rembg / onnxruntime / pillow / numpy）
- 面部精修需要 `mediapipe`：`pip install mediapipe`
- 首次运行 rembg 会自动下载抠图模型（约 170MB）

## 运行

```bash
python main.py
```

Windows 也可直接双击 `启动证件照工具.bat`。

## 接入 ComfyUI（AI 编辑 / 换装）

1. 启动本地 ComfyUI，`core/comfy_client.py` 会依次探测 8166、8188 等端口，也可手动指定。
   启动参数建议加上采样实时预览：`--preview-method latent2rgb`。
2. 需要的模型（放 ComfyUI 对应目录）：
   - UNET：`qwen_image_edit_2511_int8_convrot.safetensors`
   - LoRA：`Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors`
   - CLIP：`qwen_2.5_vl_7b_fp8_scaled.safetensors`（type 选 `qwen_image`）
   - VAE：`qwen_image_vae.safetensors`
3. 界面 AI 分组里可切换主模型与 LoRA（切换会自动匹配 4 / 8 / 20 步），采样过程在预览窗实时显示。

配套的 ComfyUI 自定义节点（综合美颜精修，含汉化）见 `ComfyUI-FaceBeauty`。

## 说明

- 所有路径均可用相对路径运行，模型与输出目录会按需自动创建。
- 处理结果保存在 `outputs/`（已在 .gitignore 中忽略）。

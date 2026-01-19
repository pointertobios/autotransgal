# autotransgal

一个跑在 Linux 桌面的 galgame 日文→中文叠加翻译器：检测屏幕文本区域 → OCR → 合并排序 → 调用 OpenAI Python SDK（或 openai-compatible 服务）翻译 → 用 Tkinter 叠加窗口显示结果。

本项目目标是“好用且可控”：配置优先、支持 ROI（只识别你选的区域）、支持用户点击确认内容是否属于游戏文本，用上下文提升翻译稳定性。

## 功能特性

- 叠加显示：每个 OCR 区域一对窗口（识别/翻译），文字统一用 Pillow 渲染（避免 Tk 原生文本渲染的坑）。
- ROI 选择：首次启动会让你框选“识别区域”，并自动写回到 config.toml 的 [ocr].roi_abs。
- 智能合并：对检测框进行排序/合并，尽量把多段对话合成更连贯的输入。
- AI 翻译：仅使用 OpenAI Python SDK 进行请求（不手写 HTTP），支持配置 base_url 适配 openai-compatible（如 deepseek）。
- 点击确认（可选）：点击叠加文本可标记“这是游戏内容”，再次点击可取消，避免误点污染上下文；确认状态会立刻变绿。
- 全局触发：通过全局键盘/鼠标事件触发流程；点击叠加窗口本身不会触发新流程（避免冲突）。

## 工作流程

1) 抓屏（可选不同 backend）
2) rapidocr-onnxruntime 做文本区域检测
3) manga-ocr 对每个区域 OCR
4) 合并/排序得到更稳定的文本片段
5) OpenAI SDK 调用模型翻译（按系统提示词的逐行 JSON 协议）
6) Tkinter + Pillow 贴图叠加显示

## 环境要求

- Linux
- Python >= 3.13
- 推荐使用 uv 管理依赖与运行

## 安装

```bash
uv sync
```

## 快速开始

1. 首次运行（生成配置文件）

```bash
uv run cli
```

如果当前目录没有 config.toml，会自动生成一份模板并退出，按提示填写后再运行。

1. 编辑 config.toml，至少填好：

- [openai].api_key

1. 再次运行

```bash
uv run cli
```

启动后会提示你框选 ROI（若 config.toml 里已有有效的 [ocr].roi_abs 则会直接复用）。

## 配置说明（config.toml）

说明：配置文件默认路径是项目目录下的 config.toml；也可以用 --config 指定。

### [openai]

- api_key：必填
- base_url：可选；默认 `https://api.openai.com/v1`。使用第三方 openai-compatible 服务时设置为类似 `https://example.com/v1`
- model：可选；默认 gpt-4.1-mini
- headers：可选；当服务需要额外请求头时使用

示例：

```toml
[openai]
api_key = "sk-..."
base_url = "https://api.openai.com/v1"
model = "gpt-4.1-mini"

# 仅当你的服务明确要求额外 header 时才配置
#[openai.headers]
# "X-My-Header" = "value"
```

### [capture]

- monitor：mss 的 monitor 索引（通常 1 是主屏）
- backend：抓屏后端
  - auto：自动选择
  - spectacle：调用外部命令 spectacle 截图（性能较差，但在 Linux 环境中最稳定）
  - mss：走 mss（在部分 Wayland 环境可能失败）
  - qt：走 Qt 截图（兼容性依环境而定）

### [ocr]

- det_min_box_area：过滤太小的检测框（像素面积）
- max_regions：每帧最多处理多少个区域
- merge_gap_in_line_ratio / merge_gap_between_lines_ratio：合并阈值（基于检测框高度中位数）
- roi_abs：识别区域的屏幕绝对坐标 [x1, y1, x2, y2]；程序会自动写回

### [overlay]

- font_name：字体名（优先用 fontconfig/fc-match 解析），如 "Noto Sans CJK SC" / "Sarasa UI SC"
- font_size：字号
- capture_hide_delay_ms：开始抓屏前，会先隐藏所有叠加窗口；等待该时间后再截图（避免把叠加层截进去）
- max_width_ratio：文本框最大宽度占屏幕宽度比例
- margin_bottom：贴近底部显示时的下边距

### [translate]

- min_interval_sec：AI 翻译节流（减少短时间重复请求）

## CLI 用法

```bash
uv run cli --help
uv run cli --config /path/to/config.toml
uv run cli --doctor
uv run cli --version
```

## 热键

- Ctrl+Alt+T：启用/暂停（暂停会清空叠加层）
- Ctrl+Alt+Q：退出

## 诊断与排错

### 1) API/WAF/Cloudflare 拦截

```bash
uv run cli --doctor
```

诊断会通过 OpenAI Python SDK 尝试列出 models，并输出可复制的结论文本。
如果服务返回 HTML（常见于 Cloudflare/WAF：Please enable cookies / Sorry you have been blocked），说明这不是可供 SDK 调用的 API 域名，需要服务方提供“不需要 Cookie/JS 的 API 入口”，或放行 /v1/*。

### 2) Wayland 下抓屏失败

现象：mss 报错、返回黑屏/空图。

建议：把 config.toml 的 [capture].backend 设为 spectacle（前提是系统有该命令）。它通常更兼容，但每帧会启动外部进程，性能会差一些。

### 3) 字体显示异常/乱码

优先设置 [overlay].font_name 为系统已安装的 CJK 字体，并确保有 fontconfig（fc-match 可用）。

## 已知限制

- 本项目目前主要面向 Linux；Wayland 下叠加窗口行为（置顶/穿透/抓屏）受桌面环境影响较大。
- 目前 [capture].fps 配置项暂未接入实际节流（预留项），实际触发由全局输入事件驱动。

## 开发

- 入口：autotransgal/cli.py
- 叠加层：autotransgal/ui/overlay.py
- 翻译：autotransgal/pipeline/translate.py（OpenAI SDK）
- 诊断：autotransgal/diagnostics.py

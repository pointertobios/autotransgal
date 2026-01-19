from __future__ import annotations

import threading
import time
import traceback
from dataclasses import dataclass
from queue import Empty, Queue
from typing import Callable

import tkinter as tk
import subprocess
import shutil

from PIL import Image, ImageDraw, ImageFont, ImageTk

from autotransgal.config import AppConfig, persist_roi_abs
from autotransgal.pipeline.capture import ScreenCapturer
from autotransgal.pipeline.merge import merge_and_sort
from autotransgal.pipeline.orchestrator import (
    PipelineCore,
    PipelineOutput,
    RegionOutput,
)
from autotransgal.pipeline.translate import make_request_id
from autotransgal.util.geometry import BBox
from autotransgal.input_events import InputActivity


def _resolve_font_via_fontconfig(font_name: str) -> tuple[str, int] | None:
    """用 fontconfig (fc-match) 按字体名称解析到 (file, index)。

    - fontconfig 会遵循 XDG 相关目录与系统字体配置
    - 对于 .ttc/.otc 字体集，fc-match 会给出正确的 face index
    """

    if not font_name:
        return None

    exe = shutil.which("fc-match")
    if not exe:
        return None

    # file: 字体文件路径
    # index: TTC/OTC 的 face index（TTF 通常是 0）
    fmt = "%{file}|%{index}\n"
    proc = subprocess.run(
        [exe, "-f", fmt, font_name],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return None

    out = (proc.stdout or "").strip()
    if not out or "|" not in out:
        return None

    file_path, index_s = out.split("|", 1)
    file_path = file_path.strip()
    index_s = index_s.strip()
    if not file_path:
        return None

    try:
        index = int(index_s) if index_s else 0
    except ValueError:
        index = 0
    return file_path, index


def _try_load_font(
    *,
    font_name: str,
    size: int,
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    resolved = _resolve_font_via_fontconfig(font_name)
    if resolved is not None:
        file_path, index = resolved
        try:
            return ImageFont.truetype(file_path, size=size, index=index)
        except OSError:
            pass

    # 回退：常见中日韩字体文件（不依赖 font_name）
    candidates = [
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansJP-Regular.otf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for p in candidates:
        try:
            return ImageFont.truetype(p, size=size)
        except OSError:
            continue

    return ImageFont.load_default()


class OverlayWindow:
    """Tkinter 叠加层：
    - 不用 Tk 的文本渲染（已知有问题），而是 Pillow 渲染文字后贴到窗口上。
    - 只使用“按区域创建多个小窗口”的策略：每个 OCR 区域一个小 Toplevel。
    - 每个 OCR 区域旁边显示“识别/翻译”，并在 warning 时用红色提示。
    """

    def __init__(self, cfg: AppConfig):
        self._cfg = cfg
        self._running = True
        self._enabled = True

        self._root = tk.Tk()
        self._root.title("autotransgal")
        # 主 root 只作为事件循环宿主，不显示任何窗口
        self._root.withdraw()

        self._screen_w = int(self._root.winfo_screenwidth())
        self._screen_h = int(self._root.winfo_screenheight())

        self._font = _try_load_font(
            font_name=str(getattr(cfg.overlay, "font_name", "") or "").strip(),
            size=int(cfg.overlay.font_size),
        )
        # 识别/翻译分别用独立窗口（每个 OCR 区域各一对）
        self._region_windows: list[_RegionWindow] = []
        self._tr_windows: list[_RegionWindow] = []

        # 用于与全局鼠标监听协作：缓存当前识别/翻译窗口的屏幕矩形，避免在 listener 线程里调用 Tk。
        # 形如 (x1, y1, x2, y2)
        self._content_window_rects: list[tuple[int, int, int, int]] = []

        # 点击确认功能：用户点击窗口表示“这是游戏内容”
        self._confirm_enabled = False
        self._confirmed_ids: set[str] = set()
        self._confirmed_since_last_flow: set[str] = set()

        # 画框：合并前 / 合并后
        self._pre_box_windows: list[_BoxBorder] = []
        self._merged_box_windows: list[_BoxBorder] = []

        self._capturer = ScreenCapturer(
            monitor=cfg.capture.monitor,
            backend=cfg.capture.backend,
        )
        self._core = PipelineCore(cfg)

        # 启动时让用户选择一次识别区域（屏幕绝对坐标）
        self._roi_abs: BBox | None = None

        self._frame_q: Queue = Queue(maxsize=1)
        # 两阶段更新：识别 -> 翻译。队列稍大一点避免识别结果被覆盖。
        self._result_q: Queue = Queue(maxsize=2)
        self._worker = threading.Thread(target=self._loop_worker, daemon=True)

        # 事件触发翻译：翻译中不再触发新的流程，且丢弃所有事件避免堆积
        self._state_lock = threading.Lock()
        self._processing = False
        self._trigger_pending = False

        # 翻译线程：识别完成后允许启动新流程；新流程开始前会取消上一轮翻译
        self._translate_lock = threading.Lock()
        self._translate_cancel: threading.Event | None = None
        self._translate_thread: threading.Thread | None = None
        self._translate_inflight_key: tuple[str, ...] | None = None
        self._translate_last_done_key: tuple[str, ...] | None = None
        self._translate_last_done_map: (
            dict[str, tuple[str | None, str, str | None]] | None
        ) = None
        self._latest_recognized_key: tuple[str, ...] | None = None
        self._translate_waiter: _TranslateWaiter | None = None

        # 先隐藏 overlay 再截图，给窗口管理器/合成器一个刷新时间
        self._capture_hide_delay_ms = int(
            getattr(cfg.overlay, "capture_hide_delay_ms", 35) or 35
        )
        # 合并短时间内的多次输入事件
        self._trigger_debounce_ms = 60

        # 流程日志：用于将“开始/识别完成/翻译完成”三类日志串起来
        self._flow_seq = 0
        self._flow_id = 0
        self._flow_t0 = 0.0

    def _log(self, msg: str) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts}] {msg}", flush=True)

    def start(self) -> None:
        # 启动时优先使用配置里缓存的 ROI；无效/缺失才弹出 ROI 选择
        cached = getattr(self._cfg.ocr, "roi_abs", None)
        if cached is not None:
            try:
                roi = BBox(
                    int(cached[0]),
                    int(cached[1]),
                    int(cached[2]),
                    int(cached[3]),
                )
            except (TypeError, ValueError, IndexError):
                roi = None
            if roi is not None and self._roi_looks_valid(roi):
                self._roi_abs = roi
                self._log(
                    "使用缓存 ROI："
                    f"({roi.x1},{roi.y1})-({roi.x2},{roi.y2})"
                )
            else:
                self._log("缓存 ROI 无效，需重新选择")

        if self._roi_abs is None:
            # 先选取 ROI，再启动 worker：避免识别无用区域
            try:
                self._roi_abs = self._select_roi_once()
                persist_roi_abs(
                    config_path=self._cfg.config_path,
                    roi_abs=(
                        int(self._roi_abs.x1),
                        int(self._roi_abs.y1),
                        int(self._roi_abs.x2),
                        int(self._roi_abs.y2),
                    ),
                )
            except (RuntimeError, OSError, ValueError):
                traceback.print_exc()
                self.quit()
                return

        # 启动时询问是否启用“点击确认游戏内容”功能（文字用 Pillow 渲染）
        try:
            self._confirm_enabled = self._ask_enable_confirm()
        except (RuntimeError, OSError, ValueError, tk.TclError):
            self._confirm_enabled = False

        self._worker.start()
        self._root.after(0, self._render_once)

    def _roi_looks_valid(self, roi_abs: BBox) -> bool:
        """尽量保守地校验缓存 ROI 是否可用。

        规则：
        - ROI 必须有正面积，且不至于过小
        - ROI 必须与当前抓屏后端得到的画面有交集（避免 monitor/分辨率变化导致空 ROI）
        """

        if roi_abs.x2 <= roi_abs.x1 or roi_abs.y2 <= roi_abs.y1:
            return False
        if (roi_abs.x2 - roi_abs.x1) < 10 or (roi_abs.y2 - roi_abs.y1) < 10:
            return False

        # 抓一帧用于验证（开销很小，且仅启动时一次）
        try:
            self._hide_all_regions()
            time.sleep(max(0.0, float(self._capture_hide_delay_ms) / 1000.0))
            frame = self._capturer.grab()
        except (OSError, ValueError, RuntimeError):
            return True

        fx1 = int(frame.left)
        fy1 = int(frame.top)
        fx2 = int(frame.left) + int(frame.width)
        fy2 = int(frame.top) + int(frame.height)

        ix1 = max(int(roi_abs.x1), fx1)
        iy1 = max(int(roi_abs.y1), fy1)
        ix2 = min(int(roi_abs.x2), fx2)
        iy2 = min(int(roi_abs.y2), fy2)
        return ix2 > ix1 and iy2 > iy1

    def _select_roi_once(self) -> BBox:
        """启动时弹出全屏截图，让用户拖拽画框选取 ROI。"""

        hint = "拖拽鼠标画框选择识别区域；回车确认；ESC 取消"
        self._log(f"请选择需要识别的区域：{hint}")

        # 启动阶段还没有 overlay 窗口，但这里仍保持一致：确保干净截图
        self._hide_all_regions()
        time.sleep(max(0.0, float(self._capture_hide_delay_ms) / 1000.0))

        frame = self._capturer.grab()

        # ROI 选择界面的提示文字也用 Pillow 渲染：直接画进截图中
        img = frame.image.copy()
        try:
            # 用一个稍小的字体显示提示，避免遮挡太多内容
            font_name = str(
                getattr(self._cfg.overlay, "font_name", "") or ""
            ).strip()
            banner_font = _try_load_font(
                font_name=font_name,
                size=max(14, int(self._cfg.overlay.font_size) - 4),
            )
            draw = ImageDraw.Draw(img)
            bbox = draw.textbbox((0, 0), hint, font=banner_font)
            text_h = int(bbox[3] - bbox[1])
            pad_x = 12
            pad_y = 8
            bar_h = text_h + pad_y * 2
            draw.rectangle(
                (0, 0, img.width, bar_h),
                fill=(0, 0, 0),
            )
            draw.text(
                (pad_x, pad_y),
                hint,
                font=banner_font,
                fill=(255, 255, 255),
            )
        except (OSError, ValueError, RuntimeError):
            # 提示条绘制失败不应阻塞 ROI 选择
            pass
        selector = _RoiSelector(
            root=self._root,
            image=img,
        )
        bbox = selector.run()
        if bbox is None:
            raise RuntimeError("ROI 选择已取消")

        # bbox 是相对于截图的坐标，转为屏幕绝对坐标
        roi_abs = BBox(
            int(bbox.x1) + int(frame.left),
            int(bbox.y1) + int(frame.top),
            int(bbox.x2) + int(frame.left),
            int(bbox.y2) + int(frame.top),
        )
        self._log(
            f"已选择识别区域：({roi_abs.x1},{roi_abs.y1})-({roi_abs.x2},{roi_abs.y2})"
        )
        return roi_abs

    def run(self) -> None:
        self._root.mainloop()

    def stop(self) -> None:
        self._running = False
        with self._translate_lock:
            if self._translate_cancel is not None:
                self._translate_cancel.set()

    def quit(self) -> None:
        self._running = False
        with self._translate_lock:
            if self._translate_cancel is not None:
                self._translate_cancel.set()
        try:
            self._root.after(0, self._root.quit)
        except tk.TclError:
            pass

    def toggle_enabled(self) -> None:
        # Hotkey 回调可能在非 UI 线程：用 after 投递到 Tk 线程
        try:
            self._root.after(0, self._toggle_enabled_ui)
        except tk.TclError:
            self._toggle_enabled_ui()

    def _toggle_enabled_ui(self) -> None:
        self._enabled = not self._enabled
        if not self._enabled:
            self._hide_all_regions()
            # 关闭时也视作“终止当前流程”
            with self._state_lock:
                self._trigger_pending = False
                self._processing = False
            # 清空可能遗留的帧/结果
            try:
                while True:
                    self._frame_q.get_nowait()
            except Empty:
                pass
            try:
                while True:
                    self._result_q.get_nowait()
            except Empty:
                pass

    def _hide_all_regions(self) -> None:
        for w in self._region_windows:
            w.hide()
        for w in self._tr_windows:
            w.hide()
        for b in self._pre_box_windows:
            b.hide()
        for b in self._merged_box_windows:
            b.hide()

        # 同步清空点击命中缓存
        self._content_window_rects = []

    def notify_activity(self, activity: InputActivity | None = None) -> None:
        """被全局键鼠监听调用：有输入事件时触发一次翻译流程。

        - 翻译流程运行中：丢弃事件
        - 触发已挂起：丢弃事件
        """

        if not self._running or not self._enabled:
            return

        # 先判断是否点击在识别/翻译窗口上：是则仅作为“确认”，不触发新流程。
        # （确认本身由 Tk 窗口的 <Button-1> 绑定处理；这里的目的只是避免全局监听触发流程。）
        if activity is not None and activity.kind == "mouse_click":
            x = activity.x
            y = activity.y
            if (
                x is not None
                and y is not None
                and self._point_hits_content_window(x, y)
            ):
                return

        with self._state_lock:
            if self._processing or self._trigger_pending:
                return
            self._trigger_pending = True

        # 投递到 Tk 线程执行（listener 可能在其它线程）
        try:
            self._root.after(self._trigger_debounce_ms, self._begin_capture)
        except tk.TclError:
            pass

    def _point_hits_content_window(self, x: int, y: int) -> bool:
        px = int(x)
        py = int(y)
        for x1, y1, x2, y2 in list(self._content_window_rects):
            if x1 <= px < x2 and y1 <= py < y2:
                return True
        return False

    def _ask_enable_confirm(self) -> bool:
        text = (
            "是否启用：点击识别/翻译窗口以确认\n"
            "该内容确实是游戏本身的文本？\n\n"
            "启用后：每次开始新流程前，会把上次点击确认的内容"
            "（context=true）作为上下文提供给大模型。"
        )

        top = tk.Toplevel(self._root)
        top.title("autotransgal")
        top.resizable(False, False)
        try:
            top.wm_attributes("-topmost", True)
        except tk.TclError:
            pass
        try:
            top.wm_attributes("-toolwindow", True)
        except tk.TclError:
            pass

        # 文本用 Pillow 渲染
        img = _render_text_box(
            text,
            font=self._font,
            warning=False,
            confirmed=False,
        )
        photo = ImageTk.PhotoImage(img)
        label = tk.Label(top, image=photo, bd=0, highlightthickness=0)
        label.pack(padx=12, pady=12)

        result = tk.BooleanVar(value=False)

        btn_row = tk.Frame(top)
        btn_row.pack(padx=12, pady=(0, 12), fill="x")

        def choose(v: bool) -> None:
            result.set(bool(v))
            try:
                top.destroy()
            except tk.TclError:
                pass

        # 按钮文字也用 Pillow 渲染（Label + PhotoImage）
        enable_imgs = {
            "normal": ImageTk.PhotoImage(
                _render_button_box(
                    "启用",
                    font=self._font,
                    bg=(245, 245, 245),
                    fg=(0, 0, 0),
                )
            ),
            "hover": ImageTk.PhotoImage(
                _render_button_box(
                    "启用",
                    font=self._font,
                    bg=(225, 255, 225),
                    fg=(0, 0, 0),
                )
            ),
        }
        disable_imgs = {
            "normal": ImageTk.PhotoImage(
                _render_button_box(
                    "不启用",
                    font=self._font,
                    bg=(245, 245, 245),
                    fg=(0, 0, 0),
                )
            ),
            "hover": ImageTk.PhotoImage(
                _render_button_box(
                    "不启用",
                    font=self._font,
                    bg=(235, 235, 235),
                    fg=(0, 0, 0),
                )
            ),
        }

        btn_enable = tk.Label(
            btn_row,
            image=enable_imgs["normal"],
            bd=0,
            highlightthickness=0,
            cursor="hand2",
        )
        btn_disable = tk.Label(
            btn_row,
            image=disable_imgs["normal"],
            bd=0,
            highlightthickness=0,
            cursor="hand2",
        )

        def _bind_pillow_button(
            widget: tk.Label,
            imgs: dict[str, ImageTk.PhotoImage],
            on_click: Callable[[], None],
        ) -> None:
            widget.bind(
                "<Enter>",
                lambda _e: widget.configure(image=imgs["hover"]),
            )
            widget.bind(
                "<Leave>",
                lambda _e: widget.configure(image=imgs["normal"]),
            )
            widget.bind("<Button-1>", lambda _e: on_click())

        _bind_pillow_button(btn_enable, enable_imgs, lambda: choose(True))
        _bind_pillow_button(btn_disable, disable_imgs, lambda: choose(False))

        btn_enable.pack(side="left", expand=True)
        btn_disable.pack(side="right", expand=True)

        # 居中显示
        top.update_idletasks()
        w = top.winfo_width()
        h = top.winfo_height()
        sw = self._root.winfo_screenwidth()
        sh = self._root.winfo_screenheight()
        x = max(0, int((sw - w) / 2))
        y = max(0, int((sh - h) / 2))
        top.geometry(f"{w}x{h}+{x}+{y}")

        # 简单模态
        try:
            top.grab_set()
        except tk.TclError:
            pass
        top.focus_force()
        self._root.wait_window(top)
        return bool(result.get())

    def _begin_capture(self) -> None:
        if not self._running or not self._enabled:
            with self._state_lock:
                self._trigger_pending = False
            return

        with self._state_lock:
            # 如果被其它逻辑抢先开始处理，就直接丢弃
            if self._processing is True:
                self._trigger_pending = False
                return
            self._processing = True
            self._trigger_pending = False

            self._flow_seq += 1
            self._flow_id = self._flow_seq
            self._flow_t0 = time.perf_counter()

            flow_id = self._flow_id

        backend = self._cfg.capture.backend
        monitor = self._cfg.capture.monitor

        # 在每次开始新流程前，先把“上次用户点击确认”的内容汇总并注入到模型上下文
        if self._confirm_enabled:
            ids = list(self._confirmed_since_last_flow)
            if ids:
                self._confirmed_since_last_flow.clear()
                try:
                    self._core.queue_confirmed_context(ids)
                    self._log(
                        f"已汇总用户确认内容并注入上下文 count={len(ids)}"
                    )
                except (RuntimeError, OSError, ValueError):
                    traceback.print_exc()

        self._log(
            f"开始翻译流程 flow={flow_id} backend={backend} "
            f"monitor={monitor}"
        )

        # 关键：开始一次翻译流程前，先清除屏幕上的 overlay，避免被截图识别
        self._hide_all_regions()

        def do_grab() -> None:
            try:
                frame = self._capturer.grab()
                # 只保留最新帧
                if self._frame_q.full():
                    try:
                        self._frame_q.get_nowait()
                    except Empty:
                        pass
                self._frame_q.put_nowait(frame)
            except (RuntimeError, OSError, ValueError):
                traceback.print_exc()
                # 抓屏失败：结束本次流程
                with self._state_lock:
                    self._processing = False

        self._root.after(self._capture_hide_delay_ms, do_grab)

    def _loop_worker(self) -> None:
        while self._running:
            if not self._enabled:
                time.sleep(0.1)
                continue
            try:
                try:
                    frame = self._frame_q.get(timeout=0.5)
                except Empty:
                    continue

                with self._state_lock:
                    flow_id = int(self._flow_id)
                    flow_t0 = float(self._flow_t0)

                roi = self._roi_abs

                # 1) 文字识别 -> 区域合并（同时合并识别文本）-> 显示识别结果
                recognized_raw = self._core.recognize_regions(frame, roi=roi)

                pre_boxes = [r.bbox for r in recognized_raw]
                merged_items = merge_and_sort(
                    [(r.bbox, r.jp_text) for r in recognized_raw],
                    merge_gap_in_line_ratio=float(
                        self._cfg.ocr.merge_gap_in_line_ratio
                    ),
                    merge_gap_between_lines_ratio=float(
                        self._cfg.ocr.merge_gap_between_lines_ratio
                    ),
                )
                merged_boxes = [m.bbox for m in merged_items]

                recognized = [
                    RegionOutput(
                        bbox=m.bbox,
                        jp_text=m.text,
                        translated=None,
                        engine="ai",
                        warning=None,
                    )
                    for m in merged_items
                    if (m.text or "").strip()
                ]
                elapsed = time.perf_counter() - flow_t0
                self._log(
                    f"识别完成 flow={flow_id} raw={len(recognized_raw)} "
                    f"merged={len(recognized)} elapsed={elapsed:.3f}s"
                )
                self._push_result(
                    PipelineOutput(
                        regions=recognized,
                        debug_boxes_pre=pre_boxes,
                        debug_boxes_merged=merged_boxes,
                    )
                )

                # 识别完成即可允许下一次输入事件触发（翻译放到后台线程）
                with self._state_lock:
                    self._processing = False

                self._start_translation_async(
                    flow_id=flow_id,
                    flow_t0=flow_t0,
                    recognized=recognized,
                    pre_boxes=pre_boxes,
                    merged_boxes=merged_boxes,
                )
            except (RuntimeError, OSError, ValueError):
                traceback.print_exc()
                with self._state_lock:
                    self._processing = False
                time.sleep(0.2)

    def _start_translation_async(
        self,
        *,
        flow_id: int,
        flow_t0: float,
        recognized: list[RegionOutput],
        pre_boxes: list[BBox],
        merged_boxes: list[BBox],
    ) -> None:
        waiter = _TranslateWaiter(
            flow_id=int(flow_id),
            flow_t0=float(flow_t0),
            recognized=recognized,
            pre_boxes=pre_boxes,
            merged_boxes=merged_boxes,
        )
        key = _regions_key(recognized)
        with self._translate_lock:
            self._latest_recognized_key = key

            # 1) 如果上一次翻译已经完成，且文本完全相同：直接复用结果
            if (
                self._translate_last_done_key == key
                and self._translate_last_done_map is not None
            ):
                self._log(f"复用已完成的翻译结果 flow={flow_id}")
                translated = _apply_translation_map(
                    recognized,
                    self._translate_last_done_map,
                )
                self._push_result(
                    PipelineOutput(
                        regions=translated,
                        debug_boxes_pre=pre_boxes,
                        debug_boxes_merged=merged_boxes,
                    )
                )
                return

            # 2) 如果有同文本的翻译正在进行：不打断，等待完成后复用
            if (
                self._translate_inflight_key == key
                and self._translate_thread is not None
                and self._translate_thread.is_alive()
            ):
                self._translate_waiter = waiter
                self._log(f"等待上一次翻译完成并复用 flow={flow_id}")
                return

            # 3) 文本不同：打断上一次翻译，启动新翻译
            if (
                self._translate_thread is not None
                and self._translate_thread.is_alive()
                and self._translate_inflight_key != key
                and self._translate_cancel is not None
            ):
                self._translate_cancel.set()

            cancel = threading.Event()
            self._translate_cancel = cancel
            self._translate_inflight_key = key
            self._translate_waiter = waiter

            t = threading.Thread(
                target=self._run_translate_worker,
                args=(key, cancel),
                daemon=True,
            )
            self._translate_thread = t
            t.start()

    def _run_translate_worker(
        self,
        key: tuple[str, ...],
        cancel: threading.Event,
    ) -> None:
        try:
            # 拿到“当次翻译应该基于哪个识别结果”的快照
            with self._translate_lock:
                waiter = self._translate_waiter
                if waiter is None or _regions_key(waiter.recognized) != key:
                    return
                recognized = waiter.recognized

            translated_regions = self._core.translate_regions(
                recognized,
                cancel_event=cancel,
            )
            if cancel.is_set():
                return

            tr_map = _build_translation_map(translated_regions)

            with self._translate_lock:
                # 如果这轮翻译已经不是当前 inflight（被新任务覆盖），丢弃
                if self._translate_inflight_key != key:
                    return

                self._translate_last_done_key = key
                self._translate_last_done_map = tr_map

                # 只更新“最新一次识别也是这个 key”的 UI，避免覆盖新内容
                if self._latest_recognized_key != key:
                    return

                waiter = self._translate_waiter
                if waiter is None or _regions_key(waiter.recognized) != key:
                    return

            elapsed = time.perf_counter() - float(waiter.flow_t0)
            self._log(
                f"翻译完成 flow={waiter.flow_id} "
                f"regions={len(translated_regions)} elapsed={elapsed:.3f}s"
            )

            translated_for_latest = _apply_translation_map(
                waiter.recognized,
                tr_map,
            )
            self._push_result(
                PipelineOutput(
                    regions=translated_for_latest,
                    debug_boxes_pre=waiter.pre_boxes,
                    debug_boxes_merged=waiter.merged_boxes,
                )
            )
        except (RuntimeError, OSError, ValueError):
            traceback.print_exc()

    def _push_result(self, out: PipelineOutput) -> None:
        try:
            if self._result_q.full():
                try:
                    self._result_q.get_nowait()
                except Empty:
                    pass
            self._result_q.put_nowait(out)
        except (RuntimeError, ValueError):
            # 队列满/竞争失败：允许丢帧，不影响下一轮刷新
            pass

    def _render_once(self) -> None:
        if not self._running:
            return

        latest: PipelineOutput | None = None
        try:
            while True:
                latest = self._result_q.get_nowait()
        except Empty:
            pass

        if latest is not None and self._enabled:
            self._redraw(latest)

        self._root.after(50, self._render_once)

    def _redraw(self, out: PipelineOutput) -> None:
        regions = out.regions

        # 复用窗口：按顺序一一对应，避免频繁创建导致抢焦点
        while len(self._region_windows) < len(regions):
            self._region_windows.append(
                _RegionWindow(
                    root=self._root,
                    font=self._font,
                    screen_w=self._screen_w,
                    screen_h=self._screen_h,
                    on_confirm=self._on_confirm_content,
                )
            )
        while len(self._tr_windows) < len(regions):
            self._tr_windows.append(
                _RegionWindow(
                    root=self._root,
                    font=self._font,
                    screen_w=self._screen_w,
                    screen_h=self._screen_h,
                    on_confirm=self._on_confirm_content,
                )
            )

        rects: list[tuple[int, int, int, int]] = []
        for i, r in enumerate(regions):
            cid = make_request_id((r.jp_text or "").strip())
            confirmed = cid in self._confirmed_ids
            # 识别窗口：只显示原文，不带前缀
            x, y, _, h = self._region_windows[i].show(
                text=(r.jp_text or "").strip(),
                bbox=r.bbox,
                warning=False,
                prefer_xy=None,
                content_id=cid,
                confirmed=confirmed,
            )
            rects.append((int(x), int(y), int(x) + int(_), int(y) + int(h)))

            # 翻译窗口：只显示译文/警告，不带前缀
            tr_lines: list[str] = []
            if (r.warning or "").strip():
                tr_lines.append((r.warning or "").strip())
            if (r.translated or "").strip():
                tr_lines.append((r.translated or "").strip())

            if tr_lines:
                pad = 6
                tx, ty, tw, th = self._tr_windows[i].show(
                    text="\n".join(tr_lines),
                    bbox=r.bbox,
                    warning=bool((r.warning or "").strip()),
                    prefer_xy=(int(x), int(y + h + pad)),
                    content_id=cid,
                    confirmed=confirmed,
                )
                rects.append(
                    (int(tx), int(ty), int(tx) + int(tw), int(ty) + int(th))
                )
            else:
                self._tr_windows[i].hide()

        for j in range(len(regions), len(self._region_windows)):
            self._region_windows[j].hide()
        for j in range(len(regions), len(self._tr_windows)):
            self._tr_windows[j].hide()

        # 更新缓存：用于全局点击事件做“命中窗口则不触发新流程”的判断。
        self._content_window_rects = rects

        # 画框：合并前/合并后
        self._draw_debug_boxes(out)

    def _on_confirm_content(self, content_id: str) -> None:
        if not self._confirm_enabled:
            return
        cid = (content_id or "").strip()
        if not cid:
            return

        if len(self._confirmed_ids) > 2048:
            # 防止无限增长（id 基于文本 hash，理论上会复用）
            self._confirmed_ids.clear()

        # 再次点击同一内容：取消确认，防止误点污染上下文
        if cid in self._confirmed_ids:
            self._confirmed_ids.discard(cid)
            self._confirmed_since_last_flow.discard(cid)
            confirmed = False
        else:
            self._confirmed_ids.add(cid)
            self._confirmed_since_last_flow.add(cid)
            confirmed = True

        # 轻量 UI 反馈：立刻刷新背景（不等下一次 redraw）
        for w in (*self._region_windows, *self._tr_windows):
            try:
                if (w.content_id or "") == cid:
                    w.set_confirmed(confirmed)
            except (RuntimeError, OSError, ValueError, tk.TclError):
                traceback.print_exc()

    def _draw_debug_boxes(self, out: PipelineOutput) -> None:
        pre = list(out.debug_boxes_pre or [])
        merged = list(out.debug_boxes_merged or [])

        # 复用窗口：每个 bbox 对应一个 _BoxBorder（内部 4 个 Toplevel）
        while len(self._pre_box_windows) < len(pre):
            self._pre_box_windows.append(
                _BoxBorder(
                    root=self._root,
                    screen_w=self._screen_w,
                    screen_h=self._screen_h,
                )
            )
        while len(self._merged_box_windows) < len(merged):
            self._merged_box_windows.append(
                _BoxBorder(
                    root=self._root,
                    screen_w=self._screen_w,
                    screen_h=self._screen_h,
                )
            )

        # 合并前：青色；合并后：黄色，并轻微向外偏移以避免重叠
        for i, b in enumerate(pre):
            self._pre_box_windows[i].show_bbox(
                bbox=b,
                color="#00e5ff",
                thickness=2,
                outset=0,
            )
        for j in range(len(pre), len(self._pre_box_windows)):
            self._pre_box_windows[j].hide()

        for i, b in enumerate(merged):
            self._merged_box_windows[i].show_bbox(
                bbox=b,
                color="#ffd500",
                thickness=2,
                outset=2,
            )
        for j in range(len(merged), len(self._merged_box_windows)):
            self._merged_box_windows[j].hide()


@dataclass(frozen=True)
class _TranslateWaiter:
    flow_id: int
    flow_t0: float
    recognized: list[RegionOutput]
    pre_boxes: list[BBox]
    merged_boxes: list[BBox]


def _regions_key(regions: list[RegionOutput]) -> tuple[str, ...]:
    return tuple((r.jp_text or "").strip() for r in regions)


def _build_translation_map(
    regions: list[RegionOutput],
) -> dict[str, tuple[str | None, str, str | None]]:
    out: dict[str, tuple[str | None, str, str | None]] = {}
    for r in regions:
        jp = (r.jp_text or "").strip()
        if not jp:
            continue
        out[jp] = (r.translated, r.engine, r.warning)
    return out


def _apply_translation_map(
    recognized: list[RegionOutput],
    tr_map: dict[str, tuple[str | None, str, str | None]],
) -> list[RegionOutput]:
    out: list[RegionOutput] = []
    for r in recognized:
        jp = (r.jp_text or "").strip()
        if jp and jp in tr_map:
            translated, engine, warning = tr_map[jp]
            out.append(
                RegionOutput(
                    bbox=r.bbox,
                    jp_text=jp,
                    translated=translated,
                    engine=engine,
                    warning=warning,
                )
            )
        else:
            out.append(r)
    return out


class _RegionWindow:
    def __init__(
        self,
        *,
        root: tk.Tk,
        font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
        screen_w: int,
        screen_h: int,
        on_confirm: Callable[[str], None] | None = None,
    ):
        self._root = root
        self._font = font
        self._screen_w = screen_w
        self._screen_h = screen_h
        self._on_confirm = on_confirm
        self._content_id: str | None = None

        self._visible = False
        self._last_text: str | None = None
        self._last_warning: bool = False
        self._last_confirmed: bool = False
        self._last_xy: tuple[int, int] | None = None

        self._top = tk.Toplevel(self._root)
        self._top.overrideredirect(True)
        try:
            self._top.wm_attributes("-topmost", True)
        except tk.TclError:
            pass

        # 避免参与任务栏/alt-tab：不同 WM 支持程度不一，失败不影响功能
        try:
            self._top.wm_attributes("-toolwindow", True)
        except tk.TclError:
            pass

        self._label = tk.Label(self._top, bd=0, highlightthickness=0)
        self._label.pack(fill="both", expand=True)

        # 点击确认：绑定在 label 与 top 上，避免不同 WM 下事件丢失
        try:
            self._label.bind("<Button-1>", self._on_click)
            self._top.bind("<Button-1>", self._on_click)
        except tk.TclError:
            pass

        self._photo: ImageTk.PhotoImage | None = None
        self.hide()

    @property
    def content_id(self) -> str | None:
        return self._content_id

    def hide(self) -> None:
        self._visible = False
        try:
            self._top.withdraw()
        except tk.TclError:
            pass

    def set_confirmed(self, confirmed: bool) -> None:
        if not self._visible:
            return
        if self._last_text is None:
            return
        confirmed = bool(confirmed)
        if self._last_confirmed == confirmed:
            return

        self._last_confirmed = confirmed

        img = _render_text_box(
            self._last_text,
            font=self._font,
            warning=bool(self._last_warning),
            confirmed=bool(self._last_confirmed),
        )
        self._photo = ImageTk.PhotoImage(img)
        self._label.configure(image=self._photo)

        if self._last_xy is not None:
            x, y = self._last_xy
            x, y = _clamp_xy(
                x=int(x),
                y=int(y),
                w=int(img.width),
                h=int(img.height),
                screen_w=int(self._screen_w),
                screen_h=int(self._screen_h),
            )
            try:
                self._top.geometry(f"{img.width}x{img.height}+{x}+{y}")
                self._top.deiconify()
            except tk.TclError:
                pass

    def show(
        self,
        *,
        text: str,
        bbox: BBox,
        warning: bool,
        prefer_xy: tuple[int, int] | None,
        content_id: str | None,
        confirmed: bool,
    ) -> tuple[int, int, int, int]:
        text = (text or "").strip()
        if not text:
            self.hide()
            return (0, 0, 0, 0)

        self._content_id = (content_id or "").strip() or None

        self._visible = True
        self._last_text = text
        self._last_warning = bool(warning)
        self._last_confirmed = bool(confirmed)

        img = _render_text_box(
            text,
            font=self._font,
            warning=bool(warning),
            confirmed=bool(confirmed),
        )
        self._photo = ImageTk.PhotoImage(img)
        self._label.configure(image=self._photo)

        pad = 6
        if prefer_xy is None:
            x, y = _choose_overlay_position(
                bbox=bbox,
                w=img.width,
                h=img.height,
                screen_w=self._screen_w,
                screen_h=self._screen_h,
                pad=pad,
            )
        else:
            x, y = _clamp_xy(
                x=int(prefer_xy[0]),
                y=int(prefer_xy[1]),
                w=int(img.width),
                h=int(img.height),
                screen_w=int(self._screen_w),
                screen_h=int(self._screen_h),
            )

        try:
            self._top.geometry(f"{img.width}x{img.height}+{x}+{y}")
            self._top.deiconify()
        except tk.TclError:
            pass

        self._last_xy = (int(x), int(y))

        return (int(x), int(y), int(img.width), int(img.height))

    def _on_click(self, e: tk.Event) -> str | None:
        if self._on_confirm is None:
            return None
        cid = (self._content_id or "").strip()
        if not cid:
            return None
        try:
            self._on_confirm(cid)
        except (RuntimeError, OSError, ValueError):
            traceback.print_exc()

        # 事件冒泡：点击 label 时也会触发到 top。
        # 为避免一次点击触发两次确认（尤其 toggle 会被立刻反向取消），
        # 在 label 的 handler 里返回 'break' 阻止继续传播。
        try:
            if e.widget is self._label:
                return "break"
        except (AttributeError, RuntimeError, tk.TclError):
            pass
        return None


def _clamp_xy(
    *,
    x: int,
    y: int,
    w: int,
    h: int,
    screen_w: int,
    screen_h: int,
) -> tuple[int, int]:
    if x + w > screen_w:
        x = screen_w - w
    if y + h > screen_h:
        y = screen_h - h
    if x < 0:
        x = 0
    if y < 0:
        y = 0
    return int(x), int(y)


class _BoxBorder:
    """用 4 个小窗口画矩形边框（只占边缘像素，尽量避免遮挡框内内容）。"""

    def __init__(self, *, root: tk.Tk, screen_w: int, screen_h: int):
        self._root = root
        self._screen_w = screen_w
        self._screen_h = screen_h

        self._tops = [tk.Toplevel(self._root) for _ in range(4)]
        for t in self._tops:
            t.overrideredirect(True)
            try:
                t.wm_attributes("-topmost", True)
            except tk.TclError:
                pass
            try:
                t.wm_attributes("-toolwindow", True)
            except tk.TclError:
                pass
        self.hide()

    def hide(self) -> None:
        for t in self._tops:
            try:
                t.withdraw()
            except tk.TclError:
                pass

    def show_bbox(
        self,
        *,
        bbox: BBox,
        color: str,
        thickness: int,
        outset: int,
    ) -> None:
        t = max(1, int(thickness))
        o = max(0, int(outset))

        x1 = int(bbox.x1) - o
        y1 = int(bbox.y1) - o
        x2 = int(bbox.x2) + o
        y2 = int(bbox.y2) + o

        # clamp
        x1 = max(0, min(self._screen_w, x1))
        y1 = max(0, min(self._screen_h, y1))
        x2 = max(0, min(self._screen_w, x2))
        y2 = max(0, min(self._screen_h, y2))

        w = max(0, x2 - x1)
        h = max(0, y2 - y1)
        if w <= 1 or h <= 1:
            self.hide()
            return

        # 上/下/左/右
        top, bot, left, right = self._tops

        # 统一背景色
        for tw in (top, bot, left, right):
            try:
                tw.configure(bg=color)
            except tk.TclError:
                pass

        # top
        self._place(top, x1, y1, w, t)
        # bottom
        self._place(bot, x1, y2 - t, w, t)
        # left
        self._place(left, x1, y1, t, h)
        # right
        self._place(right, x2 - t, y1, t, h)

    def _place(self, win: tk.Toplevel, x: int, y: int, w: int, h: int) -> None:
        if w <= 0 or h <= 0:
            try:
                win.withdraw()
            except tk.TclError:
                pass
            return
        try:
            win.geometry(f"{w}x{h}+{x}+{y}")
            win.deiconify()
        except tk.TclError:
            pass


def _render_text_box(
    text: str,
    *,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    warning: bool,
    confirmed: bool,
) -> Image.Image:
    dummy = Image.new("RGB", (10, 10), (255, 255, 255))
    draw = ImageDraw.Draw(dummy)
    bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=4)
    tw = int(bbox[2] - bbox[0])
    th = int(bbox[3] - bbox[1])

    pad_x = 10
    pad_y = 8
    w = tw + pad_x * 2
    h = th + pad_y * 2

    # 统一配色：白底黑字；warning 用红字；confirmed 用淡绿底
    bg = (225, 255, 225) if confirmed else (255, 255, 255)
    fg = (200, 0, 0) if warning else (0, 0, 0)

    img = Image.new("RGB", (w, h), bg)
    draw = ImageDraw.Draw(img)
    draw.multiline_text(
        (pad_x, pad_y),
        text,
        font=font,
        fill=fg,
        spacing=4,
    )
    return img


def _render_button_box(
    text: str,
    *,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    bg: tuple[int, int, int],
    fg: tuple[int, int, int],
    border: tuple[int, int, int] = (120, 120, 120),
) -> Image.Image:
    """用 Pillow 渲染一个“按钮风格”的文本框。"""

    t = (text or "").strip()
    dummy = Image.new("RGB", (10, 10), (255, 255, 255))
    draw = ImageDraw.Draw(dummy)
    bbox = draw.textbbox((0, 0), t, font=font)
    tw = int(bbox[2] - bbox[0])
    th = int(bbox[3] - bbox[1])

    pad_x = 18
    pad_y = 10
    w = tw + pad_x * 2
    h = th + pad_y * 2

    img = Image.new("RGB", (w, h), bg)
    draw = ImageDraw.Draw(img)
    try:
        draw.rounded_rectangle(
            (0, 0, w - 1, h - 1),
            radius=10,
            outline=border,
            width=2,
        )
    except (AttributeError, TypeError, ValueError):
        # Pillow 版本不支持 rounded_rectangle 时，退化为普通矩形
        draw.rectangle((0, 0, w - 1, h - 1), outline=border, width=2)

    draw.text((pad_x, pad_y), t, font=font, fill=fg)
    return img


def _choose_overlay_position(
    *,
    bbox: BBox,
    w: int,
    h: int,
    screen_w: int,
    screen_h: int,
    pad: int,
) -> tuple[int, int]:
    """优先把窗口放在识别区域下方，放不下则自适应挪到周围其它位置。"""

    # 候选位置顺序：下 -> 右 -> 左 -> 上
    candidates: list[tuple[int, int]] = [
        (int(bbox.x1), int(bbox.y2) + pad),  # below
        (int(bbox.x2) + pad, int(bbox.y1)),  # right
        (int(bbox.x1) - w - pad, int(bbox.y1)),  # left
        (int(bbox.x1), int(bbox.y1) - h - pad),  # above
    ]

    def fits(x: int, y: int) -> bool:
        return (
            (0 <= x)
            and (0 <= y)
            and (x + w <= screen_w)
            and (y + h <= screen_h)
        )

    for x0, y0 in candidates:
        if fits(x0, y0):
            return x0, y0

    # 都放不下：尽量贴近“下方优先”的目标点，然后 clamp 到屏幕内
    x = int(bbox.x1)
    y = int(bbox.y2) + pad

    if x + w > screen_w:
        x = screen_w - w
    if y + h > screen_h:
        y = screen_h - h

    if x < 0:
        x = 0
    if y < 0:
        y = 0
    return x, y


class _RoiSelector:
    def __init__(self, *, root: tk.Tk, image: Image.Image):
        self._root = root
        self._image = image

        self._top = tk.Toplevel(self._root)
        self._top.title("选择识别区域")
        try:
            self._top.wm_attributes("-topmost", True)
        except tk.TclError:
            pass

        # 全屏展示截图
        try:
            self._top.attributes("-fullscreen", True)
        except tk.TclError:
            self._top.geometry(
                f"{image.width}x{image.height}+0+0"
            )

        self._canvas = tk.Canvas(
            self._top,
            width=image.width,
            height=image.height,
            highlightthickness=0,
            bd=0,
        )
        self._canvas.pack(fill="both", expand=True)

        self._photo = ImageTk.PhotoImage(image)
        self._canvas.create_image(0, 0, anchor="nw", image=self._photo)

        self._start_xy: tuple[int, int] | None = None
        self._rect_id: int | None = None
        self._result: BBox | None = None

        self._canvas.bind("<ButtonPress-1>", self._on_down)
        self._canvas.bind("<B1-Motion>", self._on_move)
        self._canvas.bind("<ButtonRelease-1>", self._on_up)

        self._top.bind("<Return>", self._on_confirm)
        self._top.bind("<Escape>", self._on_cancel)
        self._top.protocol("WM_DELETE_WINDOW", self._cancel)

        self._top.focus_force()

    def run(self) -> BBox | None:
        self._root.wait_window(self._top)
        return self._result

    def _on_down(self, e: tk.Event) -> None:
        x = int(getattr(e, "x", 0))
        y = int(getattr(e, "y", 0))
        self._start_xy = (x, y)

        if self._rect_id is not None:
            try:
                self._canvas.delete(self._rect_id)
            except tk.TclError:
                pass
            self._rect_id = None

        self._rect_id = self._canvas.create_rectangle(
            x,
            y,
            x,
            y,
            outline="#00ff66",
            width=3,
        )

    def _on_move(self, e: tk.Event) -> None:
        if self._start_xy is None or self._rect_id is None:
            return
        x0, y0 = self._start_xy
        x1 = int(getattr(e, "x", 0))
        y1 = int(getattr(e, "y", 0))
        try:
            self._canvas.coords(self._rect_id, x0, y0, x1, y1)
        except tk.TclError:
            pass

    def _on_up(self, e: tk.Event) -> None:
        if self._start_xy is None:
            return
        x0, y0 = self._start_xy
        x1 = int(getattr(e, "x", 0))
        y1 = int(getattr(e, "y", 0))

        x_min = max(0, min(x0, x1))
        y_min = max(0, min(y0, y1))
        x_max = min(self._image.width, max(x0, x1))
        y_max = min(self._image.height, max(y0, y1))

        if (x_max - x_min) >= 8 and (y_max - y_min) >= 8:
            self._result = BBox(x_min, y_min, x_max, y_max)
        else:
            self._result = None

    def _on_confirm(self, _e: tk.Event) -> None:
        if self._result is None:
            return
        self._close()

    def _on_cancel(self, _e: tk.Event) -> None:
        self._cancel()

    def _cancel(self) -> None:
        self._result = None
        self._close()

    def _close(self) -> None:
        try:
            self._top.destroy()
        except tk.TclError:
            pass

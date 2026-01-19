from __future__ import annotations

from dataclasses import dataclass
import threading
import subprocess
import tempfile
from pathlib import Path
import shutil

import numpy as np
from mss import mss
from mss.exception import ScreenShotError
from PIL import Image


@dataclass(frozen=True)
class Frame:
    image: Image.Image
    width: int
    height: int
    left: int
    top: int


class ScreenCapturer:
    def __init__(self, monitor: int = 1, backend: str = "auto"):
        # mss 在 Linux 下会把 display 放在线程局部存储里。
        # 不能在主线程创建实例、却在后台线程调用 grab；否则会触发：
        # AttributeError: '_thread._local' object has no attribute 'display'
        # 因此每个线程懒初始化一个独立的 mss() 实例。
        self._tls = threading.local()
        self._monitor_index = monitor
        self._backend = (backend or "auto").strip().lower()

    def _get_sct(self):
        sct = getattr(self._tls, "sct", None)
        if sct is None:
            sct = mss()
            self._tls.sct = sct
        return sct

    def list_monitors(self) -> list[dict]:
        return list(self._get_sct().monitors)

    def grab(self) -> Frame:
        if self._backend == "qt":
            return self._grab_qt()
        if self._backend == "spectacle":
            return self._grab_spectacle()
        if self._backend == "mss":
            return self._grab_mss()
        # auto
        try:
            return self._grab_mss()
        except ScreenShotError:
            try:
                return self._grab_spectacle()
            except (OSError, RuntimeError, subprocess.SubprocessError):
                return self._grab_qt()

    def _grab_mss(self) -> Frame:
        sct = self._get_sct()
        monitors = sct.monitors
        if self._monitor_index < 0 or self._monitor_index >= len(monitors):
            raise ValueError(
                (
                    f"monitor 索引无效：{self._monitor_index}，"
                    f"可用范围 0..{len(monitors)-1}"
                )
            )

        mon = monitors[self._monitor_index]
        raw = sct.grab(mon)
        arr = np.array(raw)  # BGRA
        img = Image.fromarray(arr[:, :, :3][:, :, ::-1])  # to RGB
        return Frame(
            image=img,
            width=img.width,
            height=img.height,
            left=int(mon.get("left", 0) or 0),
            top=int(mon.get("top", 0) or 0),
        )

    def _grab_qt(self) -> Frame:
        try:
            from PySide6 import QtGui  # 懒加载：Tkinter 版本无需强依赖 PySide6
        except ImportError as e:
            raise RuntimeError(
                f"Qt 抓屏不可用：无法导入 PySide6: {e}"
            ) from e

        app = QtGui.QGuiApplication.instance()
        if app is None:
            raise RuntimeError("Qt 抓屏需要先创建 QApplication")

        screens = QtGui.QGuiApplication.screens()
        screen: QtGui.QScreen | None = None
        if self._monitor_index <= 1:
            screen = QtGui.QGuiApplication.primaryScreen()
        else:
            idx = self._monitor_index - 1
            if 0 <= idx < len(screens):
                screen = screens[idx]
        if screen is None:
            screen = QtGui.QGuiApplication.primaryScreen()
        if screen is None:
            raise RuntimeError("未找到可用屏幕")

        pix = screen.grabWindow(0)
        if pix.isNull():
            platform = QtGui.QGuiApplication.platformName()
            screens_info = ", ".join(
                (
                    f"{i}:{s.geometry().width()}x{s.geometry().height()}"
                    for i, s in enumerate(QtGui.QGuiApplication.screens())
                )
            )
            hint = ""
            if platform == "wayland":
                hint = (
                    "（检测到 Wayland：Qt 的 grabWindow(0) 通常会被禁止/返回空图像。"
                    "建议切换到 X11/Xorg 会话，或让游戏运行在 XWayland 下再试。）"
                )
            raise RuntimeError(
                (
                    f"Qt 抓屏返回空 Pixmap {hint} "
                    f"platform={platform}, screens=[{screens_info}]"
                )
            )

        qimg = pix.toImage().convertToFormat(
            QtGui.QImage.Format.Format_RGBA8888)
        w = qimg.width()
        h = qimg.height()
        if w <= 0 or h <= 0:
            platform = QtGui.QGuiApplication.platformName()
            raise RuntimeError(f"Qt 抓屏得到空图像 platform={platform}")

        ptr = qimg.bits()
        size = qimg.bytesPerLine() * h
        buf = bytes(ptr[:size])
        img = Image.frombuffer("RGBA", (w, h), buf, "raw",
                               "RGBA", qimg.bytesPerLine(), 1).convert("RGB")
        g = screen.geometry()
        return Frame(
            image=img,
            width=w,
            height=h,
            left=int(g.x()),
            top=int(g.y()),
        )

    def _grab_spectacle(self) -> Frame:
        exe = shutil.which("spectacle")
        if not exe:
            raise RuntimeError(
                "未找到 spectacle，可用 `sudo apt install spectacle` 或使用其它后端")

        # spectacle 的 -m 是“当前屏幕”。我们的 monitor 配置是 mss 风格：1=主屏。
        # 这里先用 -f 全屏更稳定；后续如需更精确可扩展为按屏幕/窗口截图。
        with tempfile.TemporaryDirectory(prefix="autotransgal-") as td:
            out_path = Path(td) / "shot.png"
            cmd = [
                exe,
                "--background",
                "--nonotify",
                "--fullscreen",
                "--output",
                str(out_path),
            ]
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                stderr = (proc.stderr or proc.stdout or "").strip()
                raise RuntimeError(
                    f"spectacle 截图失败：exit={proc.returncode} {stderr}")

            if not out_path.exists() or out_path.stat().st_size <= 0:
                raise RuntimeError("spectacle 未生成有效截图文件")

            img = Image.open(out_path).convert("RGB")
            # spectacle 的输出通常是全屏截图；多显示器下坐标可能是虚拟桌面坐标系。
            return Frame(
                image=img,
                width=img.width,
                height=img.height,
                left=0,
                top=0,
            )

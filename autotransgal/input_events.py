from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable

from pynput import keyboard, mouse


@dataclass(frozen=True)
class InputEventsConfig:
    # 是否监听鼠标移动。默认关闭，避免海量事件。
    listen_mouse_move: bool = False


@dataclass(frozen=True)
class InputActivity:
    kind: str  # key | mouse_click | mouse_scroll | mouse_move
    x: int | None = None
    y: int | None = None
    pressed: bool | None = None


class InputEventsWatcher:
    """监听全局键鼠事件，并在事件发生时调用回调。

    设计目标：
    - 只负责“通知有活动”，不做任何节流/队列
    - 回调可能在 listener 线程中调用；上层需自行线程安全
    """

    def __init__(
        self,
        *,
        on_activity: Callable[[InputActivity], None],
        cfg: InputEventsConfig | None = None,
    ) -> None:
        self._cfg = cfg or InputEventsConfig()
        self._on_activity = on_activity

        self._kbd: keyboard.Listener | None = None
        self._mouse: mouse.Listener | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        with self._lock:
            if self._kbd is not None or self._mouse is not None:
                return

            self._kbd = keyboard.Listener(on_press=self._on_kbd)
            self._kbd.start()

            self._mouse = mouse.Listener(
                on_click=self._on_mouse_click,
                on_scroll=self._on_mouse_scroll,
                on_move=(
                    (self._on_mouse_move)
                    if self._cfg.listen_mouse_move
                    else None
                ),
            )
            self._mouse.start()

    def stop(self) -> None:
        with self._lock:
            if self._kbd is not None:
                self._kbd.stop()
                self._kbd = None
            if self._mouse is not None:
                self._mouse.stop()
                self._mouse = None

    def _on_kbd(self, _key: keyboard.Key | keyboard.KeyCode | None) -> None:
        self._on_activity(InputActivity(kind="key"))

    def _on_mouse_click(
        self,
        _x: int,
        _y: int,
        _button: mouse.Button,
        pressed: bool,
    ) -> None:
        if pressed:
            self._on_activity(
                InputActivity(
                    kind="mouse_click",
                    x=int(_x),
                    y=int(_y),
                    pressed=True,
                )
            )

    def _on_mouse_scroll(self, _x: int, _y: int, _dx: int, _dy: int) -> None:
        self._on_activity(
            InputActivity(kind="mouse_scroll", x=int(_x), y=int(_y))
        )

    def _on_mouse_move(self, _x: int, _y: int) -> None:
        self._on_activity(
            InputActivity(kind="mouse_move", x=int(_x), y=int(_y))
        )

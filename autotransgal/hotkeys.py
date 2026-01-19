from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from pynput import keyboard


@dataclass(frozen=True)
class HotkeyConfig:
    toggle: str = "<ctrl>+<alt>+t"
    quit: str = "<ctrl>+<alt>+q"


class HotkeyManager:
    def __init__(
        self,
        *,
        on_toggle: Callable[[], None],
        on_quit: Callable[[], None],
        cfg: HotkeyConfig | None = None,
    ):
        self._cfg = cfg or HotkeyConfig()
        self._listener: keyboard.GlobalHotKeys | None = None
        self._on_toggle = on_toggle
        self._on_quit = on_quit

    def start(self) -> None:
        if self._listener is not None:
            return
        self._listener = keyboard.GlobalHotKeys(
            {
                self._cfg.toggle: self._on_toggle,
                self._cfg.quit: self._on_quit,
            }
        )
        self._listener.start()

    def stop(self) -> None:
        if self._listener is None:
            return
        self._listener.stop()
        self._listener = None

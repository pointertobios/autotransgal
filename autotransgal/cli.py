import argparse
from pathlib import Path

from autotransgal.config import load_config, write_default_config
from autotransgal.input_events import InputEventsWatcher
from autotransgal.hotkeys import HotkeyManager
from autotransgal.ui.overlay import OverlayWindow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cli", description="autotransgal command line interface")
    parser.add_argument("--version", action="store_true",
                        help="print version and exit")
    parser.add_argument("--config", type=str,
                        help="path to configuration file (optional)")
    parser.add_argument("--doctor", action="store_true",
                        help="probe openai-compatible endpoint and exit")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.version:
        print("autotransgal 0.1.0")
        return 0

    config_path = Path(args.config) if args.config else None
    try:
        cfg = load_config(config_path)
    except FileNotFoundError:
        created = write_default_config(config_path)
        print(f"已生成配置文件：{created}，请填写 [openai].api_key 后重新运行。")
        return 2

    if args.doctor:
        from autotransgal.diagnostics import diagnose_openai_compatible

        print(diagnose_openai_compatible(cfg.openai))
        return 0

    win = OverlayWindow(cfg)
    win.start()

    hk = HotkeyManager(
        on_toggle=win.toggle_enabled,
        on_quit=win.quit,
    )
    hk.start()

    watcher = InputEventsWatcher(on_activity=win.notify_activity)
    watcher.start()

    try:
        win.run()
        return 0
    except KeyboardInterrupt:
        pass
    finally:
        watcher.stop()
        hk.stop()
        win.stop()


if __name__ == "__main__":
    raise SystemExit(main())

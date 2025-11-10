from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import QApplication
from loguru import logger

from core.device_manager import DeviceManager
from core.game_registry import GameRegistry
from core.layout_registry import LayoutRegistry
from core.scenario_registry import ScenarioRegistry
from gui.main_window import MainWindow


def run(debug: bool = False, adb_path: Optional[str] = None) -> int:
    """
    Entry point for the GUI application.

    Parameters
    ----------
    debug:
        When True, enables verbose logging to stdout.
    adb_path:
        Optional explicit path to the adb executable. When omitted, the
        system PATH is used.
    """
    if debug:
        logger.enable("__main__")
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.disable("__main__")

    app = QApplication(sys.argv)
    app.setApplicationName("Universal Game Bot Device Manager")

    device_manager = DeviceManager(adb_path=adb_path)
    game_registry = GameRegistry(storage_path=Path("games.json"))
    layout_registry = LayoutRegistry(storage_path=Path("layouts.json"))
    scenario_registry = ScenarioRegistry(storage_path=Path("scenarios.json"))
    window = MainWindow(
        device_manager=device_manager,
        game_registry=game_registry,
        layout_registry=layout_registry,
        scenario_registry=scenario_registry,
    )
    window.show()

    exit_code = app.exec()
    return exit_code


if __name__ == "__main__":
    sys.exit(run(debug="--debug" in sys.argv))


"""Core services for the Universal Game Bot device manager."""

from .adb_client import AdbClient, DeviceInfo
from .device_manager import DeviceManager
from .game_registry import GameConfig, GameRegistry
from .layout_registry import LayoutRegistry, RegionConfig

__all__ = [
    "AdbClient",
    "DeviceInfo",
    "DeviceManager",
    "GameConfig",
    "GameRegistry",
    "LayoutRegistry",
    "RegionConfig",
]


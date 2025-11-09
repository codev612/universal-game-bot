from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Optional

from PyQt6.QtCore import QObject, QRunnable, QThreadPool, pyqtSignal
from loguru import logger

from .adb_client import AdbClient, DeviceInfo


class DeviceManager(QObject):
    """Coordinates ADB discovery and exposes results via Qt signals."""

    devices_updated = pyqtSignal(list)
    error_occurred = pyqtSignal(str)
    device_connected = pyqtSignal(str, str)  # serial, alias
    app_status_ready = pyqtSignal(str, str, bool, bool)  # serial, package, installed, running
    screenshot_captured = pyqtSignal(str, bytes)  # serial, png bytes
    tap_sent = pyqtSignal(str, int, int)  # serial, x, y
    swipe_sent = pyqtSignal(str, int, int, int, int, int)  # serial, x1, y1, x2, y2, duration

    def __init__(self, adb_path: Optional[str] = None, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._adb_client = AdbClient(adb_path=adb_path)
        self._devices: List[DeviceInfo] = []
        self._thread_pool = QThreadPool.globalInstance()
        self._aliases: Dict[str, str] = {}
        self._shutdown = False

    @property
    def devices(self) -> List[DeviceInfo]:
        return list(self._devices)

    def refresh_devices(self) -> None:
        """Refresh the device list asynchronously."""
        task = _RefreshDevicesTask(
            client=self._adb_client,
            on_success=self._handle_refresh_success,
            on_error=self._handle_refresh_error,
        )
        logger.debug("Queueing device refresh task")
        self._thread_pool.start(task)

    def connect_device(self, alias: str, host: str, port: int) -> None:
        """Connect to a remote ADB endpoint and associate it with an alias."""
        task = _ConnectDeviceTask(
            client=self._adb_client,
            host=host,
            port=port,
            on_success=lambda serial: self._handle_connect_success(serial, alias),
            on_error=lambda message: self._handle_connect_error(message, alias, host, port),
        )
        logger.debug("Queueing connect task host={} port={} alias={}", host, port, alias)
        self._thread_pool.start(task)

    def check_game_running(self, serial: str, package: str) -> None:
        """Check whether the given package is installed and running on the device."""
        task = _CheckAppTask(
            client=self._adb_client,
            serial=serial,
            package=package,
            on_success=lambda status: self._handle_check_success(serial, package, status),
            on_error=lambda message: self._handle_check_error(message, serial, package),
        )
        logger.debug("Queueing app status task serial={} package={}", serial, package)
        self._thread_pool.start(task)

    def capture_screenshot(self, serial: str) -> None:
        """Capture a screenshot from the given device."""
        task = _ScreenshotTask(
            client=self._adb_client,
            serial=serial,
            on_success=lambda data: self._handle_screenshot_success(serial, data),
            on_error=lambda message: self._handle_screenshot_error(message, serial),
        )
        logger.debug("Queueing screenshot task serial={}", serial)
        self._thread_pool.start(task)

    def send_tap(self, serial: str, x: int, y: int) -> None:
        """Send a tap command to the given device asynchronously."""
        task = _TapTask(
            client=self._adb_client,
            serial=serial,
            x=x,
            y=y,
            on_success=lambda: self._handle_tap_success(serial, x, y),
            on_error=lambda message: self.error_occurred.emit(f"Tap failed ({serial}): {message}"),
        )
        logger.debug("Queueing tap task serial={} x={} y={}", serial, x, y)
        self._thread_pool.start(task)

    def send_swipe(
        self,
        serial: str,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        duration_ms: int = 300,
    ) -> None:
        """Send a swipe command to the given device asynchronously."""
        task = _SwipeTask(
            client=self._adb_client,
            serial=serial,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            duration_ms=duration_ms,
            on_success=lambda: self._handle_swipe_success(serial, x1, y1, x2, y2, duration_ms),
            on_error=lambda message: self.error_occurred.emit(f"Swipe failed ({serial}): {message}"),
        )
        logger.debug(
            "Queueing swipe task serial={} x1={} y1={} x2={} y2={} duration={}",
            serial,
            x1,
            y1,
            x2,
            y2,
            duration_ms,
        )
        self._thread_pool.start(task)

    def shutdown(self, wait: bool = True, timeout_ms: int = 2000) -> None:
        """Prevent further signal emission and optionally wait for pending tasks."""
        self._shutdown = True
        if wait:
            self._thread_pool.waitForDone(timeout_ms)

    def _handle_refresh_success(self, devices: List[DeviceInfo]) -> None:
        if self._shutdown:
            return
        self._devices = [
            replace(device, alias=self._aliases.get(device.serial))
            for device in devices
        ]
        logger.debug("Device refresh success count={}", len(devices))
        self.devices_updated.emit(self.devices)

    def _handle_refresh_error(self, message: str) -> None:
        if self._shutdown:
            return
        logger.error("Device refresh failed: {}", message)
        self.error_occurred.emit(message)

    def _handle_connect_success(self, serial: str, alias: str) -> None:
        if self._shutdown:
            return
        logger.info("Device connected serial={} alias={}", serial, alias)
        if alias:
            self._aliases[serial] = alias
        self.device_connected.emit(serial, alias)
        self.refresh_devices()

    def _handle_connect_error(self, message: str, alias: str, host: str, port: int) -> None:
        if self._shutdown:
            return
        logger.error(
            "Device connect failed alias={} host={} port={} message={}",
            alias,
            host,
            port,
            message,
        )
        self.error_occurred.emit(f"Connect failed ({host}:{port}): {message}")

    def _handle_check_success(self, serial: str, package: str, status: tuple[bool, bool]) -> None:
        if self._shutdown:
            return
        installed, running = status
        logger.info(
            "App status serial={} package={} installed={} running={}",
            serial,
            package,
            installed,
            running,
        )
        self.app_status_ready.emit(serial, package, installed, running)

    def _handle_check_error(self, message: str, serial: str, package: str) -> None:
        if self._shutdown:
            return
        logger.error(
            "App check failed serial={} package={} message={}",
            serial,
            package,
            message,
        )
        self.error_occurred.emit(f"Game check failed ({package}): {message}")

    def _handle_screenshot_success(self, serial: str, data: bytes) -> None:
        if self._shutdown:
            return
        logger.info("Screenshot captured serial={} size={} bytes", serial, len(data))
        self.screenshot_captured.emit(serial, data)

    def _handle_screenshot_error(self, message: str, serial: str) -> None:
        if self._shutdown:
            return
        logger.error("Screenshot failed serial={} message={}", serial, message)
        self.error_occurred.emit(f"Screenshot failed ({serial}): {message}")

    def _handle_tap_success(self, serial: str, x: int, y: int) -> None:
        if self._shutdown:
            return
        logger.debug("Tap sent serial={} x={} y={}", serial, x, y)
        self.tap_sent.emit(serial, x, y)

    def _handle_swipe_success(
        self,
        serial: str,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        duration_ms: int,
    ) -> None:
        if self._shutdown:
            return
        logger.debug(
            "Swipe sent serial={} x1={} y1={} x2={} y2={} duration={}",
            serial,
            x1,
            y1,
            x2,
            y2,
            duration_ms,
        )
        self.swipe_sent.emit(serial, x1, y1, x2, y2, duration_ms)


class _RefreshDevicesTask(QRunnable):
    def __init__(
        self,
        client: AdbClient,
        on_success,
        on_error,
    ) -> None:
        super().__init__()
        self._client = client
        self._on_success = on_success
        self._on_error = on_error

    def run(self) -> None:  # type: ignore[override]
        try:
            devices = self._client.list_devices()
            self._on_success(devices)
        except Exception as exc:  # noqa: BLE001 - surface error to UI
            self._on_error(str(exc))


class _ConnectDeviceTask(QRunnable):
    def __init__(
        self,
        client: AdbClient,
        host: str,
        port: int,
        on_success,
        on_error,
    ) -> None:
        super().__init__()
        self._client = client
        self._host = host
        self._port = port
        self._on_success = on_success
        self._on_error = on_error

    def run(self) -> None:  # type: ignore[override]
        try:
            serial = self._client.connect(host=self._host, port=self._port)
            self._on_success(serial)
        except Exception as exc:  # noqa: BLE001
            self._on_error(str(exc))


class _CheckAppTask(QRunnable):
    def __init__(
        self,
        client: AdbClient,
        serial: str,
        package: str,
        on_success,
        on_error,
    ) -> None:
        super().__init__()
        self._client = client
        self._serial = serial
        self._package = package
        self._on_success = on_success
        self._on_error = on_error

    def run(self) -> None:  # type: ignore[override]
        try:
            status = self._client.get_package_status(serial=self._serial, package=self._package)
            self._on_success(status)
        except Exception as exc:  # noqa: BLE001
            self._on_error(str(exc))


class _ScreenshotTask(QRunnable):
    def __init__(
        self,
        client: AdbClient,
        serial: str,
        on_success,
        on_error,
    ) -> None:
        super().__init__()
        self._client = client
        self._serial = serial
        self._on_success = on_success
        self._on_error = on_error

    def run(self) -> None:  # type: ignore[override]
        try:
            data = self._client.capture_screenshot(serial=self._serial)
            self._on_success(data)
        except Exception as exc:  # noqa: BLE001
            self._on_error(str(exc))


class _TapTask(QRunnable):
    def __init__(
        self,
        client: AdbClient,
        serial: str,
        x: int,
        y: int,
        on_success,
        on_error,
    ) -> None:
        super().__init__()
        self._client = client
        self._serial = serial
        self._x = x
        self._y = y
        self._on_success = on_success
        self._on_error = on_error

    def run(self) -> None:  # type: ignore[override]
        try:
            self._client.input_tap(serial=self._serial, x=self._x, y=self._y)
            self._on_success()
        except Exception as exc:  # noqa: BLE001
            self._on_error(str(exc))


class _SwipeTask(QRunnable):
    def __init__(
        self,
        client: AdbClient,
        serial: str,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        duration_ms: int,
        on_success,
        on_error,
    ) -> None:
        super().__init__()
        self._client = client
        self._serial = serial
        self._x1 = x1
        self._y1 = y1
        self._x2 = x2
        self._y2 = y2
        self._duration_ms = duration_ms
        self._on_success = on_success
        self._on_error = on_error

    def run(self) -> None:  # type: ignore[override]
        try:
            self._client.input_swipe(
                serial=self._serial,
                x1=self._x1,
                y1=self._y1,
                x2=self._x2,
                y2=self._y2,
                duration_ms=self._duration_ms,
            )
            self._on_success()
        except Exception as exc:  # noqa: BLE001
            self._on_error(str(exc))


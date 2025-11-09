from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional
import io

import adbutils
from loguru import logger


@dataclass(slots=True, frozen=True)
class DeviceInfo:
    """Lightweight snapshot of an ADB device."""

    serial: str
    state: str
    model: str | None
    product: str | None
    transport_id: Optional[int]
    emulator_port: Optional[int]
    display_name: str
    is_emulator: bool
    alias: Optional[str] = None


class AdbClient:
    """Wrapper around adbutils that supports custom adb executables."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5037,
        adb_path: Optional[str | Path] = None,
    ) -> None:
        self._host = host
        self._port = port
        self._adb_path = Path(adb_path).expanduser().resolve() if adb_path else None
        self._client = adbutils.AdbClient(host=self._host, port=self._port)

    def ensure_server(self) -> None:
        """Start the ADB server if needed."""
        if self._server_is_available():
            return
        self._start_server_process()
        if not self._server_is_available():
            raise RuntimeError("Unable to communicate with the ADB server.")

    def _start_server_process(self) -> None:
        if self._adb_path and self._adb_path.exists():
            command = [str(self._adb_path), "start-server"]
        else:
            command = ["adb", "start-server"]

        logger.debug("Starting ADB server: {}", command)
        try:
            subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Unable to start ADB server; adb executable not found."
            ) from exc
        except subprocess.CalledProcessError as exc:
            logger.error("ADB start-server failed: {}", exc.stderr)
            raise RuntimeError("Failed to start ADB server") from exc

        # Refresh client connection after starting the server.
        self._client = adbutils.AdbClient(host=self._host, port=self._port)

    def _server_is_available(self) -> bool:
        try:
            if hasattr(self._client, "version"):
                self._client.version()  # type: ignore[operator]
            else:
                next(self._client.iter_device(), None)
            return True
        except (adbutils.errors.AdbError, ConnectionRefusedError):
            return False

    def iter_devices(self) -> Iterable[DeviceInfo]:
        """Yield DeviceInfo entries for each connected device."""
        self.ensure_server()
        try:
            for device in self._client.iter_device():
                yield self._build_device_info(device)
        except adbutils.errors.AdbError as exc:
            raise RuntimeError(f"ADB query failed: {exc}") from exc

    def list_devices(self) -> List[DeviceInfo]:
        return list(self.iter_devices())

    def connect(self, host: str, port: int) -> str:
        """
        Connect to a remote ADB device.

        Returns
        -------
        serial:
            The serial identifier returned by the ADB server (typically host:port).
        """
        self.ensure_server()
        try:
            device = self._client.connect(host=host, port=port)
        except adbutils.errors.AdbError as exc:
            raise RuntimeError(f"ADB connect failed: {exc}") from exc
        return device.serial

    def _get_device(self, serial: str) -> adbutils.AdbDevice:
        self.ensure_server()
        try:
            return self._client.device(serial=serial)
        except adbutils.errors.AdbError as exc:
            raise RuntimeError(f"Unable to get device '{serial}': {exc}") from exc

    def is_package_running(self, serial: str, package: str) -> bool:
        """Return True if the package has a running process on the device."""
        device = self._get_device(serial)

        cmds = [
            f"pidof {package}",
            f"ps -A | grep {package}",
        ]

        for command in cmds:
            try:
                output = device.shell(command)
            except adbutils.errors.AdbError:
                continue
            if not output:
                continue
            for line in output.splitlines():
                line = line.strip()
                if not line:
                    continue
                if "grep" in line and package in line and line.startswith("grep"):
                    continue
                if package in line:
                    return True

        return False

    def is_package_installed(self, serial: str, package: str) -> bool:
        """Return True if the package is installed on the device."""
        device = self._get_device(serial)

        commands = [
            f"pm path {package}",
            f"cmd package list packages {package}",
        ]

        for command in commands:
            try:
                output = device.shell(command)
            except adbutils.errors.AdbError:
                continue
            if not output:
                continue
            if package in output:
                return True

        return False

    def get_package_status(self, serial: str, package: str) -> tuple[bool, bool]:
        """
        Return (installed, running) tuple for the given package.

        Both checks are best-effort; failures raise RuntimeError for the caller to handle.
        """
        installed = self.is_package_installed(serial=serial, package=package)
        running = False
        if installed:
            running = self.is_package_running(serial=serial, package=package)
        return installed, running

    def capture_screenshot(self, serial: str) -> bytes:
        """Capture a screenshot from the device and return PNG bytes."""
        device = self._get_device(serial)
        try:
            image = device.screenshot()
        except adbutils.errors.AdbError as exc:
            raise RuntimeError(f"Screenshot failed for '{serial}': {exc}") from exc

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    @staticmethod
    def _build_device_info(device: adbutils.AdbDevice) -> DeviceInfo:
        props = device.prop
        model = props.get("ro.product.model")
        product = props.get("ro.product.name")
        tid = getattr(device, "transport_id", None)
        state = _resolve_device_state(device)

        emulator_port = _extract_emulator_port(device.serial)
        is_emulator = emulator_port is not None
        display_name = _build_display_name(serial=device.serial, model=model, product=product)

        logger.debug(
            "Discovered device serial={} model={} state={} emulator={}",
            device.serial,
            model,
            state,
            is_emulator,
        )

        return DeviceInfo(
            serial=device.serial,
            state=state,
            model=model,
            product=product,
            transport_id=tid,
            emulator_port=emulator_port,
            display_name=display_name,
            is_emulator=is_emulator,
        )


def _extract_emulator_port(serial: str) -> Optional[int]:
    """
    Attempt to parse the emulator port from the device serial.

    Examples:
        emulator-5554 -> 5554
        127.0.0.1:62001 -> 62001
    """
    if serial.startswith("emulator-"):
        _, port_str = serial.split("-", maxsplit=1)
        return int(port_str)

    if ":" in serial:
        _, port_str = serial.rsplit(":", maxsplit=1)
        try:
            return int(port_str)
        except ValueError:
            return None

    return None


def _build_display_name(serial: str, model: Optional[str], product: Optional[str]) -> str:
    parts = [model or product, serial]
    return " • ".join(part for part in parts if part)


def _resolve_device_state(device: adbutils.AdbDevice) -> str:
    raw_state = getattr(device, "state", None)
    if callable(raw_state):
        try:
            raw_state = raw_state()
        except TypeError:
            raw_state = None

    if raw_state is None:
        getter = getattr(device, "get_state", None)
        if callable(getter):
            try:
                raw_state = getter()
            except Exception:  # noqa: BLE001
                raw_state = None

    return str(raw_state) if raw_state else "unknown"


# Universal Game Bot — Device Manager MVP

Simple cross-emulator device manager built with Python and Qt. The app enumerates Android emulators (BlueStacks, LDPlayer, MEmu, MuMu, etc.) exposed via ADB and lets you inspect them from a desktop GUI.

## Features

- Lists all ADB-connected devices with name, serial, state, and emulator/physical flag.
- Async refresh to avoid blocking the UI when ADB queries run.
- One-click shell launcher per device (opens a new terminal window with `adb shell`).
- Status log pane for quick diagnostics.
- Manual connect dialog to add remote ADB endpoints by name/IP/port.
- Game registry to store title/package pairs and check if the selected game is running on a device.
- Per-device game status column showing install/run state for the selected title.

## Prerequisites

- Python 3.10+
- `adb` on your system `PATH` (from Android Platform Tools or emulator bundle)
- Windows PowerShell, macOS Terminal, or a Linux terminal emulator (used to spawn shells)

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
source .venv/bin/activate         # macOS/Linux

pip install -r requirements.txt
```

## Run

```bash
python app.py
```

Pass `--debug` to enable verbose logging:

```bash
python app.py --debug
```

## Connecting emulators

Ensure your emulator exposes its ADB bridge (often under settings like *ADB debugging* or *Bridge connection*). If the emulator bundles its own `adb.exe`, add that directory to `PATH` or pass the explicit path when wiring `DeviceManager(adb_path=...)`.

Use **Add Device** in the toolbar to connect to an emulator reachable over TCP/IP (e.g. `127.0.0.1:5555`). Provide a friendly name so it appears in the device table.

## Checking a game

1. Click **Add Game** and provide the game title plus its Android package (e.g. `com.supercell.clashofclans`).
2. (Optional) Select a device in the table or leave nothing selected to check all devices.
3. Press **Check Game** to verify whether the package is installed and running on the target device(s). Results show up in the log panel, status bar, and the game status column for each device.

Added games are saved to `games.json` in the application directory and reloaded automatically on startup.
If at least one game is saved, the first game is checked automatically the next time you launch the app once devices are detected.

Use **Delete Game** to remove the currently selected title from the registry.

When you pick a different game from the dropdown, the app immediately re-checks its running state on the connected devices.

## Control Layout Designer

The **Control Layout** tab lets you map on-screen UI regions for automation:

1. Pick a game in the Devices tab (and optionally select a device if you plan to capture a fresh screenshot), then open the **Control Layout** tab and choose a category (`Controls`, `State Boards`, or `Dialogues`).
2. Click **Capture Screenshot** to grab the current device screen, then drag on the canvas to draw a rectangle; provide a friendly control name when prompted. Regions are tagged with the active category.
3. Manage the list of control regions per category (rename/delete) from the panel on the right.
4. Each region is stored with pixel coordinates relative to the captured screenshot for later automation logic.

Regions are saved to `layouts.json` and reloaded automatically for each game.

When a game is selected, the layout designer automatically targets the device currently running it (or the first available device) when you capture screenshots.

## Next steps

- Capture device screenshots and pipe them through EasyOCR for on-screen recognition.
- Add per-device automation scripts (record/playback, macro controls).
- Display richer metadata (screen size, battery, CPU, running app).
- Integrate task scheduling to run automation on multiple emulators in sequence.
- Package the GUI with PyInstaller for distribution to users without Python.


# iRobMan – MuJoCo Workspace

This repository now focuses on a single MuJoCo-based manipulation stack. It provides a light, Mac-friendly simulation environment with configurable cameras, objects, and robot motion helpers.

![iRobMan](imgs/vid.gif)
---

## 1. Prerequisites

1. **Python** == `3.11`.
2. Install the MuJoCo runtime and Python package. Easiest paths:
    - With uv (fast, recommended)
    - With pip (if uv does not work well on your device)
3. (Optional, GUI issues) Set `MUJOCO_GL=egl` or `MUJOCO_GL=osmesa` when rendering headless or on macOS with multiple GL stacks.
4. Clone the repository with all submodules.
---

## 2. Repository Layout

```shell
.
├── configs
│   └── test_config_mj.yaml # configuration file for Mujoco simulation
├── docs
│   ├── Actions-control-guide.md # control guide
│   ├── General-information.md # general guide
│   └── Sensors-data-guide.md # sensor guide
├── imgs
│   └── vid.gif
├── main.py
├── pyproject.toml # uv project file
├── README.md # main readme
├── requirements.txt
├── src
│   └── mujoco_app
│       ├── __init__.py
│       ├── mj_robot.py # robot class
│       ├── mj_simulation.py # simulation class
│       ├── py.typed
│       ├── scene_builder.py # scene builder class
│       ├── scene.py # scene class
│       └── transformations.py # useful transformations
├── uv.lock
├── view_object_with_sensors.py # sensor viewer
├── view_with_sensors_windows.sh # helper shell script for Windows
└── view_with_sensors.sh # helper shell script for Linux/macOS
```

---

## 3. Quick Start

### Using uv (recommended) – Cross-Platform

UV automatically handles platform-specific dependencies for Windows, macOS, and Linux.

#### Installation UV

Follow the instructions for your appropriate platform:

**Windows PowerShell:**
```powershell
# First time only: Allow running the official installer script
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser -Force

# Install UV
irm https://astral.sh/uv/install.ps1 | iex
```

**macOS / Linux (Bash/Zsh):**
```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### Installation Simulation (uv)

```bash
# From the repo root (works on all platforms)
uv sync

# Install with optional dev dependencies
uv sync --extra dev

# To activate the virtual environment
source .venv/bin/activate

# To deactivate the virtual environment
deactivate
```
`uv sync` also installs the local GIGA inference runtime used by `src/mujoco_app/grasp.py`.

**Platform-Specific Notes:**
- **Windows**: UV installs to `%USERPROFILE%\.local\bin`. Restart PowerShell if `uv` isn't found.
- **macOS**: UV installs to `~/.local/bin`. You may need to add it to PATH in `~/.zshrc` or `~/.bash_profile`.
- **Linux**: UV installs to `~/.local/bin`. Add to PATH if needed: `export PATH="$HOME/.local/bin:$PATH"`

**VS Code Integration:**
- Use the tasks: "uv: run app" or "uv: run module (fallback)"
- VS Code will auto-detect the `.venv` created by UV

### Using pip

```bash
# Works on all platforms
# Doing the steps in the virtual environment is recommended
pip install -r requirements.txt
pip install -e .
```

Both flows will:
- build the MuJoCo scene described in `mujoco_app/configs/test_config_mj.yaml`;
- step the simulation for a short warm-up;
- save RGB/depth/segmentation/projection/view matrices under `mujoco_app/`;
- optionally open the passive MuJoCo viewer if `gui: true` in the config.

Use `--no-gui-wait` to exit immediately instead of idling in the viewer.

---

## 4. Viewing Objects with Sensor Logging

The `view_object_with_sensors.py` script allows you to interactively view any YCB object in the MuJoCo viewer while optionally logging sensor data (camera images, joint states, end-effector poses, contacts).

### Basic Usage

**macOS / Linux:**
For **macOS** if you want to run with gui turned replace `python` with `mjpython`. 
```bash
# if you are using a virtual environment activate it first
# View an object without logging
python view_object_with_sensors.py YcbBanana --enable_gui

# Enable sensor logging
python view_object_with_sensors.py YcbHammer --enable_gui --save-sensors

# Save sensor data + RGB/depth images
python view_object_with_sensors.py YcbPear --enable_gui --save-sensors --save-images

# Custom logging interval (log every N steps)
python view_object_with_sensors.py YcbStrawberry --enable_gui --save-sensors --log-interval 25
```

**Windows PowerShell:**
```powershell
# View an object without logging
python view_object_with_sensors.py YcbBanana

# Enable sensor logging
python view_object_with_sensors.py YcbHammer --save-sensors

# Save sensor data + RGB/depth images
python view_object_with_sensors.py YcbPear --save-sensors --save-images

# Custom logging interval (log every N steps)
python view_object_with_sensors.py YcbStrawberry --save-sensors --log-interval 25
```
---

## 4. Detailed Documentation:

- General Information: [General-information.md](docs/General-information.md)
- How to use sensor data: [Sensors-data-guide.md](docs/Sensors-data-guide.md)
- How to control the robot: [Actions-control-guide.md](docs/Actions-control-guide.md)
- Task information: [Task-information.md](docs/Task-information.md)

Happy experimenting!

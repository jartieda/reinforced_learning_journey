# Example 09: Asset Zoo — Popular Robots & Objects in IsaacLab

## Goal

Stop writing environments and **learn what you can put inside them**.

IsaacLab ships `isaaclab_assets`, a catalogue of pre-tuned robot configs ready
to drop into any `InteractiveSceneCfg`.  Isaac Sim adds a second layer: a
library of USD files (mobile robots, environments, props) that do not have
Python configs but can be loaded directly via `UsdFileCfg`.

Understanding both layers lets you swap robots in one line and choose the right
scene for any task.

---

## Files

| File | What it contains |
|---|---|
| `README.md` | This catalogue |
| `browse.py` | Spawn any catalogued robot, run physics, print joint info |
| `record.py` | Record an MP4 from a camera-observation Isaac Lab environment |

---

## A — Manipulation Arms (`isaaclab_assets`)

These robots have full `ArticulationCfg` objects: joint limits, actuator
models (impedance / effort), and default poses are all pre-configured.

| Robot | Config symbol | Import | DOFs | Notes |
|---|---|---|---|---|
| Franka Panda | `FRANKA_PANDA_CFG` | `isaaclab_assets.robots.franka` | 9 (7+2 fingers) | Default arm for grasping tasks |
| Franka Panda (stiff PD) | `FRANKA_PANDA_HIGH_PD_CFG` | `isaaclab_assets.robots.franka` | 9 | Stiffer gains, more responsive |
| Universal Robots UR3 | `UR3_CFG` | `isaaclab_assets.robots.universal_robots` | 6 | Compact bench-top arm |
| Universal Robots UR5 | `UR5_CFG` | `isaaclab_assets.robots.universal_robots` | 6 | Mid-reach industrial arm |
| Universal Robots UR10 | `UR10_CFG` | `isaaclab_assets.robots.universal_robots` | 6 | Long-reach industrial arm |
| Universal Robots UR10e | `UR10e_CFG` | `isaaclab_assets.robots.universal_robots` | 6 | e-Series with force/torque sensor |
| Kinova Jaco 2 (6-DOF) | `KINOVA_JACO2_N6S300_CFG` | `isaaclab_assets.robots.kinova` | 8 (6+2) | Lightweight, assistive robotics |
| Kinova Jaco 2 (7-DOF) | `KINOVA_JACO2_N7S300_CFG` | `isaaclab_assets.robots.kinova` | 9 (7+2) | Extra wrist DOF for dexterity |

```python
# Minimal scene swap: replace Franka with UR10
from isaaclab_assets.robots.universal_robots import UR10_CFG

@configclass
class MySceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = UR10_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
    )
```

> **Action space changes automatically** — `num_actions` in your task config
> must match the number of controlled joints (e.g. 6 for any UR arm).

---

## B — Legged Robots (`isaaclab_assets`)

Quadrupeds and humanoids.  All use effort/position actuators pre-matched to
real hardware specs.

| Robot | Config symbol | Import | DOFs | Notes |
|---|---|---|---|---|
| ANYmal B | `ANYMAL_B_CFG` | `isaaclab_assets.robots.anymal` | 12 | Original ANYmal research platform |
| ANYmal C | `ANYMAL_C_CFG` | `isaaclab_assets.robots.anymal` | 12 | Commercial version, CEA actuators |
| ANYmal D | `ANYMAL_D_CFG` | `isaaclab_assets.robots.anymal` | 12 | Latest generation |
| Unitree A1 | `UNITREE_A1_CFG` | `isaaclab_assets.robots.unitree` | 12 | Fast, agile quadruped |
| Unitree Go1 | `UNITREE_GO1_CFG` | `isaaclab_assets.robots.unitree` | 12 | Consumer quadruped |
| Unitree Go2 | `UNITREE_GO2_CFG` | `isaaclab_assets.robots.unitree` | 12 | Improved Go1, common in research |
| Unitree H1 | `H1_CFG` | `isaaclab_assets.robots.unitree` | 19 | Full humanoid (arms + legs) |
| Unitree H1 (minimal) | `H1_MINIMAL_CFG` | `isaaclab_assets.robots.unitree` | 10 | Legs only, simpler locomotion |
| Unitree G1 | `G1_CFG` | `isaaclab_assets.robots.unitree` | 37 | Full humanoid with dexterous hands |
| Unitree G1 (minimal) | `G1_MINIMAL_CFG` | `isaaclab_assets.robots.unitree` | 21 | Legs + simple arms |

```python
# Locomotion task: drop in a quadruped
from isaaclab_assets.robots.anymal import ANYMAL_C_CFG

@configclass
class QuadrupedSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = ANYMAL_C_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
    )
```

---

## C — Classic RL Benchmarks (`isaaclab_assets`)

| Robot | Config symbol | Import | DOFs | Notes |
|---|---|---|---|---|
| OpenAI Humanoid | `HUMANOID_CFG` | `isaaclab_assets.robots.humanoid` | 17 | Classic MuJoCo humanoid in PhysX |
| Cartpole | `CARTPOLE_CFG` | `isaaclab_assets.robots.cartpole` | 1 | Cart-on-rail, balance benchmark |

---

## D — Isaac Sim USD Assets (Mobile Robots)

These robots ship as raw USD files inside the Isaac Sim asset root.  They have
no pre-built `ArticulationCfg`, so you load them with `UsdFileCfg` and supply
your own actuator config.

The asset root is defined by the `persistent.isaac.asset_root.default` setting
(set automatically when you install Isaac Sim or run `isaacsim --download-assets`).
All paths below are relative to that root under `Isaac/Robots/`.

| Robot | USD subpath | Type | Notes |
|---|---|---|---|
| Nova Carter | `Carter/nova_carter.usd` | Holonomic AMR | Full sensor suite (lidar, cameras) |
| Carter V1 | `Carter/carter_v1.usd` | Differential AMR | Lightweight NVIDIA reference platform |
| Turtlebot3 Burger | `Turtlebot/Turtlebot3/turtlebot3_burger.usd` | Differential | Classic ROS benchmark |
| Turtlebot3 Waffle | `Turtlebot/Turtlebot3/turtlebot3_waffle.usd` | Differential | Camera-equipped variant |
| iRobot Create 3 | `iRobot/Create3/create_3.usd` | Differential | Floor cleaning; simple navigation |
| Clearpath Jackal | `Clearpath/Jackal/jackal.usd` | Differential UGV | Outdoor terrain platform |
| Clearpath Husky | `Clearpath/Husky/husky.usd` | Differential UGV | Large outdoor UGV, heavy payload |
| Forklift C | `IsaacSim/ForkliftC/forklift_c.usd` | Industrial vehicle | Warehouse logistics tasks |
| Idealworks iw.hub | `Idealworks/iwhub/iw_hub.usd` | Holonomic AMR | Omni-directional warehouse robot |
| JetBot | `NVIDIA/Robomaker/aws_robomaker_jetbot.usd` | Differential | Compact NVIDIA education platform |
| Fraunhofer Evobot | `Fraunhofer/Evobot/evobot.usd` | Differential | European research platform |
| AgilexRobotics Limo | `AgilexRobotics/limo/limo.usd` | Multi-mode | Switches between Ackermann/diff/mecanum |

```python
# Load a mobile robot from a USD path
from isaaclab.assets import ArticulationCfg
from isaaclab.sim.spawners.from_files import UsdFileCfg
import isaaclab.sim as sim_utils

JACKAL_USD = "{ISAAC_ASSETS_PATH}/Isaac/Robots/Clearpath/Jackal/jackal.usd"

jackal = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Jackal",
    spawn=UsdFileCfg(usd_path=JACKAL_USD),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.1)),
    actuators={},  # add your own effort/velocity actuator dicts here
)
```

> `{ISAAC_ASSETS_PATH}` is resolved by Isaac Sim at runtime from the asset
> root setting.  See `isaaclab.sim.spawners.from_files.UsdFileCfg` for the
> full token list.

---

## E — Environment Scenes (USD)

Drop-in environment USDs.  Loaded via `AssetBaseCfg` + `UsdFileCfg` at the
top of your `InteractiveSceneCfg`.  All paths are relative to the Isaac Sim
asset root under `Isaac/Environments/`.

| Scene | USD filename | Size | Good for |
|---|---|---|---|
| Simple Room | `Simple_Room/simple_room.usd` | ~5 × 5 m | Manipulation, grasping |
| Simple Grid | `Grid/default_environment.usd` | Flat plane | Locomotion baselines |
| Warehouse (small, 1 shelf) | `Simple_Warehouse/warehouse.usd` | ~20 × 20 m | Navigation, pick-and-place |
| Warehouse (small, forklifts) | `Simple_Warehouse/warehouse_with_forklifts.usd` | ~20 × 20 m | Dynamic obstacle avoidance |
| Warehouse (multi-shelf) | `Simple_Warehouse/warehouse_multiple_shelves.usd` | ~30 × 30 m | Shelving navigation |
| Full Warehouse | `Simple_Warehouse/full_warehouse.usd` | ~60 × 60 m | Large-scale AMR |
| Hospital | `Hospital/hospital.usd` | Multi-room | Service robots, corridor nav |
| Office | `Office/office.usd` | Open-plan | Indoor navigation |
| JetRacer Track | `Jetracer/jetracer_track_solid.usd` | Oval track | Racing/velocity control |

```python
# Add a warehouse backdrop to your scene config
from isaaclab.assets import AssetBaseCfg
from isaaclab.sim.spawners.from_files import UsdFileCfg

@configclass
class WarehouseSceneCfg(InteractiveSceneCfg):
    env_background: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/Warehouse",
        spawn=UsdFileCfg(
            usd_path="{ISAAC_ASSETS_PATH}/Isaac/Environments/"
                     "Simple_Warehouse/warehouse.usd",
        ),
    )
    robot: ArticulationCfg = FRANKA_PANDA_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
    )
```

---

## F — Rigid Object Props

Small objects for manipulation / pick-and-place.  Loaded as `RigidObjectCfg`
(physics-enabled, no joints) from `isaaclab_assets.objects` or raw USD.

| Object | Source | Notes |
|---|---|---|
| `CUBOID_CFG` | `isaaclab.sim.spawners.shapes` | Parametric box, cheapest |
| `SPHERE_CFG` | `isaaclab.sim.spawners.shapes` | Parametric sphere |
| YCB objects (banana, mug…) | `{ISAAC_ASSETS_PATH}/Isaac/Props/YCB/` | Textured real-world scans |
| KIT objects | `{ISAAC_ASSETS_PATH}/Isaac/Props/KIT/` | Household items |
| Blocks / cubes | `{ISAAC_ASSETS_PATH}/Isaac/Props/Blocks/` | Simple coloured cubes |

```python
from isaaclab.assets import RigidObjectCfg
from isaaclab.sim.spawners.from_files import UsdFileCfg

cube = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Cube",
    spawn=UsdFileCfg(
        usd_path="{ISAAC_ASSETS_PATH}/Isaac/Props/Blocks/DexCube/dex_cube_instanceable.usd",
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0.0, 0.3)),
)
```

---

## G — Sensors

Sensors are attached to a robot prim and configured as fields in the scene.

| Sensor | Config class | Import | Notes |
|---|---|---|---|
| Camera (RGB / depth) | `CameraCfg` | `isaaclab.sensors` | Configurable FOV, resolution |
| Ray-cast lidar | `RayCasterCfg` | `isaaclab.sensors` | 2-D or 3-D scan patterns |
| Contact sensor | `ContactSensorCfg` | `isaaclab.sensors` | Foot contact for locomotion |
| IMU | `ImuCfg` | `isaaclab.sensors` | Acceleration + angular rate |
| Frame transformer | `FrameTransformerCfg` | `isaaclab.sensors` | Track any prim's pose |

```python
from isaaclab.sensors import CameraCfg, ContactSensorCfg
import isaaclab.sim as sim_utils

camera = CameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/panda_hand/cam",
    update_period=0.1,
    height=84, width=84,
    data_types=["rgb", "depth"],
    spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, horizontal_aperture=20.955),
)

contact = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/.*_foot",  # regex over all foot links
    update_period=0.0,
    history_length=3,
)
```

---

## Swap-Robot Cheat Sheet

The three numbers that must agree whenever you change the robot:

```
robot DOFs  ──►  num_actions in task config  ──►  policy output_size
```

| Robot | `num_actions` | Observation addition |
|---|---|---|
| Franka Panda (arm only) | 7 | 7 joint pos + 7 joint vel |
| UR10 | 6 | 6 joint pos + 6 joint vel |
| ANYmal C | 12 | 12 joint pos + 12 joint vel + base IMU |
| Unitree H1 (minimal) | 10 | 10 joint pos + 10 joint vel + base IMU |
| Unitree G1 (minimal) | 21 | 21 joint pos + 21 joint vel + base IMU |

---

## browse.py — Quick Inspection Script

Run with:

```bash
# inside the Isaac Sim container or native install
python examples/09_asset_zoo/browse.py --robot franka
python examples/09_asset_zoo/browse.py --robot anymal_c
python examples/09_asset_zoo/browse.py --robot ur10
python examples/09_asset_zoo/browse.py --robot h1
python examples/09_asset_zoo/browse.py --robot g1_minimal
python examples/09_asset_zoo/browse.py --list
```

The script spawns the robot in a bare ground-plane scene, runs 200 physics
steps, and prints:
- All joint names
- Joint position limits [min, max]
- Default/home joint positions
- Number of controlled DOFs

No RL, no reward — pure asset inspection.

---

## H — Camera Setup and Video Recording

Use this 09 lesson to learn the full camera pipeline in Isaac Lab.

### 1) Enable camera rendering at app boot

Camera sensors do not produce images unless the app is launched with
`enable_cameras=True`:

```python
from isaaclab.app import AppLauncher

simulation_app = AppLauncher(
        {"headless": True, "enable_cameras": True, "livestream": 0}
).app
```

### 2) Add a camera to your scene config

Attach a camera to a robot link (eg. wrist) or place one in world space:

```python
import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg

camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_hand/cam",
        update_period=0.0,  # every physics step
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                horizontal_aperture=20.955,
        ),
)
```

### 3) Roll out simulation and write frames to MP4

`record.py` demonstrates the end-to-end process for camera-observation envs:
- boots AppLauncher with cameras enabled,
- steps the environment,
- converts observations to RGB `uint8`,
- writes MP4 via `imageio` (fallback to `cv2`, then PNG frames).

Run with:

```bash
python examples/09_asset_zoo/record.py \
    --env-id Isaac-Cartpole-Camera-Showcase-Box-Box-Direct-v0 \
    --steps 600 \
    --fps 30 \
    --output runs/09_asset_zoo/cartpole_cam.mp4
```

With a trained checkpoint:

```bash
python examples/09_asset_zoo/record.py \
    --env-id Isaac-Cartpole-Camera-Showcase-Box-Box-Direct-v0 \
    --checkpoint runs/07_isaac_transfer/07_isaac_transfer/checkpoints/agent_*.pt \
    --steps 1000 \
    --output runs/09_asset_zoo/cartpole_policy.mp4
```

Tip: If you prefer a live preview while recording, set `--livestream 2` and
open Isaac Sim WebRTC on port `8080`.

# CLAUDE.md — VantTec G1 teleop / sim / hardware

## Where you are

**Open Claude / Cursor workspace root here:**

```text
/home/emendezj/unitree_sim_isaaclab
```

Sibling repo (teleop + Quest), open only when editing teleop:

```text
/home/emendezj/xr_teleoperate
```

Both are **VantTec forks** (`origin` → `github.com/vanttec/...`). Do **not** push to `unitreerobotics/*`.

| Repo | Role |
|------|------|
| `unitree_sim_isaaclab` | Isaac Lab sim, DDS bridge, trajectories, deploy tooling |
| `xr_teleoperate` | Quest / hand teleop client (`teleop/teleop_hand_and_arm.py`) |

---

## What this project is

Student robotics stack (VANTTEC / Tec de Monterrey) around **Unitree G1 + dexterous hands**, built as the **robot execution backend for BCI (brain–computer interface) experiments**.

**Product framing:** a BCI (or any external classifier/app) does *not* talk DDS to the robot. It sends **small TCP intents**; this stack turns those intents into safe grasp / trajectory execution on sim or the physical G1.

```text
[BCI / ML / remote PC] --TCP--> grasp_bridge sockets --DDS--> G1 (+ hands)
                                      ^
                         also: Quest teleop, CLI, Isaac sim
```

| Layer | What |
|-------|------|
| **BCI front-end** | External (not in this repo). Classifies intent → TCP uint32 cmds |
| **TCP 5555** | Semantic grasps 1–6 (`grasp_bridge.socket_server`) |
| **TCP 5556** | Trajectory replay slots (`trajectory_socket_server` / `start_traj_socket.sh`) |
| **DDS** | Real-time robot/hand control (`rt/lowcmd`, Inspire hand topics) |
| **Teleop** | Quest path to *seed* / record trajectories the BCI later triggers |
| **Sim** | Isaac Lab pick-place (coin, stick, tennisball, cardsdeck, container) |

So: teleop + sim exist to **develop and record** behaviors; the **deploy target** is “BCI (or dummy client) → TCP → robot”, ideally **wireless** so the subject / laptop are not tethered by Ethernet.

Ops doc: `doc/RUNBOOK.md` (section “TCP — BCI u otra PC”).

---

## Hardware NOW (priority)

| Item | Detail |
|------|--------|
| Robot | **Physical Unitree G1 U7** (real hardware, not only sim) |
| Hands | **5-finger** dexterous hands (Inspire-class; treat as Inspire DFX pipeline in this codebase) |
| Goal #1 | **Deploy control / teleop / trajectories without Ethernet tether** |
| Constraint | Unitree DDS often defaults to wired NIC; we need **Wi‑Fi / wireless DDS** that is stable enough for lowcmd + hand cmds |

Everything else (sim polish, papers, ablations) is secondary until **wireless physical bring-up** works.

### Success criteria (wireless deploy)

- Laptop and G1 on the **same wireless network** (or documented Unitree Wi‑Fi mode).
- `source grasp_bridge/setup_local_dds.sh` binds a **Wi‑Fi** iface (`wlp…`), not `eth` / USB eth with no robot.
- `python -m grasp_bridge.dds_ping` sees robot state over the air.
- Can publish `rt/lowcmd` + hand topics and get safe motion **without** an Ethernet cable to the robot.
- Document the exact iface, domain id, and any CycloneDDS URI / Unitree app Wi‑Fi steps that work.

If Ethernet is required as a temporary debug path, treat it as a fallback and say so explicitly — do not leave “must use cable” as the only story.

---

## Key paths (this repo)

```text
sim_main.py                         # Isaac entry
scripts/run_inspire_teleop_env.sh   # sim env launcher (FAST=1 for light render)
grasp_bridge/                       # DDS client, traj record/play, CLI
grasp_bridge/setup_local_dds.sh     # sets UNITREE_DDS_INTERFACE
grasp_bridge/trajectory_registry.py # per-slot replay knobs (boost, freeze, offset)
grasp_bridge/trajectories/traj_N.npz
tasks/common_scene/                 # object scenes (e.g. cardsdeck)
doc/RUNBOOK.md
```

Teleop (other repo), always from `teleop/`:

```bash
conda activate tv
cd ~/xr_teleoperate/teleop
source ~/unitree_sim_isaaclab/grasp_bridge/setup_local_dds.sh
python teleop_hand_and_arm.py --sim --ee inspire_dfx --input-mode hand \
  --display-mode immersive --img-server-ip 127.0.0.1 --frequency 60
```

For **physical** robot, drop `--sim` only when DDS is confirmed on the real G1; keep safety limits.

---

## DDS / network rules

- Every terminal: `source grasp_bridge/setup_local_dds.sh` (or export `UNITREE_DDS_INTERFACE` yourself).
- Prefer **Wi‑Fi with LOWER_UP**; avoid docker/veth/br bridges.
- Domain / channel historically `1` in this stack (`ChannelFactoryInitialize(1)`).
- Topics of interest: `rt/lowstate`, `rt/lowcmd`, `rt/inspire/cmd`, `rt/inspire/state`, `rt/reset_pose/cmd`.
- Safety: physical G1 → home pose, soft limits, e-stop awareness; never blast untested high-gain cmds.

---

## Agent working rules

1. **Primary objective:** wireless physical G1 U7 deploy (no Ethernet dependency).
2. Prefer small, reversible changes; don’t rewrite Unitree SDK internals unless needed.
3. Keep sim and real paths aligned (same topics / hand encoding) so teleop and `trajectory_cli` work on both.
4. Don’t commit secrets, `__pycache__`, or submodule noise; push only to `vanttec` remotes.
5. Spanish or English OK; be direct. Physical hardware = assume real risk.

---

## Current status snapshot

- Sim + teleop + trajectory replay for Inspire pick-place: **working in sim**.
- Cardsdeck / slot 4 and replay knobs: tuned in `trajectory_registry.py` + scene cfg.
- Physical G1 U7: **available** — bring-up focus is **wireless DDS + deploy**, not more sim-only features.

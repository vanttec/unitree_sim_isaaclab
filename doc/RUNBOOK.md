# Runbook — G1 Inspire (VantTec)

Guía única para simulación, agarres semánticos (1–6), teleop, trayectorias y BCI/TCP.

## Requisitos

- Ubuntu 22.04, GPU NVIDIA, Conda `env_isaaclab` (Isaac Sim 5.0 + Isaac Lab)
- [unitree_sdk2_python](https://github.com/unitreerobotics/unitree_sdk2_python) + CycloneDDS (`CYCLONEDDS_HOME`)
- Assets: `./fetch_assets.sh` o descomprimir `unitree_sim_isaaclab_usds/assets.zip`

```bash
pip install -r requirements.txt pyyaml
PIP_IGNORE_REQUIRES_PYTHON=1 pip install -e "./teleimager[server]"
export CYCLONEDDS_HOME="$HOME/cyclonedds/install"
pip install -e ~/unitree_sdk2_python
```

Instalación Isaac Sim: [isaacsim5.0_install.md](isaacsim5.0_install.md)

---

## DDS (obligatorio en cada terminal)

```bash
cd ~/unitree_sim_isaaclab
source grasp_bridge/setup_local_dds.sh
# Debe imprimir UNITREE_DDS_INTERFACE=wlp... (WiFi con LOWER_UP, no USB sin cable)
```

Reinicia el sim si cambias de interfaz. Prueba:

```bash
python -m grasp_bridge.dds_ping
# OK — N+ messages received
```

---

## 1. Simulación base

```bash
conda activate env_isaaclab
source grasp_bridge/setup_local_dds.sh

python sim_main.py \
  --task Isaac-PickPlace-Cylinder-G129-Inspire-Joint \
  --enable_inspire_dds \
  --robot_type g129 \
  --enable_cameras
```

Esperar: `DDS network interface: ...` y `start controller success`.

### Entornos pick-place (teleop / trayectorias)

```bash
./scripts/run_inspire_teleop_env.sh coin        # slot 1
./scripts/run_inspire_teleop_env.sh stick       # slot 2
./scripts/run_inspire_teleop_env.sh tennisball  # slot 3
./scripts/run_inspire_teleop_env.sh cardsdeck   # slot 4
./scripts/run_inspire_teleop_env.sh container   # slot 5
./scripts/run_inspire_teleop_env.sh cylinder | redblock
```

Agarre firme al objeto (opcional):

```bash
python sim_main.py ... --enable_inspire_dds --enable_grasp_attach
```

---

## 2. Agarres semánticos (grasp 1–6)

| ID | Tipo |
|----|------|
| 1 | power |
| 2 | pinch |
| 3 | lateral |
| 4 | tripod |
| 5 | hook |
| 6 | precision |

```bash
source grasp_bridge/setup_local_dds.sh
python -m grasp_bridge.cli --list
python -m grasp_bridge.cli 3
python -m grasp_bridge.cli 3 --hz 80
```

Definiciones: `grasp_bridge/grasp_library.py`

### TCP — BCI u otra PC (puerto 5555)

**Servidor** (PC con sim):

```bash
source grasp_bridge/setup_local_dds.sh
python -m grasp_bridge.socket_server
```

**Cliente** (reemplaza `IP_SIM`):

```bash
python -c "import socket,struct;s=socket.create_connection(('IP_SIM',5555));s.send(struct.pack('<I',3));print(s.recv(4));s.close()"
```

Protocolo: 4 bytes `uint32` LE → respuesta `OK` / `ERROR`. Un agarre a la vez.

### ROS2 (opcional)

```bash
source /opt/ros/humble/setup.bash
python -m grasp_bridge.ros2_node
ros2 topic pub --once /grasp_command std_msgs/msg/Int32 "{data: 3}"
```

---

## 3. Tuning de agarres (GUI → Python)

1. Pausa el sim, posa brazo/mano en Property panel.
2. Copia ángulos a YAML:

```bash
cp grasp_bridge/poses/_template.yaml grasp_bridge/poses/power.yaml
# editar keyframes: approach / close / lift
python -m grasp_bridge.capture_pose --yaml grasp_bridge/poses/power.yaml
```

3. Pega la salida en `grasp_library.py` y prueba con `cli`.

Nombres de joints:

```bash
python -m grasp_bridge.capture_pose --joint-list
```

Snapshot en vivo (sim corriendo):

```bash
python -m grasp_bridge.capture_pose --live --name approach --duration 5
python -m grasp_bridge.capture_pose --live --name close --duration 3
python -m grasp_bridge.capture_pose --emit-session --grasp-id 1 --label power
```

Desde video/mocap (`trajectory_g1.npz`):

```bash
python -m grasp_bridge.from_g1_replay --traj traj.npz --grasp-id 7 --label demo \
  --keyframes "approach=0:5.0,close=45:3.0,lift=60:3.0"
```

---

## 4. Teleop + trayectorias (slots 1–5)

### Trayectorias incluidas en el repo

Tras `git clone`, ya vienen grabadas en `grasp_bridge/trajectories/`:

| Slot | Archivo | Objeto | Comando sim |
|------|---------|--------|-------------|
| 1 | `traj_1.npz` | coin | `./scripts/run_inspire_teleop_env.sh coin` |
| 2 | `traj_2.npz` | stick | `./scripts/run_inspire_teleop_env.sh stick` |
| 3 | `traj_3.npz` | tennisball | `./scripts/run_inspire_teleop_env.sh tennisball` |
| 4 | `traj_4.npz` | cardsdeck | `./scripts/run_inspire_teleop_env.sh cardsdeck` |
| 5 | `traj_5.npz` | container | `./scripts/run_inspire_teleop_env.sh container` |

**Regla:** el sim debe usar el **mismo objeto** que el slot. Si corres `play 3`, arranca `tennisball`, no `coin`.

#### Reproducir trayectorias (sin teleop, sin re-grabar)

**Terminal 1 — sim** (ejemplo slot 1 / coin):

```bash
conda activate env_isaaclab
cd ~/unitree_sim_isaaclab
source grasp_bridge/setup_local_dds.sh
./scripts/run_inspire_teleop_env.sh coin
```

Espera `start controller success`. No abras teleop.

**Terminal 2 — replay:**

```bash
conda activate env_isaaclab
cd ~/unitree_sim_isaaclab
source grasp_bridge/setup_local_dds.sh
python -m grasp_bridge.trajectory_cli list
python -m grasp_bridge.trajectory_cli play 1
```

Otros slots (mismo patrón: sim con el objeto correcto, luego `play N`):

```bash
./scripts/run_inspire_teleop_env.sh stick       && python -m grasp_bridge.trajectory_cli play 2
./scripts/run_inspire_teleop_env.sh tennisball  && python -m grasp_bridge.trajectory_cli play 3
./scripts/run_inspire_teleop_env.sh cardsdeck   && python -m grasp_bridge.trajectory_cli play 4
./scripts/run_inspire_teleop_env.sh container   && python -m grasp_bridge.trajectory_cli play 5
```

(`play` va en otra terminal, con el sim ya en marcha.)

---

### Sim + teleop (grabar nuevas trayectorias)

**Terminal 1:**

```bash
./scripts/run_inspire_teleop_env.sh coin
```

**Terminal 2** (`xr_teleoperate`):

```bash
conda activate tv
cd ~/xr_teleoperate/teleop
python teleop_hand_and_arm.py --sim --ee inspire_dfx --input-mode hand \
  --display-mode immersive --img-server-ip 127.0.0.1 --frequency 60
```

Pulsa **`r`** para tracking.

### Grabar

```bash
source grasp_bridge/setup_local_dds.sh
python -m grasp_bridge.trajectory_cli record --slot 1 --label coin_pick
# ENTER → mover → ENTER para guardar → grasp_bridge/trajectories/traj_1.npz
```

Desde episodio xr_teleoperate:

```bash
python -m grasp_bridge.trajectory_cli from-episode \
  ./utils/data/coin_demo/episode_0001/data.json --slot 1
```

### Reproducir (grabación propia o la del repo)

Teleop **detenido**. Mismo entorno que el slot (ver tabla arriba).

```bash
source grasp_bridge/setup_local_dds.sh
python -m grasp_bridge.trajectory_cli list
python -m grasp_bridge.trajectory_cli play 1
```

No ejecutar `cli` y `trajectory_cli play` a la vez (mismo DDS).

### TCP trayectorias (puerto 5556)

```bash
./scripts/start_traj_socket.sh
python -m grasp_bridge.trajectory_send --interactive
```

| TCP cmd | Slot | Objeto |
|---------|------|--------|
| 1 | 5 | container |
| 2 | 2 | stick |
| 4 | 3 | tennisball |
| 6 | 4 | cardsdeck |

### Reset objeto

```bash
python reset_pose_test.py 1   # solo objeto
python reset_pose_test.py 2   # escena completa
```

---

## 5. Referencia rápida

| Módulo | Comando |
|--------|---------|
| DDS ping | `python -m grasp_bridge.dds_ping` |
| Agarre CLI | `python -m grasp_bridge.cli <1-6>` |
| Agarre TCP | `python -m grasp_bridge.socket_server` |
| Trayectoria | `python -m grasp_bridge.trajectory_cli play <1-5>` |
| Trayectoria TCP | `./scripts/start_traj_socket.sh` |
| Pose → código | `python -m grasp_bridge.capture_pose --yaml ...` |

**Topics DDS:** `rt/lowstate`, `rt/lowcmd`, `rt/inspire/cmd`, `rt/inspire/state`

**Control:** brazos = posición (rad) + kp/kd; manos = normalizado 0=cerrado, 1=abierto.

---

## Problemas frecuentes

| Síntoma | Acción |
|---------|--------|
| `No rt/lowstate` | `source setup_local_dds.sh` en sim y bridge; reiniciar sim |
| `does not match an available interface` | USB Ethernet sin cable; re-source script (elige WiFi) |
| `'G1RobotDDS' has no attribute 'publisher'` | DDS mal inicializado; reiniciar sim con script correcto |
| Movimiento brusco | Subir `duration_s` en `grasp_library.py` o `--hz 80` |
| Dedos flojos en replay | Grabar a 100 Hz (`trajectory_cli record` sin `--hz 30`) |

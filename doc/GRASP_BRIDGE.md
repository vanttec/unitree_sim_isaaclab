# Grasp bridge — G1 Inspire

Guía para simular agarres con **Unitree G1** y manos **Inspire** en Isaac Sim.

## Requisitos

- Ubuntu 22.04, GPU NVIDIA (serie RTX 50 → Isaac Sim 5.0)
- Entorno Conda `env_isaaclab` (Isaac Sim 5.0 + Isaac Lab)
- [unitree_sdk2_python](https://github.com/unitreerobotics/unitree_sdk2_python) con CycloneDDS (`CYCLONEDDS_HOME`)
- `teleimager` instalado desde el submodule de **este** repositorio
- Assets: `./fetch_assets.sh` o descomprimir `unitree_sim_isaaclab_usds/assets.zip`

```bash
pip install -r requirements.txt
PIP_IGNORE_REQUIRES_PYTHON=1 pip install -e "./teleimager[server]"
export CYCLONEDDS_HOME="$HOME/cyclonedds/install"
pip install -e ~/unitree_sdk2_python
```

## Ejecución

En **cada terminal**, antes de sim o bridge:

```bash
source grasp_bridge/setup_local_dds.sh
```

### 1. Simulación

```bash
python sim_main.py \
  --task Isaac-PickPlace-Cylinder-G129-Inspire-Joint \
  --enable_inspire_dds \
  --robot_type g129 \
  --enable_cameras
```

Confirmar en log: `DDS network interface: ...` y `start controller success`.

### 2. Diagnóstico DDS

```bash
python -m grasp_bridge.dds_ping
```

Salida esperada: `OK — N+ messages received`.

### 3. Agarre

```bash
python -m grasp_bridge.cli --list
python -m grasp_bridge.cli 3
python -m grasp_bridge.cli 3 --hz 80
```

| ID | Tipo |
|----|------|
| 1 | power |
| 2 | pinch |
| 3 | lateral |
| 4 | tripod |
| 5 | hook |
| 6 | precision |

### 4. ROS2 (opcional)

```bash
source /opt/ros/humble/setup.bash
python -m grasp_bridge.ros2_node
```

```bash
ros2 topic pub --once /grasp_command std_msgs/msg/Int32 "{data: 3}"
ros2 topic echo /grasp_status
```

## Control (qué se envía)

**Brazos** (`rt/lowcmd`): posición objetivo por joint (rad), con gains `kp` / `kd`. No es PWM.

**Manos** (`rt/inspire/cmd`): cierre normalizado por motor (`0` = abierto, `1` = cerrado).

**DDS:** dominio `ChannelFactoryInitialize(1)` en sim y bridge.

| Topic | Dirección |
|-------|-----------|
| `rt/lowstate` | sim → bridge |
| `rt/lowcmd` | bridge → sim |
| `rt/inspire/cmd` | bridge → sim |

## Tuning

Archivo: `grasp_bridge/grasp_library.py`

| Parámetro | Descripción |
|-----------|-------------|
| `_APPROACH_LEFT`, `_LIFT_LEFT` | 7 joints del brazo izquierdo (rad) |
| `_hand_left(...)` | Mano izquierda, motores 6–11 (0–1) |
| `duration_s` | Duración de cada fase (s) |

Orden joints brazo: `shoulder_pitch`, `shoulder_roll`, `shoulder_yaw`, `elbow`, `wrist_roll`, `wrist_pitch`, `wrist_yaw`.

Orden mano izquierda: `pinky`, `ring`, `middle`, `index`, `thumb_p`, `thumb_y`.

No es necesario reiniciar el sim; guardar y volver a ejecutar `cli`.

## Módulo `grasp_bridge/`

| Archivo | Función |
|---------|---------|
| `grasp_library.py` | Definición de los 6 agarres |
| `executor.py` | Interpolación y publicación DDS |
| `dds_client.py` | Cliente `lowcmd` + Inspire |
| `cli.py` | Entrada sin ROS2 |
| `ros2_node.py` | `/grasp_command` (Int32 1–6) |
| `dds_ping.py` | Prueba de conectividad |
| `setup_local_dds.sh` | Interfaz de red multicast |

## Problemas frecuentes

| Síntoma | Acción |
|---------|--------|
| `No rt/lowstate` | Mismo `setup_local_dds.sh` en ambas terminales; reiniciar sim |
| `uvc` no encontrado | `pip install -e "./teleimager[server]"` desde este repo |
| `unitree_sdk2py` | Instalar SDK con `CYCLONEDDS_HOME` configurado |
| Movimiento brusco | Aumentar `duration_s` o usar `--hz 80` |

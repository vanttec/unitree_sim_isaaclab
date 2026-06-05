# Fork custom — Unitree G1 Inspire + grasp bridge

Documentación de **nuestros cambios** sobre [unitreerobotics/unitree_sim_isaaclab](https://github.com/unitreerobotics/unitree_sim_isaaclab).

El README oficial del upstream sigue en [README.md](README.md). Instalación base de Isaac Sim / Isaac Lab: [doc/isaacsim5.0_install.md](doc/isaacsim5.0_install.md).

---

## Qué agregamos

| Componente | Descripción |
|------------|-------------|
| `grasp_bridge/` | Módulo para ejecutar 6 tipos de agarre vía DDS (mismo protocolo que robot real) |
| `grasp_bridge/setup_local_dds.sh` | Configura interfaz de red para DDS en la misma PC |
| `dds/dds_master.py` | Respeta `UNITREE_DDS_INTERFACE` al iniciar DDS |

**Robot en sim:** G1 29-DOF + manos **Inspire** (5 dedos).  
**Control:** posiciones articulares (`rt/lowcmd`) + cierre normalizado de mano (`rt/inspire/cmd`). No es PWM.

---

## Requisitos previos (resumen)

- Ubuntu 22.04, GPU NVIDIA (RTX 50 → Isaac Sim 5.0)
- Conda env `env_isaaclab` con Isaac Sim 5.0 + Isaac Lab
- `unitree_sdk2_python` + CycloneDDS compilado (`CYCLONEDDS_HOME`)
- `teleimager` instalado desde el submodule de **este** repo (no el de `xr_teleoperate`)
- Assets del robot: `./fetch_assets.sh` o descomprimir `unitree_sim_isaaclab_usds/assets.zip`

```bash
# Dependencias Python del repo + bridge
pip install -r requirements.txt
PIP_IGNORE_REQUIRES_PYTHON=1 pip install -e "./teleimager[server]"
pip install -e ~/unitree_sdk2_python   # con CYCLONEDDS_HOME exportado
```

---

## Cómo correr (flujo habitual)

### Terminal 1 — Simulación

```bash
conda activate env_isaaclab
source ~/unitree_sim_isaaclab/grasp_bridge/setup_local_dds.sh
# Debe imprimir: UNITREE_DDS_INTERFACE=wlp131s0f0 (o tu interfaz)

cd ~/unitree_sim_isaaclab
python sim_main.py \
  --task Isaac-PickPlace-Cylinder-G129-Inspire-Joint \
  --enable_inspire_dds \
  --robot_type g129 \
  --enable_cameras
```

Esperar en log: `[DDSManager] DDS network interface: ...` y **`start controller success`**.

### Terminal 2 — Probar DDS

```bash
conda activate env_isaaclab
source ~/unitree_sim_isaaclab/grasp_bridge/setup_local_dds.sh
cd ~/unitree_sim_isaaclab
python -m grasp_bridge.dds_ping
```

Debe mostrar `OK — N+ messages received`.

### Terminal 3 — Ejecutar agarre

```bash
conda activate env_isaaclab
source ~/unitree_sim_isaaclab/grasp_bridge/setup_local_dds.sh
cd ~/unitree_sim_isaaclab
python -m grasp_bridge.cli --list
python -m grasp_bridge.cli 3          # lateral, brazo/mano izquierda
python -m grasp_bridge.cli 3 --hz 80  # más suave
```

### ROS2 (opcional)

```bash
source /opt/ros/humble/setup.bash
python -m grasp_bridge.ros2_node
# otra terminal:
ros2 topic pub --once /grasp_command std_msgs/msg/Int32 "{data: 3}"
```

---

## Tuning de agarres

Editar **`grasp_bridge/grasp_library.py`**:

- Brazo izquierdo: `_APPROACH_LEFT`, `_LIFT_LEFT` (7 joints en radianes)
- Mano izquierda: `_hand_left(index=..., thumb_p=...)` — valores 0.0 (abierto) a 1.0 (cerrado)
- Duración de fases: `duration_s` en cada `GraspKeyframe`

No hace falta reiniciar el sim; solo guardar el archivo y volver a correr `cli`.

---

## Topics DDS (sim = robot real)

| Topic | Uso |
|-------|-----|
| `rt/lowstate` | Estado del G1 (sim publica, bridge escucha) |
| `rt/lowcmd` | Comando de joints (bridge publica) |
| `rt/inspire/cmd` | Comando manos Inspire |
| `rt/inspire/state` | Estado manos |

Dominio DDS: **`ChannelFactoryInitialize(1)`** en sim y bridge.

---

## Estructura `grasp_bridge/`

```
grasp_bridge/
├── grasp_library.py    # 6 agarres + keyframes
├── executor.py         # Interpolación 50 Hz, lado activo
├── dds_client.py       # Publica/escucha DDS
├── cli.py              # Prueba sin ROS2
├── ros2_node.py        # /grasp_command Int32 1-6
├── dds_ping.py         # Diagnóstico de conexión
└── setup_local_dds.sh  # Exportar interfaz de red
```

---

## Problemas frecuentes

| Síntoma | Solución |
|---------|----------|
| `No rt/lowstate` | Mismo `setup_local_dds.sh` en **ambas** terminales; **reiniciar** sim después de `source` |
| `ModuleNotFoundError: uvc` | `pip install -e "./teleimager[server]"` desde este repo |
| `unitree_sdk2py` | `pip install -e ~/unitree_sdk2_python` con `CYCLONEDDS_HOME` |
| Mueve brazo derecho | Usar versión actual de `grasp_library` (`active_side="left"`) |
| Movimiento brusco | Subir `duration_s` o `--hz 80` |

---

## Créditos upstream

Basado en [unitreerobotics/unitree_sim_isaaclab](https://github.com/unitreerobotics/unitree_sim_isaaclab), Isaac Lab, y `unitree_sdk2_python`.

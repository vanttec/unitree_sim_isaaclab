# unitree_sim_isaaclab (VantTec)

Fork de [unitreerobotics/unitree_sim_isaaclab](https://github.com/unitreerobotics/unitree_sim_isaaclab) con soporte para **agarres del G1 + manos Inspire** vía DDS (mismo protocolo que el robot físico).

## Documentación

| Documento | Contenido |
|-----------|-----------|
| [doc/GRASP_BRIDGE.md](doc/GRASP_BRIDGE.md) | Instalación, comandos, tuning y troubleshooting |
| [doc/isaacsim5.0_install.md](doc/isaacsim5.0_install.md) | Isaac Sim 5.0 + Isaac Lab (upstream) |

## Inicio rápido

```bash
conda activate env_isaaclab
source grasp_bridge/setup_local_dds.sh

# Terminal 1 — simulación
python sim_main.py \
  --task Isaac-PickPlace-Cylinder-G129-Inspire-Joint \
  --enable_inspire_dds --robot_type g129 --enable_cameras

# Terminal 2 — agarre (sim ya en marcha)
python -m grasp_bridge.dds_ping
python -m grasp_bridge.cli 3
```

## Cambios principales

- `grasp_bridge/` — seis agarres (CLI y ROS2 opcional)
- `dds/dds_master.py` — interfaz DDS configurable (`UNITREE_DDS_INTERFACE`)

## Créditos

Unitree Robotics · Isaac Lab · [unitree_sdk2_python](https://github.com/unitreerobotics/unitree_sdk2_python)

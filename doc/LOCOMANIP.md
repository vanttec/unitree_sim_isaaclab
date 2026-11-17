# Loco-manipulación teleoperada en sim → datos para fine-tunear SONIC

Piernas: **GR00T-WholeBodyControl** (Balance/Walk de NVIDIA) dentro de la sim, comandada con el **teclado**.
Brazos y manos: **Quest** con `xr_teleoperate` (hand tracking, manos Inspire).
Cada episodio se graba con la verdad de la sim y se exporta al formato que usa SONIC.

## 1. Terminales

**T1 · Sim** (conda `env_isaaclab`)
```bash
cd ~/unitree_sim_isaaclab && conda activate env_isaaclab
./scripts/run_locomanip_sim.sh                  # con ventana de Isaac
HEADLESS=1 ./scripts/run_locomanip_sim.sh       # sin ventana: más rápido, el Quest sigue viendo las cámaras
```
Debe decir `WBC gains ...`, `WBC internal: loaded Balance + Walk ONNX` y `[locomanip] ready`.

**T2 · Brazos y manos con el Quest** (conda `tv`, igual que antes; **sin `--motion`** en sim)
```bash
cd ~/xr_teleoperate/teleop && conda activate tv
python teleop_hand_and_arm.py --sim --ee inspire_dfx --input-mode hand --display-mode immersive --img-server-ip 127.0.0.1 --frequency 60
```
En sim las piernas no salen de `rt/lowcmd`: la sim solo toma los brazos de ahí y las piernas son de la WBC.

**T3 · Caminar con el teclado** (conda `env_isaaclab`)
```bash
cd ~/unitree_sim_isaaclab && conda activate env_isaaclab && source grasp_bridge/setup_local_dds.sh
LOCO_HEIGHT=0.74 LOCO_CROUCH=0.2 python send_commands_keyboard.py
```
`w/s` adelante/atrás · `a/d` lateral · `z/x` girar · `c` agacharse · `q` sale.
Ojo: escucha **todo** el teclado (pynput), sin importar la ventana activa.

**T4 · Grabar episodios** (conda `env_isaaclab`)
```bash
cd ~/unitree_sim_isaaclab && conda activate env_isaaclab && source grasp_bridge/setup_local_dds.sh
python -m locomanip.cli keys pick_cylinder
```
`r` empieza (`pick_cylinder_001`, `_002`, …) · `f` guarda · `g` descarta · `Ctrl-C` sale.
Estas teclas no chocan con las de T3. Mientras manejas, deja el foco en T4 o en la ventana de Isaac:
en la terminal de T2, `s` y `q` también son teclas de `xr_teleoperate`.

## 2. Qué guarda cada episodio (`data/locomanip/<fecha>_<label>/`)

| Archivo | Contenido |
|---|---|
| `episode.npz` | cada paso de control (~50 Hz en tiempo de sim): pose y velocidad de la pelvis, 29 joints (q, dq, orden MuJoCo), 24 joints de manos, `cmd_vel` [vx, vy, wz, altura], objetivos de brazos de `rt/lowcmd`, comando de manos normalizado (0–1, como `rt/inspire/cmd`), pose del objeto, `t_sim` y `t_wall` |
| `motion_50hz.csv` | formato Bones-SEED (pelvis en cm, rotación en grados, 29 joints en grados), remuestreado a 50 Hz. Es lo que lee el conversor de SONIC |
| `meta.json` | tarea, label, duración, nombres de joints, offset x/y de la pelvis |

El episodio que esté grabando al cerrar la sim (Ctrl-C en T1) se guarda, no se pierde.

## 3. Exportar a SONIC
```bash
python -m locomanip.export data/locomanip --out data/locomanip_sonic
```
Junta los CSV en `data/locomanip_sonic/csv/` y corre `gear_sonic/data_process/convert_soma_csv_to_motion_lib.py`
→ `data/locomanip_sonic/locomanip_motion_lib.pkl` (50 fps). Ese PKL se mezcla con el dataset de SONIC en el fine-tune.

## 4. Cambios en este fork para que funcione

- **`--wbc_internal`** (PR upstream #127): la política de piernas que trae Unitree (`assets/model/policy.onnx`) no
  camina en esta escena; reacciona un instante a cada comando y se queda quieta (issues upstream #117 y #130).
  Con GR00T-WBC sí camina y gira. Los ONNX están en `assets/model/groot_wbc/`, copiados de
  `~/GR00T-WholeBodyControl/decoupled_wbc/sim2mujoco/resources/robots/g1/policy/`.
- **Ganancias PD de la WBC** en piernas y cintura (`_WBC_GAINS` en `action_provider_wh_dds.py`). Con las del asset
  de Isaac (tobillos kp 20 en vez de 40), Balance daba pasitos y derivaba ~0.1 m/s con comando cero. Con estas
  ganancias la deriva es 0.00 m.
- **`--render_interval`** respetado por el proveedor wholebody, y física en CPU por defecto: ~32 Hz de loop
  (~0.65× tiempo real) contra ~8 Hz antes. Medido con `WH_PROFILE=1` y solo la cámara frontal; con las 3 cámaras
  va algo más lento.
- **`send_commands_keyboard.py`**: `LOCO_HEIGHT` y `LOCO_CROUCH`. La WBC está parada a 0.74 m; el default de
  Unitree (0.8) sigue igual si no se definen.

## 5. Limitaciones conocidas

- **La sim corre por debajo del tiempo real (~0.65×).** El grabador usa tiempo de sim, así que la física es
  consistente, pero el movimiento de brazos del operador (hecho en tiempo real) queda ~1.5× más rápido en el clip.
  `t_wall` también se guarda.
- **Solo la escena del cilindro tiene versión wholebody con Inspire.** Las escenas de moneda, palito, pelota,
  cartas y contenedor son de base fija.
- **En el robot real el plan es distinto:** piernas con el control R3 de Unitree (`--motion`), manos BrainCo y
  pose de la pelvis estimada (odometría), no verdad de sim.

## 6. Robot real (piernas con R3, brazos con Quest, sin manos al inicio)

Grabador: `locomanip/real_recorder.py` (solo lee DDS; mismo formato de episodio, la pelvis viene de la
odometría `rt/odommodestate`, que deriva en episodios largos).

1. Cable USB-Ethernet de la laptop al robot. El adaptador necesita 192.168.123.222/24: los adaptadores
   `enx0c37969f434c` (Wired connection 3) y otros ya lo tienen; `enx764218d8fef2` (Wired connection 8) no.
2. Robot de pie con el controlador de Unitree en modo **Regular (R1+X)** y alguien con el R3. Si antes se corrió
   `dev_mode_on.py` (box), primero `python3 ~/mode_switch/dev_mode_off.py eth0 ai` en el Jetson.
3. Quest, laptop y Jetson en la misma Wi-Fi. En el Quest: abrir `https://<IP wlan0 del Jetson>:60001`, aceptar el
   certificado y presionar start (cámara de la cabeza); luego `https://<IP laptop>:8012/?ws=wss://<IP laptop>:8012`.
4. Teleop (conda `tv`), sin `--ee` = solo brazos:
   `python teleop_hand_and_arm.py --motion --input-mode hand --img-server-ip <IP Jetson> --network-interface <enx…>`
5. Grabador (conda `tv`): `python -m locomanip.real_recorder --iface <enx…> --label <tarea>` (r / f / g / q).
6. Exportar igual que en sim: `python -m locomanip.export data/locomanip_real --out data/locomanip_real_sonic`.

### 6b. Por Wi-Fi (recomendado): teleop y grabador corren en el Jetson

El Jetson está cableado a la red interna del robot, así que los comandos de brazos (`rt/arm_sdk`) nunca pasan por
Wi-Fi; solo el Quest ↔ Jetson (video WebRTC y poses de manos). La laptop solo entra por SSH.
En el Jetson: env `tv` (Python 3.10, pinocchio 3.1.0, numpy 1.26, versiones igual que la laptop), repo en
`~/xr_teleoperate` (copia del fork de la laptop) y grabador en `~/locomanip_rec`.

    ssh unitree@<IP wlan0 del Jetson>
    conda activate tv && cd ~/xr_teleoperate/teleop
    python teleop_hand_and_arm.py --motion --input-mode hand --img-server-ip <IP wlan0 del Jetson> --network-interface eth0

    ssh unitree@<IP wlan0 del Jetson>
    conda activate tv && cd ~/locomanip_rec
    python -m locomanip.real_recorder --iface eth0 --label prueba --out ~/locomanip_data

Quest: `https://<IP Jetson>:60001` (start) y luego `https://<IP Jetson>:8012/?ws=wss://<IP Jetson>:8012`.
Bajar episodios: `rsync -az unitree@<IP Jetson>:~/locomanip_data/ data/locomanip_real/`.

### 6c. Cámara de la cabeza en el grabador real

`real_recorder` guarda la cámara de la cabeza que publica `teleimager` (ZMQ `tcp://127.0.0.1:55555`, JPEG 640×480
a 30 fps), sin re-codificar: `head_cam.mjpeg` (JPEGs concatenados; `ffmpeg -f mjpeg -i head_cam.mjpeg out.mp4`) y
`head_cam_index.npz` (`t_wall`, `t_rel`, `offset`, `length` por cuadro, mismo reloj que `episode.npz`).
~3.3 MB/s (~200 MB por minuto). `--no-cam` la desactiva.

Después de **cada reinicio del Jetson** el módulo `video_hub_pc4` de Unitree vuelve a tomar la cámara y
`teleimager` no arranca: `conda activate tv && python ~/svc_switch.py video_hub_pc4 off` en el Jetson.

# Teleoperar el G1 real con el Meta Quest (y grabar tomas)

Guía completa, desde prender el robot. Piernas: el control **R3** de Unitree (modo Regular). Brazos y manos
BrainCo: **tus manos** con el Quest (hand tracking). Todo corre en el Jetson del robot; tu laptop solo entra por SSH
y el Quest se conecta por Wi-Fi. Los comandos de brazos nunca pasan por Wi-Fi.

> **Seguridad primero.** Siempre: una persona con el **R3 en la mano** lista para el combo de damping, el área
> despejada, y las primeras pruebas con el robot **colgado de la grúa** (pies tocando el piso). Empieza parado en el
> lugar, solo brazos; luego camina despacio.

---

## 0. Qué necesitas

- Robot G1 con batería, manos BrainCo montadas y su adaptador USB conectado al robot.
- Laptop con acceso SSH al Jetson (usuario `unitree`; la contraseña la tiene el equipo).
- Meta Quest 3 cargado, en la **misma Wi-Fi** que el Jetson.
- Las manos deben quedar con la **luz verde fija** en el dorso (verde parpadeando = todavía calibrando,
  amarillo = poco voltaje, rojo = error).

## 1. Prender el robot

1. Prende el G1 y espera a que el Jetson arranque (~1–2 min).
2. Con el R3, pon el robot de pie como siempre y entra a **modo Regular (R1+Y)**. *No* uses el modo Running.
3. Si antes alguien corrió la demo de box (`dev_mode_on.py`), primero regresa el control a Unitree en el Jetson:
   `conda activate g1deploy && python3 ~/mode_switch/dev_mode_off.py eth0 ai`

## 2. Entrar al Jetson

```bash
ssh unitree@<IP del Jetson>
```

- IP fija por ZeroTier (si tu laptop está en la misma red ZeroTier): `10.144.124.246`.
- IP en la Wi-Fi (la que usa el Quest). Ya dentro del Jetson:
  ```bash
  ip -4 -br a show wlan0
  ```
  Hasta ahora ha sido `10.22.129.75`. **Usa la IP de `wlan0` en todos los comandos y URLs de abajo** (aquí: `<IP>`).

## 3. Revisar servicios (Terminal A, en el Jetson)

**Cámara de la cabeza.** Después de **cada** encendido del Jetson, el módulo de video de Unitree le gana la cámara a
nuestro servidor de imágenes. Apágalo (solo deja de mandar video a la app de Unitree; no afecta caminar):

```bash
conda activate tv && python ~/svc_switch.py video_hub_pc4 off
```

Comprueba (deben salir `active` y los puertos 55555 y 60001; espera ~15 s si aún no aparecen):

```bash
systemctl is-active teleimager.service && ss -ltn | grep -E ":55555|:60001"
```

**Manos BrainCo.** Deben salir 2 líneas `bound` (left y right) y 2 `Starting worker` (slave 126 y 127):

```bash
journalctl -u brainco_hand.service -n 40 --no-pager | grep -E "bound|Starting worker"
```

Si falta una mano: espera a que su luz esté verde fija y reinicia el servicio (solo busca las manos al arrancar):

```bash
sudo systemctl restart brainco_hand.service
```

## 4. Teleop (Terminal A, en el Jetson)

```bash
cd ~/xr_teleoperate/teleop && python teleop_hand_and_arm.py --motion --input-mode hand --ee brainco --img-server-ip <IP> --network-interface eth0
```

Variantes:
- **Solo brazos** (sin mover las manos del robot): quita `--ee brainco`.
- **Ver el robot real a través del visor** (passthrough) en lugar del video de la cámara del robot:
  agrega `--display-mode pass-through`. Útil si quieres ver al robot directo sin quitarte el visor.

Debe terminar en `🟢 Press [r] to start syncing the robot with your movements.`

## 5. En el Meta Quest

1. Abre el navegador del Quest en `https://<IP>:60001` → **Avanzado → Continuar (no seguro)** → **start**.
   Debes ver la cámara del robot. (Solo hace falta la primera vez en ese Quest.)
2. Abre `https://<IP>:8012/?ws=wss://<IP>:8012` → **Avanzado → Continuar** → **Virtual Reality** → acepta permisos.
3. **Suelta los controles** (déjalos en la mesa) y levanta las manos hasta ver tus manos virtuales.
   En modo `hand`, si tienes los controles en la mano el Quest no rastrea tus manos y el robot no se mueve.
4. Pon brazos y manos en la pose del robot (manos abiertas) para que no haya un brinco al empezar.
5. Alguien aprieta **`r`** en la Terminal A. El robot empieza a seguir tus brazos y dedos.
6. El R3 sigue caminando al robot como siempre.

## 6. Grabar tomas (Terminal B, otra sesión SSH al Jetson)

```bash
ssh unitree@<IP>
```
```bash
conda activate tv && cd ~/locomanip_rec && python -m locomanip.real_recorder --iface eth0 --label <tarea> --out ~/locomanip_data
```

- `<tarea>`: nombre corto sin espacios ni acentos, una tarea por label (`agarrar_caja`, `saludar`, …).
  Para cambiar de tarea: `q` y vuelve a correr el comando con otro `--label` (el teleop no se reinicia).
- Al arrancar imprime `topics seen`; con el teleop activo deben aparecer `lowstate`, `odom`, `arm_sdk`,
  `left_state`, `right_state`, `left_cmd`, `right_cmd` y `head cam frames` > 0.

Teclas (en la Terminal B):

| Tecla | Acción |
|---|---|
| `r` | empieza una toma (`<tarea>_001`, `_002`, …) |
| `f` | guarda la toma |
| `g` | descarta la toma |
| `q` | sale (si estaba grabando, guarda primero) |

Buenas tomas: al apretar `r` quédate **quieto ~1 s con las manos abiertas**, haz la tarea, y `f`. Tomas de 20–40 s.

Cada toma queda en `~/locomanip_data/<fecha>_<tarea>_NNN/`:

| Archivo | Contenido |
|---|---|
| `episode.npz` | 50 Hz: 29 joints (posición, velocidad, torque), pelvis (odometría), IMU, sticks del R3, objetivos de brazos del teleop, estado y comando de las dos manos BrainCo (0 = abierta, 1 = cerrada) |
| `head_cam.mjpeg` | cámara de la cabeza: JPEG 640×480 a 30 fps (`ffmpeg -f mjpeg -i head_cam.mjpeg video.mp4`) |
| `head_cam_index.npz` | hora de cada cuadro (mismo reloj que `episode.npz`) y su posición en el `.mjpeg` |
| `motion_50hz.csv` | movimiento del cuerpo en el formato del conversor de SONIC |
| `meta.json` | label, duración, nombres de joints, cuadros de cámara y fps |

~3.3 MB por segundo de toma (casi todo es la cámara).

## 7. Terminar

1. **Terminal A:** `q` → los brazos regresan a su pose en ~5 s y el teleop se cierra.
2. **Terminal B:** `q`.
3. Con el R3, deja al robot en una pose segura antes de apagarlo.

## 8. Bajar y exportar las tomas (en tu laptop)

```bash
cd ~/unitree_sim_isaaclab && rsync -az unitree@<IP>:~/locomanip_data/ data/locomanip_real/
```
```bash
conda activate tv && python -m locomanip.export data/locomanip_real --out data/locomanip_real_sonic
```

Sale `data/locomanip_real_sonic/locomanip_motion_lib.pkl` (50 fps) para el fine-tune de SONIC.

---

## Problemas comunes

| Síntoma | Causa | Solución |
|---|---|---|
| Teleop: `Request to <IP>:60000 timed out` / el Quest no ve video | `teleimager` no tiene la cámara (el video de Unitree la tomó al encender) | `conda activate tv && python ~/svc_switch.py video_hub_pc4 off`, espera ~15 s, y vuelve a correr el teleop |
| `journalctl -u teleimager` dice `Cannot find OpenCVCamera` | mismo de arriba | mismo de arriba |
| Brainco: `No ttyUSB serial ports found` | adaptador USB de las manos desconectado | conéctalo; el servicio lo detecta solo en ≤5 s |
| Solo aparece una mano `bound` | esa mano no tenía corriente o seguía calibrando | luz verde fija y `sudo systemctl restart brainco_hand.service` |
| Los dedos del robot no se mueven | el Quest no está mandando tus manos (controles en la mano, manos fuera de vista) | suelta los controles, mira tus manos en VR, luego `r` |
| Revisar si llegan comandos a las manos | — | en otra terminal del Jetson: `conda activate tv && python ~/hand_watch.py` (20 s; `cmd Hz` ~100 y `cmd`/`state` cambiando = todo bien) |
| Traceback largo `Websocket session is missing` / `BrokenPipeError` en el teleop | el Quest recargó la página o entró/salió de VR | no es grave; se reconecta solo (`websocket is connected`) |
| El Quest se duerme si te lo quitas | sensor de proximidad | usa `--display-mode pass-through` y déjatelo puesto |
| Brazos del robot en una pose rara / no siguen | arrancaste sin alinear la pose o con el visor fuera de la cara | `q`, vuelve a correr el teleop, alinea la pose y `r` |

**Emergencia:** combo de damping en el R3. Ojo: con damping el robot pierde fuerza; si no está en la grúa, se cae.

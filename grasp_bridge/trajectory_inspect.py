"""Print diagnostics for a recorded trajectory."""

from __future__ import annotations

import numpy as np

from .trajectory_io import load_trajectory


def inspect_slot(slot: int) -> None:
    traj = load_trajectory(slot)
    hz = float(traj["hz"])
    n = int(traj["left_arm_q"].shape[0])
    body = traj.get("body_q")
    hand = traj["hand_q"]
    left = traj["left_arm_q"]
    right = traj["right_arm_q"]

    def pinch_side(s):
        # Inspire DDS: q=1 open, q=0 closed → closure = 1 - q
        return max(1.0 - s[:4].mean(), 1.0 - (s[3] + s[4]) / 2)

    h0 = hand[0]
    r0, l0 = h0[:6], h0[6:12]
    max_r = max(pinch_side(h[:6]) for h in hand)
    max_l = max(pinch_side(h[6:12]) for h in hand)

    print(f"=== Slot {slot}: {traj.get('label')} ===")
    src = traj.get("record_source", "")
    if traj.get("body_cmd") is not None:
        print(f"Record source: {src or 'body_cmd present'}")
    else:
        print("Record source: lowstate only (re-graba con recorder actualizado)")
    print(f"Frames: {n}  Duration: {n/hz:.1f}s @ {hz} Hz")
    print(f"Start L elbow: {left[0, 3]:.3f}  R elbow: {right[0, 3]:.3f}")
    print(f"Start hand closure R: {pinch_side(r0):.2f}  L: {pinch_side(l0):.2f}  (closed > 0.75)")
    print(f"Max closure during traj R: {max_r:.2f}  L: {max_l:.2f}")

    open_frames = sum(
        1 for h in hand if pinch_side(h[:6]) < 0.25 and pinch_side(h[6:12]) < 0.25
    )
    print(f"Frames with both hands open: {open_frames}/{n}")

    if body is not None:
        gap = float(np.max(np.abs(body[1] - body[0]))) if n > 1 else 0.0
        print(f"Max joint jump frame 0->1: {gap:.4f} rad")

    warnings = []
    if pinch_side(r0) > 0.4 or pinch_side(l0) > 0.4:
        warnings.append("Grabación empezó con manos ya cerradas — replay puede chocar.")
    if open_frames == 0:
        warnings.append("Nunca se abrieron las manos en la grabación.")
    if max_r < 0.6 and max_l < 0.6:
        warnings.append("Los dedos casi no cierran en la grabación — re-graba a 100 Hz con teleop activo.")
    if hz < 60:
        warnings.append(f"Grabado a {hz:.0f} Hz — usa --hz 100 al grabar (teleop manda brazos a 250 Hz).")
    if traj.get("body_cmd") is None:
        warnings.append("Grabación sin rt/lowcmd — el replay no puede igualar teleop. Re-graba.")

    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(f"  ⚠ {w}")
    else:
        print("\n✓ Grabación parece empezar desde pose neutral.")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Runner dla Deep-EIoU (tracking piłkarzy z drona).

Zastępuje ręczne klejenie komórek w Colabie. Robi dwie rzeczy:
  1. dociąga checkpointy (detektor + reid) jeśli ich nie ma,
  2. odpala tools/demo.py z sensownymi flagami i realnym fps.

Poprawki trackera (crop-filter, realny fps, tuning progów) są już WSZYTE
w tools/demo.py i tracker/Deep_EIoU.py — nic nie trzeba patchować.

Użycie (z katalogu Deep-EIoU/):
    python run.py --path stal2.mp4
    python run.py --path /content/stal10minut.mp4 --no-fuse
    python run.py --download-only          # tylko pobierz checkpointy
    python run.py --path X --python python3.7   # inny interpreter (Colab)

Wynik:
    - wideo z boxami: YOLOX_outputs/.../track_vis/<timestamp>/<nazwa>.mp4
    - plik MOT (frame,id,x,y,w,h,conf,-1,-1,-1): YOLOX_outputs/.../track_vis/<timestamp>.txt
Ten .txt jest wejściem do visualize_tracks.py i relink_tracks.py.
"""
import argparse
import os
import os.path as osp
import subprocess
import sys

HERE = osp.dirname(osp.abspath(__file__))
CKPT_DIR = osp.join(HERE, "checkpoints")

# (nazwa pliku docelowego, id z Google Drive) — te same, których używał Colab.
CHECKPOINTS = [
    ("best_ckpt.pth.tar", "1834kh10-X0Tu743fgmN7jXPVDKgq4ZqR"),        # detektor YOLOX-x (SportsMOT)
    ("sports_model.pth.tar-60", "14zzlm1nI9Ws_Il9RYNChwPC7Fsul7xwl"),  # model ReID (osnet)
]


def ensure_checkpoints():
    os.makedirs(CKPT_DIR, exist_ok=True)
    missing = [(name, gid) for name, gid in CHECKPOINTS
               if not osp.exists(osp.join(CKPT_DIR, name))]
    if not missing:
        print("[run] checkpointy obecne, pomijam pobieranie.")
        return
    try:
        import gdown  # noqa: F401
    except ImportError:
        sys.exit("[run] Brak pakietu 'gdown'. Zainstaluj: pip install gdown")
    import gdown
    for name, gid in missing:
        out = osp.join(CKPT_DIR, name)
        print(f"[run] pobieram {name} ...")
        gdown.download(id=gid, output=out, quiet=False)
        if not osp.exists(out):
            sys.exit(f"[run] Nie udało się pobrać {name}.")


def build_cmd(args):
    cmd = [args.python, osp.join("tools", "demo.py"),
           "--path", args.path,
           "--save_result", "True"]
    if args.fp16:
        cmd.append("--fp16")
    if args.fuse:
        cmd.append("--fuse")
    if args.tsize is not None:
        cmd += ["--tsize", str(args.tsize)]
    # progi trackera można nadpisać z CLI; domyślne (wytuningowane) siedzą w demo.py
    for flag in ("track_buffer", "match_thresh", "new_track_thresh"):
        val = getattr(args, flag)
        if val is not None:
            cmd += [f"--{flag}", str(val)]
    return cmd


def main():
    p = argparse.ArgumentParser("Deep-EIoU runner")
    p.add_argument("--path", help="ścieżka do wideo (mp4) lub folderu z klatkami")
    p.add_argument("--python", default=sys.executable or "python3",
                   help="interpreter do uruchomienia demo (w Colabie: python3.7)")
    p.add_argument("--download-only", action="store_true",
                   help="tylko pobierz checkpointy i wyjdź")
    p.add_argument("--no-fp16", dest="fp16", action="store_false", help="wyłącz FP16")
    p.add_argument("--no-fuse", dest="fuse", action="store_false", help="wyłącz fuse conv+bn")
    p.add_argument("--tsize", type=int, default=None, help="rozmiar wejścia detektora")
    # override tuningu (opcjonalnie); None => użyj domyślnych z demo.py
    p.add_argument("--track_buffer", type=int, default=None)
    p.add_argument("--match_thresh", type=float, default=None)
    p.add_argument("--new_track_thresh", type=float, default=None)
    p.set_defaults(fp16=True, fuse=True)
    args = p.parse_args()

    ensure_checkpoints()
    if args.download_only:
        return
    if not args.path:
        sys.exit("[run] Podaj --path do wideo (albo użyj --download-only).")

    cmd = build_cmd(args)
    print("[run] uruchamiam:", " ".join(cmd))
    # demo.py zakłada uruchomienie z katalogu Deep-EIoU/ (sys.path.append('.'))
    sys.exit(subprocess.call(cmd, cwd=HERE))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rysowanie wyników trackingu (plik MOT + wideo -> wideo z kolorowymi boxami).

Rozwinięcie Twojego untitled68.py: to samo (deterministyczny kolor per ID),
ale z argumentami CLI zamiast sztywnych ścieżek, i z opcją rysowania tylko
"długich" tracków (żeby zobaczyć realną liczbę graczy bez śmieciowych 1-klatkowych ID).

Format MOT (wyjście demo.py): frame, id, x, y, w, h, conf, -1, -1, -1

Użycie:
    python visualize_tracks.py --mot wynik.txt --video stal.mp4 --out out.mp4
    python visualize_tracks.py --mot wynik.txt --video stal.mp4 --out out.mp4 --min-len 15
"""
import argparse
from collections import defaultdict

import cv2
import numpy as np


def load_mot(mot_path):
    tracks_by_frame = defaultdict(list)
    track_len = defaultdict(int)
    with open(mot_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            frame_id = int(float(parts[0]))
            track_id = int(float(parts[1]))
            x, y, w, h = (float(parts[2]), float(parts[3]),
                          float(parts[4]), float(parts[5]))
            tracks_by_frame[frame_id].append((track_id, x, y, w, h))
            track_len[track_id] += 1
    return tracks_by_frame, track_len


def id_to_color(track_id):
    # deterministyczny kolor per ID — ten sam gracz zawsze ma ten sam kolor
    rng = np.random.RandomState(track_id)
    return tuple(int(c) for c in rng.randint(60, 255, size=3))


def main():
    p = argparse.ArgumentParser("visualize tracks")
    p.add_argument("--mot", required=True, help="plik MOT z demo.py")
    p.add_argument("--video", required=True, help="wejściowe wideo")
    p.add_argument("--out", default="output_tracked.mp4", help="wyjściowe wideo")
    p.add_argument("--min-len", type=int, default=1,
                   help="rysuj tylko tracki dłuższe niż N klatek (odsiew śmieci)")
    p.add_argument("--start-frame", type=int, default=1,
                   help="numer pierwszej klatki w MOT (demo.py numeruje od 1)")
    args = p.parse_args()

    tracks_by_frame, track_len = load_mot(args.mot)
    n_ids = len(track_len)
    kept_ids = {tid for tid, n in track_len.items() if n >= args.min_len}
    print(f"[viz] unikalnych ID: {n_ids} | po filtrze min-len={args.min_len}: {len(kept_ids)}")

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (width, height))

    frame_idx = args.start_frame - 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        for track_id, x, y, w, h in tracks_by_frame.get(frame_idx, []):
            if track_id not in kept_ids:
                continue
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
            color = id_to_color(track_id)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"ID {track_id}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        writer.write(frame)

    cap.release()
    writer.release()
    print(f"[viz] zapisano: {args.out}, klatek: {frame_idx - (args.start_frame - 1)}")


if __name__ == "__main__":
    main()

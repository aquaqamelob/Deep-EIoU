#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Offline re-linking fragmentów tracków po kolorze koszulki (redukcja liczby ID).

To jest "Faza 2 lite": nie robi jeszcze SigLIP+UMAP i NIE wymusza sztywnych 22.
Zamiast tego skleja fragmenty, które są ewidentnie tym samym graczem:
  - rozłączne w czasie (koniec A przed startem B, luka <= max-gap),
  - ciągłe przestrzennie (start B blisko przewidzianej pozycji końca A),
  - podobne kolorem koszulki (mediana Lab z rejonu tułowia, po odsianiu murawy).

Sklejanie: greedy po najtańszych krawędziach + union-find (każdy "ogon" i "głowa"
tracku użyte co najwyżej raz), więc powstają łańcuchy A->B->C bez nakładania w czasie.

Dodatkowo (opcjonalnie) klasteryzuje finalne ID na drużyny po kolorze (k-means, k=2..3)
i zapisuje mapę id->team — przydatne jako sanity-check i wstęp do Fazy 2.

Wejście: plik MOT z demo.py (frame,id,x,y,w,h,conf,-1,-1,-1) + to samo wideo.
Wyjście: nowy plik MOT ze zremapowanymi ID (mniej unikalnych ID).

Użycie:
    python relink_tracks.py --mot wynik.txt --video stal.mp4 --out wynik_relinked.txt
    python relink_tracks.py --mot wynik.txt --video stal.mp4 --out wynik_relinked.txt \
        --max-gap 90 --color-thresh 22 --teams 2
Potem:
    python visualize_tracks.py --mot wynik_relinked.txt --video stal.mp4 --out out.mp4
"""
import argparse
from collections import defaultdict

import cv2
import numpy as np


# ----------------------------- I/O -----------------------------------------

def load_mot(mot_path):
    """Zwraca (rows, per_track) gdzie rows to surowe wiersze do przepisania."""
    rows = []                                   # (frame, id, x, y, w, h, tail_str)
    per_track = defaultdict(list)               # id -> list of (frame, x, y, w, h)
    with open(mot_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 6:
                continue
            frame = int(float(parts[0]))
            tid = int(float(parts[1]))
            x, y, w, h = (float(parts[2]), float(parts[3]),
                          float(parts[4]), float(parts[5]))
            tail = ",".join(parts[6:]) if len(parts) > 6 else "-1,-1,-1"
            rows.append((frame, tid, x, y, w, h, tail))
            per_track[tid].append((frame, x, y, w, h))
    for tid in per_track:
        per_track[tid].sort(key=lambda r: r[0])
    return rows, per_track


# ----------------------- cechy koloru per track ----------------------------

def torso_region(x, y, w, h, W, H):
    """Prostokąt tułowia w obrębie boxa, przycięty do obrazu."""
    x1 = int(x + 0.20 * w); x2 = int(x + 0.80 * w)
    y1 = int(y + 0.15 * h); y2 = int(y + 0.50 * h)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    return x1, y1, x2, y2


def crop_median_lab(frame, box, W, H):
    """Mediana Lab tułowia po odsianiu pikseli murawy (zielone w HSV)."""
    x1, y1, x2, y2 = torso_region(*box, W=W, H=H)
    if x2 <= x1 or y2 <= y1:
        return None
    patch = frame[y1:y2, x1:x2]
    if patch.size == 0:
        return None
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    h = hsv[..., 0]; s = hsv[..., 1]
    # murawa: hue ~ zielony (35..85 w skali OpenCV 0..180) i wystarczająco nasycona
    grass = (h >= 35) & (h <= 85) & (s >= 60)
    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2Lab).reshape(-1, 3).astype(np.float32)
    keep = ~grass.reshape(-1)
    sel = lab[keep] if keep.sum() >= 10 else lab
    return np.median(sel, axis=0)


def track_features(per_track, video_path, samples_per_track=12):
    """Dla każdego tracku liczy medianę Lab + geometrię (start/end, centra, prędkość)."""
    cap = cv2.VideoCapture(video_path)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # które klatki próbkować dla którego tracku
    frame_to_samples = defaultdict(list)   # frame -> [(tid, box), ...]
    for tid, recs in per_track.items():
        if len(recs) <= samples_per_track:
            idxs = range(len(recs))
        else:
            idxs = np.linspace(0, len(recs) - 1, samples_per_track).astype(int)
        for i in idxs:
            fr, x, y, w, h = recs[i]
            frame_to_samples[fr].append((tid, (x, y, w, h)))

    lab_samples = defaultdict(list)
    if frame_to_samples:
        max_frame = max(frame_to_samples)
        frame_idx = 0
        while frame_idx <= max_frame:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1                      # MOT numeruje od 1
            for tid, box in frame_to_samples.get(frame_idx, []):
                lab = crop_median_lab(frame, box, W, H)
                if lab is not None:
                    lab_samples[tid].append(lab)
    cap.release()

    feats = {}
    for tid, recs in per_track.items():
        color = (np.median(np.stack(lab_samples[tid]), axis=0)
                 if lab_samples.get(tid) else None)

        def center(rec):
            _, x, y, w, h = rec
            return np.array([x + w / 2.0, y + h / 2.0])

        start_c, end_c = center(recs[0]), center(recs[-1])
        vel = np.zeros(2)
        if len(recs) >= 2:
            dt = max(1, recs[-1][0] - recs[-2][0])
            vel = (center(recs[-1]) - center(recs[-2])) / dt
        avg_h = float(np.mean([r[4] for r in recs]))
        feats[tid] = dict(
            start_frame=recs[0][0], end_frame=recs[-1][0],
            start_c=start_c, end_c=end_c, vel=vel,
            color=color, avg_h=avg_h, n=len(recs),
        )
    return feats


# --------------------------- sklejanie -------------------------------------

class UnionFind:
    def __init__(self, ids):
        self.p = {i: i for i in ids}

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def build_links(feats, max_gap, color_thresh, dist_base, dist_per_frame):
    """Zwraca listę par (cost, a, b): a kończy się przed b, sklejalne."""
    ids = list(feats.keys())
    edges = []
    for a in ids:
        fa = feats[a]
        for b in ids:
            if a == b:
                continue
            fb = feats[b]
            gap = fb["start_frame"] - fa["end_frame"]
            if gap < 1 or gap > max_gap:
                continue                        # musi być rozłączne w czasie i blisko
            # predykcja pozycji A w chwili startu B
            pred = fa["end_c"] + fa["vel"] * gap
            dist = float(np.linalg.norm(fb["start_c"] - pred))
            # dozwolony promień rośnie z luką i skalą gracza (wysokość boxa)
            scale = 0.5 * (fa["avg_h"] + fb["avg_h"]) / 62.0    # ~1.0 dla ~62px boxów
            max_dist = (dist_base + dist_per_frame * gap) * max(0.5, scale)
            if dist > max_dist:
                continue
            if fa["color"] is None or fb["color"] is None:
                continue                        # bez koloru nie ryzykujemy sklejenia
            cdist = float(np.linalg.norm(fa["color"] - fb["color"]))
            if cdist > color_thresh:
                continue
            cost = cdist / color_thresh + dist / max_dist       # w [0, 2]
            edges.append((cost, a, b))
    edges.sort(key=lambda e: e[0])
    return edges


def chain(feats, edges):
    """Greedy: każdy ogon (a) i każda głowa (b) użyte raz -> łańcuchy."""
    uf = UnionFind(feats.keys())
    used_tail, used_head = set(), set()
    for cost, a, b in edges:
        if a in used_tail or b in used_head:
            continue
        if uf.find(a) == uf.find(b):
            continue
        uf.union(a, b)
        used_tail.add(a)
        used_head.add(b)
    # remap: reprezentant -> kolejny mały numer (od 1)
    reps = {}
    remap = {}
    for tid in feats:
        r = uf.find(tid)
        if r not in reps:
            reps[r] = len(reps) + 1
        remap[tid] = reps[r]
    return remap


# --------------------------- drużyny (opcja) -------------------------------

def cluster_teams(feats, remap, k):
    """K-means po medianie Lab finalnych ID -> mapa final_id -> team (0..k-1)."""
    final_colors = {}
    for tid, fid in remap.items():
        c = feats[tid]["color"]
        if c is not None:
            final_colors.setdefault(fid, []).append(c)
    fids = [f for f in final_colors]
    if len(fids) < k:
        return {f: 0 for f in fids}
    data = np.stack([np.median(np.stack(final_colors[f]), axis=0)
                     for f in fids]).astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, _ = cv2.kmeans(data, k, None, crit, 5, cv2.KMEANS_PP_CENTERS)
    return {f: int(labels[i][0]) for i, f in enumerate(fids)}


# ------------------------------- main --------------------------------------

def write_mot(out_path, rows, remap):
    fmt = "{frame},{id},{x},{y},{w},{h},{tail}\n"
    with open(out_path, "w") as f:
        for frame, tid, x, y, w, h, tail in rows:
            f.write(fmt.format(frame=frame, id=remap.get(tid, tid),
                               x=round(x, 1), y=round(y, 1),
                               w=round(w, 1), h=round(h, 1), tail=tail))


def main():
    p = argparse.ArgumentParser("re-linking tracków po kolorze")
    p.add_argument("--mot", required=True, help="plik MOT z demo.py")
    p.add_argument("--video", required=True, help="to samo wideo")
    p.add_argument("--out", required=True, help="wyjściowy plik MOT (zremapowane ID)")
    p.add_argument("--max-gap", type=int, default=90,
                   help="maks. luka czasowa (klatki) do sklejenia dwóch fragmentów")
    p.add_argument("--color-thresh", type=float, default=22.0,
                   help="maks. odległość Lab koloru koszulki (mniejszy = ostrożniej)")
    p.add_argument("--dist-base", type=float, default=60.0,
                   help="bazowy promień dopuszczalnego skoku pozycji (px)")
    p.add_argument("--dist-per-frame", type=float, default=8.0,
                   help="ile px/klatkę doliczyć do promienia za każdą klatkę luki")
    p.add_argument("--min-len", type=int, default=1,
                   help="pomiń w statystykach tracki krótsze niż N klatek")
    p.add_argument("--teams", type=int, default=0,
                   help="jeśli >0: klasteryzuj finalne ID na tyle drużyn i zapisz mapę")
    p.add_argument("--samples", type=int, default=12,
                   help="ile klatek próbkować na track do koloru")
    args = p.parse_args()

    rows, per_track = load_mot(args.mot)
    n_before = len(per_track)
    print(f"[relink] ID przed: {n_before}")

    feats = track_features(per_track, args.video, samples_per_track=args.samples)
    edges = build_links(feats, args.max_gap, args.color_thresh,
                        args.dist_base, args.dist_per_frame)
    remap = chain(feats, edges)
    n_after = len(set(remap.values()))
    print(f"[relink] sklejono {len(edges)} kandydatów -> ID po: {n_after} "
          f"(redukcja {n_before} -> {n_after})")

    write_mot(args.out, rows, remap)
    print(f"[relink] zapisano MOT: {args.out}")

    if args.teams > 0:
        team_map = cluster_teams(feats, remap, args.teams)
        team_path = args.out.rsplit(".", 1)[0] + "_teams.csv"
        with open(team_path, "w") as f:
            f.write("final_id,team\n")
            for fid in sorted(team_map):
                f.write(f"{fid},{team_map[fid]}\n")
        sizes = defaultdict(int)
        for t in team_map.values():
            sizes[t] += 1
        print(f"[relink] drużyny (k={args.teams}) liczności: {dict(sizes)} -> {team_path}")


if __name__ == "__main__":
    main()

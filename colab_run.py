# -*- coding: utf-8 -*-
"""colab_run.py — gotowiec do wklejenia do Google Colab.

To NIE jest skrypt do odpalania lokalnie (ma magie Colaba: !komendy i %cd).
Kopiuj kolejne "CELE" do osobnych komórek w Colabie i odpalaj po kolei.

Klonuje TWÓJ fork (aquaqamelob/Deep-EIoU) z brancha z tuningiem trackera,
wszytym crop-fixem i skryptami do redukcji ID kolorem.
Branch: claude/soccer-tracking-id-fragmentation-ba78ab
"""

# =====================================================================
# CELA 1 — setup + tracking  (wklej do PIERWSZEJ komórki)
# =====================================================================
!sudo apt-get install python3.7 python3.7-dev python3.7-distutils -y
!wget https://bootstrap.pypa.io/pip/3.7/get-pip.py
!python3.7 get-pip.py

!python3.7 -m pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 \
torchaudio==0.13.0 --index-url https://download.pytorch.org/whl/cu116

!python3.7 -m pip install cython_bbox gdown \
https://github.com/KaiyangZhou/deep-person-reid/archive/master.zip

# TWÓJ fork + branch ze zmianami (tuning, crop-fix, skrypty do kolorów)
!git clone -b claude/soccer-tracking-id-fragmentation-ba78ab \
https://github.com/aquaqamelob/Deep-EIoU.git

%cd /content/Deep-EIoU/Deep-EIoU/reid
!python3.7 -m pip install -r requirements.txt

%cd /content/Deep-EIoU/Deep-EIoU
!pip install cython_bbox

!gdown --fuzzy 'https://drive.google.com/file/d/1DQj5-kiU4VySymDXZuV84-nU1AZ4HZaj/view?usp=drive_link'
!gdown --fuzzy 'https://drive.google.com/file/d/1834kh10-X0Tu743fgmN7jXPVDKgq4ZqR/view?usp=drive_link' --output checkpoints/best_ckpt.pth.tar
!gdown --fuzzy 'https://drive.google.com/file/d/14zzlm1nI9Ws_Il9RYNChwPC7Fsul7xwl/view?usp=drive_link' --output checkpoints/sports_model.pth.tar-60

!python3.7 -m pip install https://github.com/KaiyangZhou/deep-person-reid/archive/master.zip

# --- wejściowe wideo: upload do /content ALBO zamontuj Drive ---
VIDEO = "/content/stal2.mp4"     # <- ustaw swoją ścieżkę
# from google.colab import drive; drive.mount('/content/drive')
# VIDEO = "/content/drive/MyDrive/stal/stal10minut.mp4"

# --- tracking (wytuningowane progi są już domyślne w demo.py) ---
!python3.7 tools/demo.py --path {VIDEO} --fp16 --fuse --save_result True


# =====================================================================
# CELA 2 — redukcja ID kolorem + wizualizacja  (wklej do DRUGIEJ komórki)
# =====================================================================
import glob, os
mot = max(glob.glob('YOLOX_outputs/**/track_vis/*.txt', recursive=True),
          key=os.path.getmtime)
print("surowy MOT:", mot)
print("surowych ID:", len(set(l.split(',')[1] for l in open(mot))))

# sklejanie fragmentów po kolorze koszulki -> mniej ID (nie wymusza 22)
!python3.7 relink_tracks.py --mot {mot} --video {VIDEO} \
    --out relinked.txt --max-gap 90 --color-thresh 22 --teams 3

# render: kolorowe boxy, tylko tracki > 15 klatek (odsiew śmieci)
!python3.7 visualize_tracks.py --mot relinked.txt --video {VIDEO} \
    --out out.mp4 --min-len 15

from google.colab import files
files.download('out.mp4')


# =====================================================================
# POKRĘTŁA (gdy wynik nie zadowala) — podmień flagi w CELI 2:
#   za mało sklejeń        -> --max-gap 150 --color-thresh 28
#   skleja różnych graczy  -> --max-gap 60  --color-thresh 16
#   dużo migających ID     -> --min-len 25   (tylko filtr rysowania)
#
# UWAGA: przy ponownym uruchomieniu CELI 1 klon się nie nadpisze.
#   Wtedy najpierw:  !rm -rf /content/Deep-EIoU   (albo Runtime -> Restart)
# =====================================================================

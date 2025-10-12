# split_subset.py
import os, shutil, random
from pathlib import Path

SRC = "dataset"  # existing folder with A, B, C ... (your dataset)
DST = "dataset_subset"
CLASSES = ['A','B','C','F','K','Y']  # change this list if you want other letters
RATIOS = (0.7, 0.15, 0.15)  # train, val, test
SEED = 42

def split():
    random.seed(SEED)
    Path(DST).mkdir(parents=True, exist_ok=True)
    for split in ("train","val","test"):
        Path(os.path.join(DST, split)).mkdir(parents=True, exist_ok=True)

    for cls in CLASSES:
        src_cls = os.path.join(SRC, cls)
        if not os.path.isdir(src_cls):
            print(f"Warning: {src_cls} not found, skipping.")
            continue
        imgs = [f for f in os.listdir(src_cls) if f.lower().endswith(('.jpg','.png','.jpeg'))]
        random.shuffle(imgs)
        n = len(imgs)
        t = int(n * RATIOS[0])
        v = int(n * RATIOS[1])
        splits = {
            'train': imgs[:t],
            'val': imgs[t:t+v],
            'test': imgs[t+v:]
        }
        for split_name, files in splits.items():
            dst_dir = os.path.join(DST, split_name, cls)
            os.makedirs(dst_dir, exist_ok=True)
            for f in files:
                shutil.copy(os.path.join(src_cls, f), os.path.join(dst_dir, f))
        print(f"Copied {cls}: total {n}, train {len(splits['train'])}, val {len(splits['val'])}, test {len(splits['test'])}")

if __name__ == "__main__":
    split()
    print("Done.")
